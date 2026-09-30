"""PDF -> per-page text, with OCR fallback and a per-page quality signal.

Most IFW PDFs are scanned images (TIFF-derived) with no text layer, but newer
e-filed documents often carry real text. We use the embedded text when it is
substantial and only rasterise + OCR when it is not. Every page records *how* its
text was obtained and, for OCR pages, the mean word confidence. That confidence
feeds the grounding evaluator: a quote that fails to verify on a 45%-confidence
page is a different problem from one that fails on clean text.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf
from PIL import Image
from pydantic import BaseModel

from .config import Settings

log = logging.getLogger(__name__)


class PageText(BaseModel):
    page: int                   # 1-based page number within the document
    text: str
    method: str                 # "text-layer" | "ocr"
    confidence: float | None    # mean Tesseract word confidence (0-100), None for text layer

    def is_low_quality(self, threshold: float) -> bool:
        return self.method == "ocr" and (self.confidence or 0) < threshold


class DocText(BaseModel):
    doc_id: str
    pages: list[PageText]

    def page(self, n: int) -> PageText | None:
        return next((p for p in self.pages if p.page == n), None)

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)


def _ocr_image(img: Image.Image, tesseract_cmd: str | None) -> tuple[str, float]:
    import pytesseract

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    data = pytesseract.image_to_data(
        img, config="--psm 6", output_type=pytesseract.Output.DICT
    )
    confs = [float(c) for c, w in zip(data["conf"], data["text"], strict=False)
             if w.strip() and float(c) >= 0]
    # Rebuild text line-by-line from the same pass so text and confidence agree.
    lines: dict[tuple[int, int, int], list[str]] = {}
    for i, word in enumerate(data["text"]):
        if word.strip():
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
    text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
    return text, (sum(confs) / len(confs) if confs else 0.0)


def _process_page(pdf_path: Path, idx: int, settings: Settings) -> PageText:
    with pymupdf.open(pdf_path) as doc:
        page = doc[idx]
        layer = page.get_text("text") or ""
        if len(layer.strip()) >= settings.min_text_layer_chars:
            return PageText(page=idx + 1, text=clean_text(layer), method="text-layer",
                            confidence=None)
        pix = page.get_pixmap(dpi=settings.ocr_dpi, colorspace=pymupdf.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    text, conf = _ocr_image(img, settings.tesseract_cmd)
    return PageText(page=idx + 1, text=clean_text(text), method="ocr", confidence=round(conf, 1))


def extract_document(pdf_path: Path, doc_id: str, settings: Settings,
                     cache_dir: Path | None = None, workers: int = 4) -> DocText:
    """OCR/extract one PDF, caching the result as JSON next to the PDF."""
    cache = (cache_dir or pdf_path.parent.parent / "text") / f"{doc_id}.json"
    if cache.exists():
        return DocText.model_validate_json(cache.read_text(encoding="utf-8"))
    with pymupdf.open(pdf_path) as doc:
        n = doc.page_count
    # Tesseract runs as a subprocess, so threads give real parallelism here.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        pages = list(ex.map(lambda i: _process_page(pdf_path, i, settings), range(n)))
    result = DocText(doc_id=doc_id, pages=pages)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result.model_dump(), indent=1), encoding="utf-8")
    return result


# ---------------------------------------------------------------- cleanup

_BOILERPLATE = [
    re.compile(r"^\s*Page \d+\s*$", re.I | re.M),
    re.compile(r"^\s*Application/Control Number:\s*\S+\s*$", re.I | re.M),
    re.compile(r"^\s*Art Unit:\s*\d+\s*$", re.I | re.M),
]


def clean_text(text: str) -> str:
    """Light, reversible-in-spirit cleanup. We do NOT paraphrase or reflow sentences,
    because the grounding step needs model quotes to match this text."""
    t = text.replace("­", "")                 # soft hyphens
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)          # de-hyphenate line breaks
    t = t.replace("“", '"').replace("”", '"').replace("’", "'")
    for pat in _BOILERPLATE:
        t = pat.sub("", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def render_for_llm(doc_id: str, label: str, text: DocText, max_chars: int | None = None) -> str:
    """Serialise a document with explicit page markers the model must cite."""
    parts = [f'<document id="{doc_id}" label="{label}">']
    used = 0
    for p in text.pages:
        block = f'<page n="{p.page}">\n{p.text}\n</page>'
        if max_chars and used + len(block) > max_chars:
            parts.append(f"<!-- truncated after page {p.page - 1} -->")
            break
        parts.append(block)
        used += len(block)
    parts.append("</document>")
    return "\n".join(parts)
