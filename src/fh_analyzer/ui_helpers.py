"""Small helpers shared by the two Streamlit apps (app.py and label_app.py)."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from rapidfuzz import fuzz


def highlight(text: str, needle: str | None) -> str:
    """HTML-escape page text and wrap the best match for `needle` in <mark>.
    Tries an exact (whitespace-tolerant) match first, then a fuzzy alignment, since
    quotes and OCR text often differ by a character or two."""
    if not needle or not needle.strip():
        return html.escape(text).replace("\n", "<br>")
    words = [re.escape(w) for w in needle.split()[:60]]
    m = re.search(r"\s+".join(words), text, re.I)
    if m:
        span = (m.start(), m.end())
    else:
        al = fuzz.partial_ratio_alignment(needle.lower(), text.lower())
        span = (al.dest_start, al.dest_end) if al and al.score >= 80 else None
    if not span:
        return html.escape(text).replace("\n", "<br>")
    a, b = span
    out = (html.escape(text[:a]) + '<mark class="hit">' + html.escape(text[a:b]) + "</mark>"
           + html.escape(text[b:]))
    return out.replace("\n", "<br>")


def page_image(pdf_dir: str | None, doc_id: str, page: int, dpi: int = 110) -> bytes | None:
    if not pdf_dir:
        return None
    pdf = Path(pdf_dir) / f"{doc_id}.pdf"
    if not pdf.exists():
        return None
    import pymupdf

    with pymupdf.open(pdf) as d:
        if not 1 <= page <= d.page_count:
            return None
        return d[page - 1].get_pixmap(dpi=dpi).tobytes("png")


def fmt_patent(n: str | None) -> str | None:
    if not n:
        return None
    return f"US {int(n):,}" if n.isdigit() else f"US {n}"


def fmt_app(n: str) -> str:
    return f"{n[:2]}/{n[2:5]},{n[5:]}" if len(n) == 8 and n.isdigit() else n


def analysis_label(path: str) -> str:
    """'US 1,234,567 · App. 12/345,678 · Title' for a saved analysis.json."""
    try:
        with open(path, encoding="utf-8") as fh:
            app_info = json.load(fh).get("app", {})
    except (OSError, ValueError):
        return path
    parts = [fmt_patent(app_info.get("patent_number")),
             f"App. {fmt_app(app_info.get('application_number', '?'))}"]
    title = (app_info.get("title") or "").strip()
    if title:
        parts.append(title if len(title) <= 60 else title[:57] + "…")
    return " · ".join(p for p in parts if p)


# ------------------------------------------------------------------ reference links

def google_patents_url(key: str) -> str:
    """Google Patents page for a kind-less match key from family.match_key
    ('US9123456', 'US2005123456', 'EP1234567', ...)."""
    m = re.fullmatch(r"US(20\d\d)(\d{6})", key)
    if m:                                   # docdb pre-grant pub -> US20050123456
        key = f"US{m.group(1)}0{m.group(2)}"
    return f"https://patents.google.com/patent/{key}"


def _ref_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, match key) for each patent number written in `text`, no overlaps."""
    from .family import _REF_PATTERNS, match_key

    spans = []
    for pat in _REF_PATTERNS:
        for m in pat.finditer(text or ""):
            num = re.sub(r"[\s,.\-/]", "", m.group(2))
            if len(num) >= 5:
                spans.append((m.start(), m.end(), match_key((m.group(1) or "US").upper(), num)))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    out, end = [], -1
    for s in spans:
        if s[0] >= end:
            out.append(s)
            end = s[1]
    return out


def linkify_patents(text: str) -> str:
    """Markdown with every patent number in `text` linked to Google Patents."""
    parts, pos = [], 0
    for start, end, key in _ref_spans(text):
        label = text[start:end].strip()
        parts += [text[pos:start], f"[{label}]({google_patents_url(key)})"]
        pos = end
    return "".join(parts) + (text or "")[pos:]


def reference_url(ref: str) -> str | None:
    """Google Patents URL for a reference like 'Smith (US 9,123,456)'; None if it names
    no patent number (e.g. 'Smith et al.' or non-patent literature)."""
    spans = _ref_spans(ref)
    return google_patents_url(spans[0][2]) if spans else None
