"""Citation grounding: does the evidence the model cited actually exist where it says?

Three layers, cheapest first:

1. **Quote verification (deterministic).** Each finding's quote is fuzzy-matched against
   the OCR text of the cited doc/page. Fuzzy, because the page text is OCR output and a
   model copying it may normalise a stray character; a threshold of 90 tolerates that
   while still rejecting paraphrase. If the quote is not on the cited page we search the
   rest of that document, which separates "real text, wrong pin cite" from "invented".
2. **Support check (optional, LLM-as-judge).** A verified quote proves the text exists,
   not that it supports the model's statement. A second, independent prompt is shown only
   the statement and the surrounding page text and asked whether it is supported.
3. **Synthesis traceability.** Every point in the overview must reference finding ids that
   exist; dangling or missing references are counted.

Nothing here decides whether the *legal analysis* is right. It measures whether the
output is anchored to the record so a human reviewer can check it quickly, and it
tells the reviewer where to look first (unverified, wrong-page, and low-OCR items).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Literal

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from .models import Citation
from .ocr import DocText

Status = Literal["verified", "partial", "wrong_page", "unverified", "bad_reference", "uncited"]

VERIFIED_AT = 90.0
PARTIAL_AT = 75.0
MIN_QUOTE_WORDS = 5


class Finding(BaseModel):
    id: str
    kind: str                     # rejection | amendment | estoppel | claim | allowance | ...
    doc_id: str
    doc_label: str
    date: str | None
    statement: str
    citation: Citation
    meta: dict = Field(default_factory=dict)


class CitationCheck(BaseModel):
    finding_id: str
    status: Status
    score: float
    matched_page: int | None = None
    matched_text: str | None = None
    short_quote: bool = False
    page_ocr_confidence: float | None = None
    low_quality_page: bool = False
    support: Literal["supported", "partial", "unsupported"] | None = None
    support_reason: str | None = None


class GroundingReport(BaseModel):
    checks: dict[str, CitationCheck]
    totals: dict[str, int]
    rate_verified: float
    rate_grounded: float           # verified + partial + wrong_page (text exists somewhere)
    by_kind: dict[str, dict[str, int]]
    by_doc: dict[str, dict[str, int]]
    low_quality_share_of_failures: float | None
    support_totals: dict[str, int] | None = None
    synthesis: dict | None = None


# ------------------------------------------------------------------ normalisation

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).translate(_DASHES)
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace(
        "’", "'")
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def _best_match(quote: str, text: str) -> tuple[float, str | None]:
    q, t = normalize(quote), normalize(text)
    if not q or not t:
        return 0.0, None
    if q in t:
        return 100.0, quote
    al = fuzz.partial_ratio_alignment(q, t)
    if al is None:
        return 0.0, None
    return al.score, t[al.dest_start:al.dest_end]


# ------------------------------------------------------------------ layer 1

def check_citation(finding: Finding, texts: dict[str, DocText],
                   low_conf_threshold: float = 70.0) -> CitationCheck:
    c = finding.citation
    if not c.quote.strip():
        # The model gave a statement with no evidence at all.
        return CitationCheck(finding_id=finding.id, status="uncited", score=0.0)
    short = len(c.quote.split()) < MIN_QUOTE_WORDS
    doc = texts.get(c.doc_id)
    page = doc.page(c.page) if doc else None
    if doc is None or page is None:
        # The model cited a document/page that is not in the record it was given.
        return CitationCheck(finding_id=finding.id, status="bad_reference", score=0.0,
                             short_quote=short)

    score, matched = _best_match(c.quote, page.text)
    common = dict(
        finding_id=finding.id, short_quote=short, page_ocr_confidence=page.confidence,
        low_quality_page=page.is_low_quality(low_conf_threshold),
    )
    if score >= VERIFIED_AT:
        return CitationCheck(status="verified", score=score, matched_page=c.page,
                             matched_text=matched, **common)

    # Not on the cited page (well enough) -> is it elsewhere in the same document?
    best = (score, c.page, matched)
    for p in doc.pages:
        if p.page == c.page:
            continue
        s, m = _best_match(c.quote, p.text)
        if s > best[0]:
            best = (s, p.page, m)
    if best[1] != c.page and best[0] >= VERIFIED_AT:
        return CitationCheck(status="wrong_page", score=best[0], matched_page=best[1],
                             matched_text=best[2], **common)
    if score >= PARTIAL_AT:
        return CitationCheck(status="partial", score=score, matched_page=c.page,
                             matched_text=matched, **common)
    return CitationCheck(status="unverified", score=best[0], matched_page=None, **common)


# ------------------------------------------------------------------ layer 2 (optional)

class _Verdict(BaseModel):
    id: str
    support: Literal["supported", "partial", "unsupported"]
    reason: str = Field(description="One sentence.")


class _Verdicts(BaseModel):
    verdicts: list[_Verdict]


JUDGE_PROMPT = """\
You are auditing another analyst's notes on a patent file history. For each item you get a
STATEMENT and the SOURCE page text it cites. Decide whether the source supports the
statement: "supported" (fully), "partial" (some of it, or overstated), or "unsupported".
Judge only against the source text shown; do not use outside knowledge. Be strict about
overstatement, e.g. calling an ordinary argument a "clear disavowal".
"""


def judge_support(findings: list[Finding], checks: dict[str, CitationCheck],
                  texts: dict[str, DocText], llm, batch: int = 15) -> None:
    """Fill `support` on checks, in place. Only judges findings whose text was located."""
    todo = [f for f in findings
            if checks[f.id].status in ("verified", "partial", "wrong_page")]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        items = []
        for f in chunk:
            pg = texts[f.citation.doc_id].page(checks[f.id].matched_page or f.citation.page)
            items.append(
                f'<item id="{f.id}">\n<statement>{f.statement}</statement>\n'
                f'<source doc="{f.citation.doc_id}" page="{pg.page}">\n{pg.text[:6000]}\n'
                f"</source>\n</item>"
            )
        out = llm.extract(JUDGE_PROMPT, "\n\n".join(items), _Verdicts)
        for v in out.verdicts:
            if v.id in checks:
                checks[v.id].support = v.support
                checks[v.id].support_reason = v.reason


# ------------------------------------------------------------------ report

def build_report(findings: list[Finding], checks: dict[str, CitationCheck],
                 synthesis=None) -> GroundingReport:
    totals = Counter(c.status for c in checks.values())
    n = max(len(checks), 1)
    by_kind: dict[str, Counter] = defaultdict(Counter)
    by_doc: dict[str, Counter] = defaultdict(Counter)
    for f in findings:
        st = checks[f.id].status
        by_kind[f.kind][st] += 1
        by_doc[f.doc_label][st] += 1

    failures = [c for c in checks.values() if c.status in ("unverified", "partial")]
    lq = (sum(c.low_quality_page for c in failures) / len(failures)) if failures else None

    support = Counter(c.support for c in checks.values() if c.support)

    syn = None
    if synthesis is not None:
        ids = {f.id for f in findings}
        pts = synthesis.key_points
        dangling = sorted({s for p in pts for s in p.support if s not in ids})
        unsupported_pts = sum(1 for p in pts if not any(s in ids for s in p.support))
        # A point is "anchored" if at least one referenced finding has a verified quote.
        anchored = sum(
            1 for p in pts
            if any(s in checks and checks[s].status in ("verified", "wrong_page")
                   for s in p.support)
        )
        syn = {"points": len(pts), "points_without_valid_support": unsupported_pts,
               "dangling_ids": dangling, "points_anchored_to_verified_quote": anchored}

    return GroundingReport(
        checks=checks,
        totals=dict(totals),
        rate_verified=round(totals["verified"] / n, 3),
        rate_grounded=round(
            (totals["verified"] + totals["partial"] + totals["wrong_page"]) / n, 3),
        by_kind={k: dict(v) for k, v in by_kind.items()},
        by_doc={k: dict(v) for k, v in by_doc.items()},
        low_quality_share_of_failures=None if lq is None else round(lq, 3),
        support_totals=dict(support) or None,
        synthesis=syn,
    )
