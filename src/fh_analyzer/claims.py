"""Claim evolution: deterministic word-level diffs between successive claim listings.

The LLM only *transcribes* each claim listing. The comparison between versions is done
in code with difflib, so the "what changed" view is reproducible and cannot be
hallucinated.
"""

from __future__ import annotations

import difflib
import html
import re
from datetime import date

from pydantic import BaseModel


class ClaimSnapshot(BaseModel):
    number: int
    doc_id: str
    date: date | None
    status: str
    text: str


class ClaimChange(BaseModel):
    from_doc: str | None
    to_doc: str
    date: date | None
    status: str
    added: list[str]
    removed: list[str]
    diff_html: str
    similarity: float   # 0..1 ratio vs previous version
    # A diff on a claim whose status says it did NOT change is almost certainly OCR or
    # transcription noise. Flag rather than hide it, so the reviewer can see it.
    noise_suspect: bool = False


class ClaimHistory(BaseModel):
    number: int
    snapshots: list[ClaimSnapshot]
    changes: list[ClaimChange]

    @property
    def final_text(self) -> str:
        live = [s for s in self.snapshots if s.status != "canceled" and s.text]
        return live[-1].text if live else ""

    @property
    def canceled(self) -> bool:
        return bool(self.snapshots) and self.snapshots[-1].status == "canceled"


_UNCHANGED = {"previously presented", "original"}
_TOKEN = re.compile(r"\w+|[^\w\s]")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)


def _join(tokens: list[str]) -> str:
    out = ""
    for t in tokens:
        out += t if (not out or re.fullmatch(r"[^\w\s(\[]", t)) else " " + t
    return out


def diff_claim(old: str, new: str) -> tuple[list[str], list[str], str, float]:
    a, b = _tokens(old), _tokens(new)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    added, removed = [], []
    out = ""

    def emit(tokens: list[str], tag: str | None) -> None:
        nonlocal out
        seg = _join(tokens)
        sep = "" if not out or re.fullmatch(r"[^\w\s(\[]", tokens[0]) else " "
        body = html.escape(seg)
        out += sep + (f'<{tag} class="{"add" if tag == "ins" else "rm"}">{body}</{tag}>'
                      if tag else body)

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            emit(a[i1:i2], None)
            continue
        if op in ("delete", "replace"):
            removed.append(_join(a[i1:i2]))
            emit(a[i1:i2], "del")
        if op in ("insert", "replace"):
            added.append(_join(b[j1:j2]))
            emit(b[j1:j2], "ins")
    return added, removed, out, sm.ratio()


def build_histories(snapshots: list[ClaimSnapshot]) -> list[ClaimHistory]:
    by_num: dict[int, list[ClaimSnapshot]] = {}
    for s in sorted(snapshots, key=lambda s: (s.date or date.min, s.doc_id)):
        by_num.setdefault(s.number, []).append(s)
    out = []
    for num, snaps in sorted(by_num.items()):
        changes: list[ClaimChange] = []
        prev: ClaimSnapshot | None = None
        for s in snaps:
            if prev is None or s.status == "canceled" or prev.text != s.text:
                if prev is not None and s.status != "canceled":
                    add, rm, dh, ratio = diff_claim(prev.text, s.text)
                else:
                    add, rm, dh, ratio = [], [], html.escape(s.text), 1.0
                if prev is None or s.status == "canceled" or add or rm:
                    changes.append(ClaimChange(
                        from_doc=prev.doc_id if prev else None, to_doc=s.doc_id, date=s.date,
                        status=s.status, added=add, removed=rm, diff_html=dh,
                        similarity=round(ratio, 3),
                        noise_suspect=bool(prev) and s.status in _UNCHANGED and bool(add or rm),
                    ))
            if s.status != "canceled":
                prev = s
        out.append(ClaimHistory(number=num, snapshots=snaps, changes=changes))
    return out
