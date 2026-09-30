"""Completeness checks: did the model miss something the document plainly contains?

Grounding (grounding.py) catches *precision* problems: a finding whose evidence is not
where it says. It cannot catch *omissions*. These checks are cheap, deterministic
tripwires on the OCR text: if an office action says "rejected under 35 U.S.C." but the
model returned no rejections, something was missed. A tripped check triggers one retry
with specific feedback; anything still unresolved is reported in the Evaluation tab.

They are deliberately simple signals, not a recall metric - a proper recall number
needs a hand-labeled gold set.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from .models import AllowanceExtraction, OfficeActionExtraction, ResponseCoreExtraction
from .ocr import DocText

# "rejected under" appears in the body of real rejections. (The PTOL-326 cover form's
# "Claim(s) __ is/are rejected" label appears even when nothing is rejected, so skip it.)
_REJECTED = re.compile(r"\brejected\s+under\b", re.I)
_OBJECTED = re.compile(r"\bobjected\s+to\b", re.I)
# A real statement ("The following is an examiner's statement of reasons for allowance:"
# or a "REASONS FOR ALLOWANCE" heading) - not the PTOL-37 checkbox label of the same name.
_REASONS = re.compile(r"statement\s+of\s+reasons\s+for\s+allowance\s*:|"
                      r"^\s*reasons?\s+for\s+allowance\s*$", re.I | re.M)
# Language typical of substantive remarks (as opposed to a transmittal cover sheet).
_ARGUMENT = re.compile(
    r"respectfully\s+(submit|traverse|disagree|request)|does\s+not\s+(teach|disclose|"
    r"suggest)|fails?\s+to\s+(teach|disclose)|as\s+amended|has\s+been\s+amended|"
    r"amended\s+(independent\s+)?claims?", re.I)
# Language that says claims were changed (for filings checked without the estoppel list,
# where pure argument without amendment is a legitimate, empty result).
_AMENDED = re.compile(r"as\s+amended|has\s+been\s+amended|amended\s+(independent\s+)?claims?|"
                      r"currently\s+amended", re.I)


def check(ex: BaseModel, text: DocText) -> list[str]:
    """Return human-readable issues; empty list means nothing obviously missed."""
    body = text.full_text
    issues: list[str] = []
    if isinstance(ex, OfficeActionExtraction):
        n = len(_REJECTED.findall(body))
        if n and not ex.rejections:
            issues.append(
                f"The action text says 'rejected under' {n} time(s) but no rejections were "
                "recorded. Record each ground of rejection as its own item in `rejections`."
            )
        elif not ex.rejections and _OBJECTED.search(body) and not ex.allowable_claims:
            issues.append("The action objects to claims but nothing was recorded.")
    elif isinstance(ex, ResponseCoreExtraction):
        has_estoppel = hasattr(ex, "estoppel")
        hits = len((_ARGUMENT if has_estoppel else _AMENDED).findall(body))
        if hits >= 2 and not ex.amendments and not getattr(ex, "estoppel", None):
            issues.append(
                f"The filing contains {hits} passages of "
                + ("argument/amendment" if has_estoppel else "amendment") + " language but no "
                + ("amendments or estoppel statements were recorded. Record each claim change "
                   "in `amendments` and each argument distinguishing art or characterising "
                   "claim scope in `estoppel` (low risk is fine)." if has_estoppel else
                   "amendments were recorded. Record each claim change in `amendments`.")
            )
    elif isinstance(ex, AllowanceExtraction):
        if _REASONS.search(body) and not ex.reasons:
            issues.append(
                "The document contains a statement of reasons for allowance but no reasons "
                "were recorded. Record each reason in `reasons`."
            )
    return issues
