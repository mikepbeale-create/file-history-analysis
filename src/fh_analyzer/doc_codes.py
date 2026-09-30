"""USPTO Image File Wrapper (IFW) document codes relevant to prosecution-history analysis.

A file wrapper for a mature patent can hold 100+ documents (fee sheets, receipts, filing
formalities, IDS forms ...). Only a fraction matter for understanding claim scope, so we
classify each document and only send the substantive ones to the model. That keeps cost
down and, more importantly, keeps noise out of the context window.
"""

from __future__ import annotations

from enum import Enum


class DocCategory(str, Enum):
    CLAIMS = "claims"                  # claim sets as filed / as amended
    REJECTION = "rejection"            # non-final / final office actions, Quayle
    RESTRICTION = "restriction"
    ADVISORY = "advisory"
    APPLICANT_RESPONSE = "response"    # amendments + remarks/arguments
    INTERVIEW = "interview"
    ALLOWANCE = "allowance"            # notice of allowance, reasons for allowance, exr's amendment
    APPEAL = "appeal"
    RCE = "rce"
    OTHER = "other"


# Primary mapping by IFW document code. Codes vary a little over time, so classification
# falls back to keywords in the document description (see `classify`).
CODE_MAP: dict[str, DocCategory] = {
    "CLM": DocCategory.CLAIMS,
    "CTNF": DocCategory.REJECTION,
    "CTFR": DocCategory.REJECTION,
    "CTEQ": DocCategory.REJECTION,       # Ex parte Quayle
    "CTRS": DocCategory.RESTRICTION,
    "CTAV": DocCategory.ADVISORY,
    "REM": DocCategory.APPLICANT_RESPONSE,
    "A...": DocCategory.APPLICANT_RESPONSE,
    "A.NE": DocCategory.APPLICANT_RESPONSE,  # amendment after final
    "AMSB": DocCategory.APPLICANT_RESPONSE,  # supplemental amendment
    "A.PE": DocCategory.APPLICANT_RESPONSE,  # preliminary amendment
    "ELC.": DocCategory.APPLICANT_RESPONSE,  # election in response to restriction
    "EXIN": DocCategory.INTERVIEW,
    "NOA": DocCategory.ALLOWANCE,
    "EXAN": DocCategory.ALLOWANCE,
    "APBR": DocCategory.APPEAL,
    "APEA": DocCategory.APPEAL,
    "APRB": DocCategory.APPEAL,
    "RCEX": DocCategory.RCE,
    "OA.FAI": DocCategory.REJECTION,        # first-action interview office action
    "OA.FAI.PRELM": DocCategory.REJECTION,  # pre-interview first office action
    "A.NA": DocCategory.APPLICANT_RESPONSE,  # amendment after allowance (Rule 312)
    # Codes whose descriptions match a keyword but carry no substantive content:
    "FWCLM": DocCategory.OTHER,             # examiner's index of claims (a status table)
    "N271": DocCategory.OTHER,              # PTO response to Rule 312 amendment (form)
    "M327": DocCategory.OTHER,              # misc. communication, no action count
    "OA.POSTCARD": DocCategory.OTHER,       # e-Office Action courtesy postcard
    "FAI.REQ": DocCategory.OTHER,           # first-action interview enrollment request
}

_KEYWORDS: list[tuple[str, DocCategory]] = [
    ("reasons for allowance", DocCategory.ALLOWANCE),
    ("notice of allowance", DocCategory.ALLOWANCE),
    ("examiner's amendment", DocCategory.ALLOWANCE),
    ("non-final rejection", DocCategory.REJECTION),
    ("final rejection", DocCategory.REJECTION),
    ("restriction", DocCategory.RESTRICTION),
    ("advisory action", DocCategory.ADVISORY),
    ("interview summary", DocCategory.INTERVIEW),
    ("applicant arguments", DocCategory.APPLICANT_RESPONSE),
    ("amendment", DocCategory.APPLICANT_RESPONSE),
    ("office action", DocCategory.REJECTION),
    ("appeal", DocCategory.APPEAL),
    ("request for continued examination", DocCategory.RCE),
    ("claims", DocCategory.CLAIMS),
]

# Categories worth OCR + LLM time. RCE transmittals are mostly forms; the amendment filed
# with an RCE carries its own REM/CLM codes, so RCE is only used for the timeline.
SUBSTANTIVE = {
    DocCategory.CLAIMS,
    DocCategory.REJECTION,
    DocCategory.RESTRICTION,
    DocCategory.ADVISORY,
    DocCategory.APPLICANT_RESPONSE,
    DocCategory.INTERVIEW,
    DocCategory.ALLOWANCE,
    DocCategory.APPEAL,
}


def classify(code: str | None, description: str | None) -> DocCategory:
    if code and code.strip() in CODE_MAP:
        return CODE_MAP[code.strip()]
    desc = (description or "").lower()
    for kw, cat in _KEYWORDS:
        if kw in desc:
            return cat
    return DocCategory.OTHER
