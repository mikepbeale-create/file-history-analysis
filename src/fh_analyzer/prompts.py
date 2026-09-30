"""Prompts. Kept in one place so they can be reviewed and versioned like code."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

PROMPT_VERSION = "2026-09-25.2"

BASE = """\
You are a senior US patent litigation analyst reviewing a patent's prosecution history
(file wrapper). You extract facts from ONE document at a time into a structured record.

Rules — these are checked automatically after you answer:
1. Every item needs a citation: the document id, the page number from the <page n="..">
   tag, and a VERBATIM quote of 8-60 words copied exactly from that page. The text is
   OCR output and may contain errors: copy the errors as they appear; do not correct them.
2. Never quote across a page boundary. Never cite a page you did not quote from.
3. Only report what the document says - but report ALL of it. The structured lists are
   the product; the summary is only a one-paragraph orientation and is NOT a substitute
   for them. If your summary mentions a rejection, amendment, argument or reason, it must
   also appear as its own list item with its own citation. Use an empty list only when
   the document genuinely contains none (e.g. a cover sheet or transmittal form).
4. Claim numbers are integers. Expand ranges ("claims 1-3" -> [1, 2, 3]).
"""

OFFICE_ACTION = BASE + """
This document is an examiner communication (office action, advisory action, restriction,
or Quayle action). Record each distinct ground of rejection or objection separately,
with the claims, statutory basis, the references relied on, and the examiner's reasoning.
Record claims indicated allowable (or objected to but allowable if rewritten).
"""

RESPONSE = """
This document is an applicant filing (amendment and/or remarks). Record:
- amendments: each substantive change to a claim (added/removed limitations).
- estoppel: each statement or amendment that could limit claim scope in later litigation:
  * argument-based disclaimer — applicant characterises the invention or a claim term
    narrowly, or distinguishes prior art by saying the claims do NOT cover something;
  * amendment-based estoppel — a narrowing amendment made to overcome a rejection
    (Festo presumption);
  * definition/lexicography — applicant defines a term.
  Rate risk conservatively and explain why. Routine statements ("the combination does not
  teach every element") with no characterisation of scope are low risk.
"""

RESPONSE = BASE + RESPONSE

# Applicant filing without the estoppel analysis (FHA_ESTOPPEL off).
RESPONSE_CORE = BASE + """
This document is an applicant filing (amendment and/or remarks). Record:
- amendments: each substantive change to a claim (added/removed limitations).
"""

CLAIMS = BASE + """
This document is a claim listing. Extract every INDEPENDENT claim (a claim that does not
refer to another claim) with its status identifier and its clean text as amended in this
document, plus any claim marked canceled. In amended listings, added text is underlined and
deleted text is struck through or in [[double brackets]]; OCR loses underlining, so use the
brackets and context to remove deleted text as best you can. Cite the claim's opening words.
"""

ALLOWANCE = BASE + """
This document is a notice of allowance (possibly with an examiner's amendment and a
statement of reasons for allowance). Record each distinct reason or limitation the
examiner relied on, and summarise any examiner's amendment.
"""

INTERVIEW = BASE + """
This document is an examiner interview summary. Summarise what was discussed and whether
agreement was reached.
"""

SYNTHESIS = """\
You are a senior US patent litigation analyst. Below are findings already extracted from a
patent's file history, each with an id (F1, F2, ...). Write a short overview of the
prosecution and the key takeaways for claim construction, validity and estoppel.
Use ONLY the findings provided. Every key point must list the finding ids that support it.
Do not introduce facts that are not in the findings.
"""


# ------------------------------------------------------------------ prompt versions
#
# A PromptSet is one complete, named version of every prompt the pipeline uses. The
# experiment runner (fh-experiment) compares versions side by side, so older or
# alternative versions stay registered here instead of being overwritten. A version can
# also be loaded from a folder of .txt files (see PromptSet.from_dir), so a new prompt
# can be tried without editing code.

@dataclass(frozen=True)
class PromptSet:
    version: str
    office_action: str
    response: str
    claims: str
    allowance: str
    interview: str
    synthesis: str
    response_core: str = RESPONSE_CORE      # applicant filings when estoppel is off

    @classmethod
    def from_dir(cls, folder: str | Path, base: PromptSet | None = None) -> PromptSet:
        """Load <folder>/office_action.txt, response.txt, ... ; any file that is missing
        falls back to `base` (default: the current version). The version name is the
        folder name, or the first line of <folder>/VERSION if present."""
        folder = Path(folder)
        base = base or CURRENT
        if not folder.is_dir():
            raise FileNotFoundError(f"Prompt folder not found: {folder}")
        vfile = folder / "VERSION"
        version = (vfile.read_text(encoding="utf-8").strip().splitlines() or [""])[0] \
            if vfile.exists() else folder.name
        over = {}
        for f in fields(cls):
            p = folder / f"{f.name}.txt"
            if f.name != "version" and p.exists():
                over[f.name] = p.read_text(encoding="utf-8")
        if not over:
            raise ValueError(f"{folder} has no prompt files (expected e.g. office_action.txt)")
        return replace(base, version=version or folder.name, **over)


CURRENT = PromptSet(PROMPT_VERSION, OFFICE_ACTION, RESPONSE, CLAIMS, ALLOWANCE, INTERVIEW,
                    SYNTHESIS)

# Ablation: the same prompts without the "report ALL of it" rule that was added after
# live runs came back with empty rejection lists. Running both shows what that rule buys.
_RULE3 = BASE[BASE.index("3. Only report"):BASE.index("4. Claim numbers")]
_BASE_NO_RULE3 = BASE.replace(_RULE3, "3. Only report what the document says.\n")
ABLATE_RULE3 = PromptSet(
    PROMPT_VERSION + "-no-rule3",
    *(p.replace(BASE, _BASE_NO_RULE3, 1)
      for p in (OFFICE_ACTION, RESPONSE, CLAIMS, ALLOWANCE, INTERVIEW)),
    SYNTHESIS,
    RESPONSE_CORE.replace(BASE, _BASE_NO_RULE3, 1),
)

REGISTRY: dict[str, PromptSet] = {p.version: p for p in (CURRENT, ABLATE_RULE3)}


def get_prompt_set(spec: str | None) -> PromptSet:
    """A registered version name, or a path to a prompt folder. None = current."""
    if not spec:
        return CURRENT
    if spec in REGISTRY:
        return REGISTRY[spec]
    if Path(spec).is_dir():
        return PromptSet.from_dir(spec)
    raise KeyError(f"Unknown prompt version {spec!r}. Registered: {', '.join(REGISTRY)}; "
                   "or give a folder of .txt prompt files.")
