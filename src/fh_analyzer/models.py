"""Structured output schemas.

Every factual item the model returns carries a `Citation` (document id + page + a
short verbatim quote). The schemas are passed to the model as tool input schemas,
so the model's output is validated JSON rather than free text we have to parse.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: str = Field(description="The id attribute of the <document> the quote comes from.")
    page: int = Field(description="The n attribute of the <page> the quote comes from.")
    quote: str = Field(
        description=(
            "An exact, verbatim excerpt (8-60 words) copied character-for-character from "
            "that page. Do not paraphrase, fix OCR errors, or join text from different pages."
        )
    )


class Risk(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


# ------------------------------------------------------------------ office actions

class Rejection(BaseModel):
    claims: list[int] = Field(description="Claim numbers rejected on this ground.")
    basis: Literal[
        "101", "102", "103", "112(a)", "112(b)", "112(d)", "112(f)",
        "double patenting", "objection", "other",
    ]
    references: list[str] = Field(
        default_factory=list,
        description="Prior art relied on, as named by the examiner (e.g. 'Smith US 9,123,456').",
    )
    summary: str = Field(description="One or two sentences: the examiner's reasoning.")
    citation: Citation


class OfficeActionExtraction(BaseModel):
    action_type: Literal[
        "non-final rejection", "final rejection", "advisory action",
        "restriction requirement", "ex parte quayle", "other",
    ]
    rejections: list[Rejection] = Field(
        default_factory=list,
        description="One entry per ground of rejection or objection stated in the action "
        "(e.g. a 101, a 102 over X, a 103 over X in view of Y are three entries). Must not "
        "be empty if the action rejects or objects to any claim.",
    )
    allowable_claims: list[int] = Field(
        default_factory=list,
        description="Claims indicated allowable or 'objected to but would be allowable'.",
    )
    allowable_citation: Citation | None = None
    summary: str
    summary_citation: Citation | None = None


# ------------------------------------------------------------------ applicant responses

class Amendment(BaseModel):
    claim: int
    change: str = Field(description="What limitation was added, removed, or rewritten.")
    citation: Citation


class EstoppelStatement(BaseModel):
    """An applicant statement or amendment that may narrow claim scope."""

    claims: list[int]
    limitation: str = Field(description="The claim term/limitation the statement concerns.")
    kind: Literal[
        "argument-based disclaimer", "amendment-based estoppel", "definition/lexicography"
    ]
    statement: str = Field(description="Plain-English summary of what applicant said or did.")
    distinguished_art: list[str] = Field(default_factory=list)
    risk: Risk = Field(
        description=(
            "high = clear and unmistakable disavowal or narrowing amendment made to overcome art; "
            "medium = argument that arguably narrows scope; low = routine or ambiguous."
        )
    )
    rationale: str = Field(description="Why this risk level.")
    citation: Citation


class ResponseCoreExtraction(BaseModel):
    """Applicant filing without the estoppel analysis (used when FHA_ESTOPPEL is off)."""
    amendments: list[Amendment] = Field(
        default_factory=list,
        description="One entry per claim whose limitations were changed, as described in "
        "the remarks or shown in a claim listing in this document.",
    )
    summary: str
    summary_citation: Citation | None = None


class ResponseExtraction(ResponseCoreExtraction):
    estoppel: list[EstoppelStatement] = Field(
        default_factory=list,
        description="One entry per argument distinguishing prior art or characterising "
        "claim scope, and per narrowing amendment made to overcome a rejection. Include "
        "low-risk ones; the risk field grades them.",
    )


# ------------------------------------------------------------------ claims

class ClaimVersion(BaseModel):
    number: int
    status: Literal[
        "original", "currently amended", "previously presented", "canceled",
        "withdrawn", "new", "unknown",
    ]
    independent: bool
    text: str = Field(
        description=(
            "Clean claim text AS AMENDED in this document: omit text shown struck-through or "
            "in [[double brackets]]; keep added (underlined) text. Empty for canceled claims."
        )
    )
    citation: Citation = Field(description="Quote the claim's opening words as they appear.")


class ClaimsExtraction(BaseModel):
    claims: list[ClaimVersion] = Field(
        description="Independent claims only, plus any claim whose status is 'canceled'."
    )


# ------------------------------------------------------------------ allowance / interviews

class AllowanceReason(BaseModel):
    point: str = Field(description="A limitation or reason the examiner relied on to allow.")
    claims: list[int] = Field(default_factory=list)
    citation: Citation


class AllowanceExtraction(BaseModel):
    reasons: list[AllowanceReason] = Field(
        default_factory=list,
        description="One entry per reason or distinguishing limitation in the examiner's "
        "statement of reasons for allowance, if the document contains one.",
    )
    examiners_amendment: str | None = Field(
        default=None, description="Summary of any examiner's amendment, if present."
    )
    examiners_amendment_citation: Citation | None = None
    summary: str
    summary_citation: Citation | None = None


class InterviewExtraction(BaseModel):
    agreement_reached: bool | None
    summary: str
    summary_citation: Citation | None = None


# ------------------------------------------------------------------ synthesis

class KeyPoint(BaseModel):
    text: str
    support: list[str] = Field(
        description="Finding ids (e.g. 'F12') from the input that support this point. "
        "Every point must cite at least one."
    )


class Synthesis(BaseModel):
    overview: str = Field(description="3-6 sentence narrative of the prosecution.")
    key_points: list[KeyPoint] = Field(
        description="The most important takeaways for claim construction / validity."
    )
