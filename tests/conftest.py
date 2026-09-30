"""Shared fixtures: a small, entirely fictional file history (app 99/999,999) and a
scripted fake LLM, so the whole pipeline runs offline and deterministically."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from fh_analyzer.config import Settings
from fh_analyzer.doc_codes import DocCategory
from fh_analyzer.llm import Usage
from fh_analyzer.ocr import DocText, PageText
from fh_analyzer.uspto import ApplicationInfo, FileWrapperDoc

FIXTURES = Path(__file__).parent / "fixtures"

CLM1 = "1. A widget comprising: a housing; and a spring disposed within the housing."
CLM2 = ("1. (Currently Amended) A widget comprising: a housing; and a coil spring disposed "
        "within the housing, wherein the coil spring is preloaded.")
OA_P1 = ("DETAILED ACTION. Claim Rejections - 35 USC 102. Claim 1 is rejected under 35 U.S.C. "
         "102(a)(1) as being anticipated by Smith (US 1,234,567).")
OA_P2 = ("Smith discloses a widget comprising a housing (Fig. 2, element 10) and a spring "
         "disposed within the housing (element 12). Therefore claim 1 is anticipated.")
REM_P1 = "REMARKS. Claim 1 has been amended. Support is found at paragraph [0021]."
REM_P2 = ("Smith's element 12 is a leaf spring. The claimed invention is limited to a coil "
          "spring and does not encompass leaf springs, which cannot be preloaded as claimed.")
NOA_P1 = ("REASONS FOR ALLOWANCE. The prior art of record does not teach or suggest a widget "
          "having a preloaded coil spring disposed within the housing as recited in claim 1.")


def _docs():
    return [
        FileWrapperDoc(doc_id="CLM-A", code="CLM", description="Claims",
                       official_date=date(2020, 1, 10), category=DocCategory.CLAIMS),
        FileWrapperDoc(doc_id="OA-1", code="CTNF", description="Non-Final Rejection",
                       official_date=date(2020, 6, 1), category=DocCategory.REJECTION),
        FileWrapperDoc(doc_id="REM-1", code="REM", description="Remarks",
                       official_date=date(2020, 9, 1), category=DocCategory.APPLICANT_RESPONSE),
        FileWrapperDoc(doc_id="CLM-B", code="CLM", description="Claims",
                       official_date=date(2020, 9, 1), category=DocCategory.CLAIMS),
        FileWrapperDoc(doc_id="NOA-1", code="NOA", description="Notice of Allowance",
                       official_date=date(2021, 1, 5), category=DocCategory.ALLOWANCE),
    ]


def _texts():
    def dt(doc_id, *pages, conf=92.0):
        return DocText(doc_id=doc_id, pages=[
            PageText(page=i + 1, text=t, method="ocr", confidence=conf)
            for i, t in enumerate(pages)])
    return {
        "CLM-A": dt("CLM-A", CLM1),
        "OA-1": dt("OA-1", OA_P1, OA_P2),
        "REM-1": dt("REM-1", REM_P1, REM_P2, conf=55.0),
        "CLM-B": dt("CLM-B", CLM2),
        "NOA-1": dt("NOA-1", NOA_P1),
    }


@pytest.fixture
def docs():
    return _docs()


@pytest.fixture
def texts():
    return _texts()


@pytest.fixture
def app_info():
    return ApplicationInfo(application_number="99999999", patent_number="99999999",
                           title="Widget (fictional test data)")


@pytest.fixture
def settings(tmp_path):
    # estoppel=True: the fixture file history exercises the (currently optional) estoppel
    # analysis; tests of the switched-off path pass estoppel=False explicitly.
    return Settings(uspto_api_key="test", anthropic_api_key="test", model="fake-model",
                    cache_dir=tmp_path / "cache", tesseract_cmd=None, estoppel=True)


class FakeLLM:
    """Returns scripted outputs keyed on the <document id=...> in the prompt.

    The script deliberately includes one wrong-page cite and one fabricated quote so
    the grounding tests have something to catch.
    """

    model = "fake-model"

    def __init__(self):
        self.usage = Usage()
        self.calls: list[str] = []

    def extract(self, system, user, schema):
        self.usage.calls += 1
        name = schema.__name__
        self.calls.append(name)
        c = lambda d, p, q: {"doc_id": d, "page": p, "quote": q}  # noqa: E731
        if name == "ClaimsExtraction" and 'id="CLM-A"' in user:
            data = {"claims": [{"number": 1, "status": "original", "independent": True,
                                "text": CLM1[3:], "citation": c("CLM-A", 1, CLM1)}]}
        elif name == "ClaimsExtraction":
            data = {"claims": [{"number": 1, "status": "currently amended", "independent": True,
                                "text": CLM2[23:], "citation": c("CLM-B", 1, CLM2[:80])}]}
        elif name == "OfficeActionExtraction":
            data = {"action_type": "non-final rejection",
                    "summary": "Claim 1 rejected as anticipated by Smith.",
                    "summary_citation": c("OA-1", 1, "Claim 1 is rejected under 35 U.S.C. "
                                          "102(a)(1) as being anticipated by Smith"),
                    "rejections": [{
                        "claims": [1], "basis": "102", "references": ["Smith US 1,234,567"],
                        "summary": "Smith shows a housing and spring.",
                        # WRONG PAGE on purpose: this text is on page 2.
                        "citation": c("OA-1", 1, "Smith discloses a widget comprising a "
                                      "housing (Fig. 2, element 10) and a spring"),
                    }]}
        elif name in ("ResponseExtraction", "ResponseCoreExtraction"):
            data = {"summary": "Applicant amended claim 1 to recite a preloaded coil spring.",
                    "summary_citation": c("REM-1", 1, "Claim 1 has been amended. Support is "
                                          "found at paragraph [0021]."),
                    "amendments": [{"claim": 1, "change": "Added coil spring, preloaded.",
                                    # FABRICATED on purpose: not in the record.
                                    "citation": c("REM-1", 1, "Applicant adds the limitation "
                                                  "of a helical torsion spring made of steel")}],
                    "estoppel": [{"claims": [1], "limitation": "spring",
                                  "kind": "argument-based disclaimer",
                                  "statement": "Disclaimed leaf springs.",
                                  "distinguished_art": ["Smith"], "risk": "high",
                                  "rationale": "Express statement excluding leaf springs.",
                                  "citation": c("REM-1", 2, "The claimed invention is limited "
                                                "to a coil spring and does not encompass "
                                                "leaf springs")}]}
        elif name == "AllowanceExtraction":
            data = {"summary": "Allowed for preloaded coil spring.",
                    "summary_citation": c("NOA-1", 1, "The prior art of record does not teach "
                                          "or suggest a widget having a preloaded coil spring"),
                    "reasons": [{"point": "Preloaded coil spring in housing", "claims": [1],
                                 "citation": c("NOA-1", 1, "a preloaded coil spring disposed "
                                               "within the housing as recited in claim 1")}]}
        elif name == "Synthesis":
            data = {"overview": "Claim 1 was narrowed to a preloaded coil spring to overcome "
                                "Smith.",
                    "key_points": [{"text": "Leaf springs disclaimed.", "support": ["F6"]},
                                   {"text": "Made-up point.", "support": ["F999"]}]}
        elif name == "_Verdicts":
            import re
            ids = re.findall(r'<item id="(F\d+)"', user)
            data = {"verdicts": [{"id": i, "support": "supported", "reason": "ok"} for i in ids]}
        else:
            raise AssertionError(f"unexpected schema {name}")
        return schema.model_validate(data)


@pytest.fixture
def fake_llm():
    return FakeLLM()
