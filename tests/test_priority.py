"""Priority-chain support analysis, end to end with fake USPTO + fake LLM.

Fictional chain:   V (provisional) <-PRO- G <-CON- P <-CIP- S (the patent)
  V discloses a housing and a spring          -> claim 1 entitled to V
  G/P add a coil spring                       -> claim 2 (coil) entitled to G
  S (CIP) adds a preloaded spring             -> claim 3 only gets S; flagged new matter
"""

import re
from datetime import date

import pymupdf
import pytest

from fh_analyzer import priority as P
from fh_analyzer.llm import Usage
from fh_analyzer.uspto import ApplicationInfo, FileWrapperDoc

HOUSING = "The widget includes a rigid housing that encloses the moving parts of the device."
SPRING = "A spring is disposed within the housing and biases the lever toward the rest position."
COIL = "In a preferred embodiment the spring is a helical coil spring made of spring steel wire."
PRELOAD = "The coil spring is preloaded during assembly so that it exerts force at rest position."
FILLER = "The lever pivots about a pin that is fixed to the frame of the widget in all versions."

SPECS = {
    "63000001": [HOUSING, SPRING, FILLER],
    "12000002": [HOUSING, SPRING, COIL, FILLER],
    "13000003": [HOUSING, SPRING, COIL, FILLER],
    "14000004": [HOUSING, SPRING, COIL, PRELOAD, FILLER],
}
DATES = {"63000001": date(2010, 1, 5), "12000002": date(2011, 1, 4),
         "13000003": date(2013, 6, 1), "14000004": date(2015, 3, 2)}

CONTINUITY = {"count": 1, "patentFileWrapperDataBag": [{
    "applicationNumberText": "14000004",
    "parentContinuityBag": [
        {"parentApplicationNumberText": "13000003", "childApplicationNumberText": "14000004",
         "claimParentageTypeCode": "CIP",
         "claimParentageTypeCodeDescriptionText": "is a Continuation in part of",
         "parentApplicationFilingDate": "2013-06-01", "parentPatentNumber": "9000003"},
        {"parentApplicationNumberText": "12000002", "childApplicationNumberText": "13000003",
         "claimParentageTypeCode": "CON", "parentApplicationFilingDate": "2011-01-04"},
        {"parentApplicationNumberText": "63000001", "childApplicationNumberText": "12000002",
         "claimParentageTypeCode": "PRO", "parentApplicationFilingDate": "2010-01-05"},
    ],
    "childContinuityBag": [
        {"parentApplicationNumberText": "14000004", "childApplicationNumberText": "15000005",
         "claimParentageTypeCode": "CON"}],
}]}

CLAIMS = """1. A widget comprising: a housing; and a spring disposed within the housing.
2. The widget of claim 1, wherein the spring is a coil spring.
3. The widget of claim 1, wherein the spring is preloaded."""


def _pdf(path, paragraphs):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 560, 800), "\n\n".join(paragraphs), fontsize=10)
    doc.save(path)


class FakeODP:
    def __init__(self, tmp):
        self.tmp = tmp

    def continuity(self, app_no):
        return CONTINUITY

    def list_documents(self, app_no):
        return [
            FileWrapperDoc(doc_id=f"{app_no}-SPEC", code="SPEC", description="Specification",
                           official_date=DATES[app_no], pdf_url="x"),
            # A later SPEC (amendment) must be ignored: only the as-filed disclosure counts.
            FileWrapperDoc(doc_id=f"{app_no}-SPEC2", code="SPEC", description="Specification",
                           official_date=date(2020, 1, 1), pdf_url="x"),
            FileWrapperDoc(doc_id=f"{app_no}-N417", code="N417", description="Receipt",
                           official_date=DATES[app_no], pdf_url="x"),
        ]

    def download_pdf(self, app_no, doc):
        pdf_dir = self.tmp / app_no / "pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        path = pdf_dir / f"{doc.doc_id}.pdf"
        _pdf(path, SPECS[app_no] if doc.doc_id.endswith("SPEC") else ["Amended text only."])
        return path


KEYWORDS = {"1[pre]": "widget includes", "1[a]": "rigid housing", "1[b]": "spring is disposed",
            "2[a]": "helical coil", "3[a]": "preloaded"}


class FakeLLM:
    model = "fake"

    def __init__(self):
        self.usage = Usage()
        self.support_calls = 0

    def extract(self, system, user, schema):
        self.usage.calls += 1
        if schema is P.ClaimSet:
            return P.ClaimSet(claims=[
                P.ParsedClaim(number=1, text="A widget comprising: a housing; and a spring "
                              "disposed within the housing."),
                P.ParsedClaim(number=2, depends_on=1, text="The widget of claim 1, wherein "
                              "the spring is a coil spring."),
                P.ParsedClaim(number=3, depends_on=1, text="The widget of claim 1, wherein "
                              "the spring is preloaded."),
            ])
        if schema is P.ClaimElements:
            return P.ClaimElements(elements=[
                P.Element(id="1[pre]", claim=1, text="A widget comprising"),
                P.Element(id="1[a]", claim=1, text="a housing"),
                P.Element(id="1[b]", claim=1, text="a spring disposed within the housing"),
                P.Element(id="2[a]", claim=2, text="wherein the spring is a coil spring"),
                P.Element(id="3[a]", claim=3, text="wherein the spring is preloaded"),
            ])
        if schema is P.SupportExtraction:
            self.support_calls += 1
            doc_id = re.search(r'<document id="([^"]+)"', system).group(1)
            items = []
            for eid in re.findall(r"^(\S+):", user, re.M):
                kw = KEYWORDS[eid]
                m = re.search(r"[^.<>\n]*" + re.escape(kw) + r"[^.<>\n]*\.", system)
                if m:
                    items.append(P.ElementSupport(
                        element_id=eid, level="explicit", explanation="stated",
                        citations=[{"doc_id": doc_id, "page": 1, "quote": m.group(0).strip()}]))
                else:
                    items.append(P.ElementSupport(element_id=eid, level="none",
                                                  explanation="absent"))
            return P.SupportExtraction(items=items)
        raise AssertionError(schema)


class FakeAnalysis:
    app = ApplicationInfo(application_number="14000004", patent_number="9999999",
                          filing_date=DATES["14000004"])
    documents = []
    texts = {}


def test_parse_chain_builds_ancestors_only():
    ch = P.parse_chain(CONTINUITY, "14000004", DATES["14000004"])
    assert set(ch.apps) == {"14000004", "13000003", "12000002", "63000001"}  # child dropped
    assert ch.apps["63000001"].provisional
    assert ch.path_to("63000001") == ["14000004", "13000003", "12000002", "63000001"]
    assert [a.app_no for a in ch.ordered()][0] == "63000001"
    assert ch.edge("14000004", "13000003").code == "CIP"


def test_claim_element_inheritance():
    claims = {c.number: c for c in FakeLLM().extract("", "", P.ClaimSet).claims}
    els = FakeLLM().extract("", "", P.ClaimElements).elements
    assert P.claim_element_ids(3, claims, els) == ["1[pre]", "1[a]", "1[b]", "3[a]"]


@pytest.fixture
def report(tmp_path, settings):
    llm = FakeLLM()
    client = FakeODP(tmp_path)
    rep = P.prepare(client, FakeAnalysis(), llm, claims_text=CLAIMS)
    assert rep.selected == [1]                              # independents preselected
    rep = P.run(client, rep, llm, settings, [1, 2, 3])
    return rep, llm


def test_earliest_supported_dates(report):
    rep, llm = report
    assert not rep.errors and not rep.element_warnings
    res = {r.claim: r for r in rep.results}
    assert res[1].strict_date == DATES["63000001"]           # provisional supports claim 1
    assert res[2].strict_date == DATES["12000002"]           # coil first appears in G
    assert res[3].strict_date == DATES["14000004"]           # preload only in the CIP
    assert any("63/000,001" in b and "2[a]" in b for b in res[2].breaks)
    assert llm.support_calls == 4                            # one batch per application


def test_as_filed_only_and_quotes_verified(report):
    rep, _ = report
    assert rep.disclosures["12000002"].doc_ids == ["12000002-SPEC"]   # later SPEC ignored
    green = [c for c in rep.cells if c.level == "explicit"]
    assert green and all(all(c.verified) and c.color == "green" for c in green)


def test_cip_new_matter_flagged(report):
    rep, _ = report
    (nm,) = rep.new_matter
    assert (nm.child, nm.parent) == ("14000004", "13000003")
    assert [t for _, t in nm.new_passages] == [PRELOAD]
    assert nm.new_sentences == 1 and 0 < nm.new_fraction < 0.5
    cell = rep.grid()[("14000004", "3[a]")]
    assert cell.new_matter and cell.color == "green"
    assert not rep.grid()[("14000004", "1[a]")].new_matter


def test_unverified_support_is_downgraded():
    c = P.SupportCell(element_id="1[a]", app_no="x", level="explicit",
                      citations=[{"doc_id": "d", "page": 1, "quote": "made up quote here ok"}],
                      verified=[False])
    assert c.color == "yellow"


def test_priority_tab_renders(report, tmp_path, monkeypatch, app_info, docs, texts,
                              settings):
    """The Priority tab renders a finished report (smoke test via Streamlit AppTest)."""
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from fh_analyzer.pipeline import analyze, save
    from tests.conftest import FakeLLM as PipelineLLM

    rep, _ = report
    a = analyze(app_info, docs, texts, PipelineLLM(), settings)
    a.priority = rep
    save(a, tmp_path / "data" / "cache" / app_info.application_number / "analysis.json")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FHA_CACHE_DIR", "data/cache")
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Load saved analysis").run()
    next(b for b in at.sidebar.button if b.label == "Load").click().run()
    assert not at.exception
    labels = [s.value for s in at.subheader]
    assert "Earliest supported date per claim" in labels and "Continuity chain" in labels
    grid = next(d.value for d in at.dataframe if "element" in d.value.columns)
    assert grid.shape[0] == 5
