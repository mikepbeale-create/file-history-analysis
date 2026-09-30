from fh_analyzer.grounding import Finding, build_report, check_citation, normalize
from fh_analyzer.models import Citation


def _f(doc, page, quote, fid="F1"):
    return Finding(id=fid, kind="test", doc_id=doc, doc_label=doc, date=None, statement="s",
                   citation=Citation(doc_id=doc, page=page, quote=quote))


def test_exact_quote_verified(texts):
    c = check_citation(_f("OA-1", 2, "Smith discloses a widget comprising a housing"), texts)
    assert c.status == "verified" and c.score == 100


def test_ocr_noise_tolerated(texts):
    # Model "fixed" OCR / typography slightly: still verified, not exact.
    c = check_citation(_f("OA-1", 2, "Smith discloses a widget comprising a hous1ng (Fig 2, "
                                     "element 10) and a spring"), texts)
    assert c.status == "verified" and 90 <= c.score < 100


def test_wrong_page_detected(texts):
    c = check_citation(_f("OA-1", 1, "Smith discloses a widget comprising a housing (Fig. 2, "
                                     "element 10)"), texts)
    assert c.status == "wrong_page" and c.matched_page == 2


def test_fabricated_quote_unverified_and_low_quality_flagged(texts):
    c = check_citation(_f("REM-1", 1, "Applicant adds the limitation of a helical torsion "
                                      "spring made of steel"), texts)
    assert c.status == "unverified"
    assert c.low_quality_page  # REM-1 fixture pages have 55% OCR confidence


def test_bad_reference(texts):
    assert check_citation(_f("NOPE", 1, "x y z a b c"), texts).status == "bad_reference"
    assert check_citation(_f("OA-1", 9, "x y z a b c"), texts).status == "bad_reference"


def test_short_quote_flagged(texts):
    assert check_citation(_f("OA-1", 1, "Claim 1"), texts).short_quote


def test_normalize():
    assert normalize("A “B” — c-\n d") == 'a "b" - cd'


def test_report_rates(texts):
    fs = [_f("OA-1", 2, "Smith discloses a widget comprising a housing", "F1"),
          _f("OA-1", 1, "completely invented sentence that is nowhere in the record", "F2")]
    checks = {f.id: check_citation(f, texts) for f in fs}
    r = build_report(fs, checks)
    assert r.totals == {"verified": 1, "unverified": 1}
    assert r.rate_verified == 0.5


def test_uncited(texts):
    assert check_citation(_f("OA-1", 0, ""), texts).status == "uncited"
