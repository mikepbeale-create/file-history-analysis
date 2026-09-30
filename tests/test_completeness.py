from fh_analyzer import completeness
from fh_analyzer.models import AllowanceExtraction, OfficeActionExtraction, ResponseExtraction
from fh_analyzer.ocr import DocText, PageText
from fh_analyzer.pipeline import analyze


def _dt(*pages):
    return DocText(doc_id="D", pages=[PageText(page=i + 1, text=t, method="ocr",
                                               confidence=90) for i, t in enumerate(pages)])


def test_office_action_with_rejection_text_but_no_items_trips():
    ex = OfficeActionExtraction(action_type="non-final rejection", summary="Claims rejected.")
    assert completeness.check(ex, _dt("Claim 1 is rejected under 35 U.S.C. 102(a)(1)."))


def test_cover_form_label_alone_does_not_trip():
    ex = OfficeActionExtraction(action_type="ex parte quayle", summary="Quayle.")
    assert not completeness.check(ex, _dt("Claim(s) ____ is/are rejected."))


def test_response_with_arguments_but_no_items_trips():
    ex = ResponseExtraction(summary="Applicant argued.")
    text = _dt("Claim 1 has been amended. Applicant respectfully submits that Smith does "
               "not teach a coil spring.")
    assert completeness.check(ex, text)
    assert not completeness.check(ex, _dt("Transmittal form. Amendment enclosed."))


def test_allowance_checkbox_label_is_not_a_statement():
    ex = AllowanceExtraction(summary="Allowed.")
    assert not completeness.check(ex, _dt("6. [ ] Examiner's Statement of Reasons for Allowance"))
    assert completeness.check(ex, _dt("The following is an examiner's statement of reasons "
                                      "for allowance: the prior art does not teach X."))


def test_pipeline_retries_once_with_feedback(app_info, docs, texts, fake_llm, settings):
    """First pass returns a summary but no rejections; the tripwire forces a retry."""
    real = fake_llm.extract
    seen = []

    def lazy(system, user, schema):
        out = real(system, user, schema)
        if schema is OfficeActionExtraction:
            seen.append(user)
            if len(seen) == 1:
                out.rejections = []
        return out

    fake_llm.extract = lazy
    a = analyze(app_info, docs, texts, fake_llm, settings)
    assert len(seen) == 2 and "missed content" in seen[1]
    assert any(f.kind == "rejection" for f in a.findings)
    assert not a.completeness
