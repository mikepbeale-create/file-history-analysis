import shutil

import pymupdf
import pytest

from fh_analyzer.ocr import clean_text, extract_document, render_for_llm

TEXT = ("Claim 1 is rejected under 35 U.S.C. 103 as being unpatentable over Smith in view of "
        "Jones. Smith discloses a housing and a spring disposed within the housing. ") * 3


def _pdf_with_text(path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), TEXT, fontsize=12)
    doc.save(path)


def _image_only_pdf(path):
    """Render a text page to an image and wrap it in a PDF: no text layer, like IFW scans."""
    src = pymupdf.open()
    p = src.new_page()
    p.insert_textbox(pymupdf.Rect(50, 50, 550, 800), TEXT, fontsize=14)
    pix = p.get_pixmap(dpi=200)
    out = pymupdf.open()
    page = out.new_page(width=p.rect.width, height=p.rect.height)
    page.insert_image(page.rect, stream=pix.tobytes("png"))
    out.save(path)


def test_text_layer_is_used_when_present(tmp_path, settings):
    pdf = tmp_path / "pdf" / "D1.pdf"
    pdf.parent.mkdir()
    _pdf_with_text(pdf)
    dt = extract_document(pdf, "D1", settings)
    assert dt.pages[0].method == "text-layer"
    assert "Smith in view of Jones" in dt.pages[0].text
    assert (tmp_path / "text" / "D1.json").exists()   # cached


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_ocr_fallback_on_image_only_pdf(tmp_path, settings):
    pdf = tmp_path / "pdf" / "D2.pdf"
    pdf.parent.mkdir()
    _image_only_pdf(pdf)
    dt = extract_document(pdf, "D2", settings)
    p = dt.pages[0]
    assert p.method == "ocr"
    assert p.confidence and p.confidence > 70
    assert "Smith" in p.text and "housing" in p.text


def test_clean_text():
    raw = "Application/Control Number: 16/123,456\nThe hous-\ning has a “spring”.\n\n\n\nEnd"
    assert clean_text(raw) == 'The housing has a "spring".\n\nEnd'


def test_render_for_llm_marks_pages(texts):
    out = render_for_llm("OA-1", "label", texts["OA-1"])
    assert '<document id="OA-1"' in out and '<page n="2">' in out


def test_load_local_folder_parses_patent_center_names(tmp_path, settings):
    from fh_analyzer.doc_codes import DocCategory
    from fh_analyzer.pipeline import load_local_folder

    for name in ["99999999-2020-06-01-00005-CTNF.pdf", "99999999-2020-01-10-00002-N417.pdf"]:
        _pdf_with_text(tmp_path / name)
    app, docs, texts = load_local_folder(tmp_path, settings)
    oa = next(d for d in docs if d.code == "CTNF")
    assert oa.category == DocCategory.REJECTION and str(oa.official_date) == "2020-06-01"
    assert set(texts) == {oa.doc_id}          # the filing receipt is not OCR'd
