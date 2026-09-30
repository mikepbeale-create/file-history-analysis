import json

import httpx
import pytest

from fh_analyzer.doc_codes import DocCategory
from fh_analyzer.uspto import ODPClient, USPTOError, normalize_number, parse_documents

from .conftest import FIXTURES


@pytest.mark.parametrize("raw,expected", [
    ("US 10,123,456 B2", ("patent", "10123456")),
    ("10123456", ("patent", "10123456")),
    ("7,654,321", ("patent", "7654321")),
    ("RE49,123", ("patent", "RE49123")),
    ("16/123,456", ("application", "16123456")),
])
def test_normalize_number(raw, expected):
    assert normalize_number(raw) == expected


def test_normalize_rejects_garbage():
    with pytest.raises(ValueError):
        normalize_number("hello")


def test_parse_documents_sorts_and_classifies():
    docs = parse_documents(json.loads((FIXTURES / "documents.json").read_text()))
    assert [d.doc_id for d in docs] == ["DOC0001", "DOC0002", "DOC0003", "DOC0004", "DOC0006"]
    cats = {d.doc_id: d.category for d in docs}
    assert cats["DOC0001"] == DocCategory.CLAIMS
    assert cats["DOC0002"] == DocCategory.OTHER          # filing receipt: skipped
    assert cats["DOC0003"] == DocCategory.REJECTION
    assert cats["DOC0006"] == DocCategory.ALLOWANCE      # unknown code -> description fallback
    oa = next(d for d in docs if d.doc_id == "DOC0003")
    assert oa.pdf_url.endswith("DOC0003.pdf") and oa.page_count == 2


def _client(tmp_path, handler):
    return ODPClient("k", tmp_path, transport=httpx.MockTransport(handler), max_retries=3)


def test_resolve_patent_and_list_documents(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    seen = []

    def handler(req: httpx.Request):
        seen.append(req)
        assert req.headers["X-API-KEY"] == "k"
        if req.url.path.endswith("/search"):
            assert req.url.params["q"] == "applicationMetaData.patentNumber:10123456"
            return httpx.Response(200, json={"count": 1, "patentFileWrapperDataBag": [{
                "applicationNumberText": "99999999",
                "applicationMetaData": {"patentNumber": "10123456", "inventionTitle": "Widget",
                                        "filingDate": "2020-01-10", "grantDate": "2021-06-01"}}]})
        if req.url.path.endswith("/documents"):
            return httpx.Response(200, json=json.loads((FIXTURES / "documents.json").read_text()))
        raise AssertionError(req.url)

    with _client(tmp_path, handler) as c:
        app = c.resolve("US 10,123,456 B2")
        assert app.application_number == "99999999" and app.title == "Widget"
        assert len(c.list_documents("99999999")) == 5
        # Second call is served from cache: no new HTTP request.
        n = len(seen)
        c.list_documents("99999999")
        assert len(seen) == n


def test_download_retries_on_429(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    attempts = {"n": 0}

    def handler(req):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, content=b"%PDF-1.7 fake")

    docs = parse_documents(json.loads((FIXTURES / "documents.json").read_text()))
    with _client(tmp_path, handler) as c:
        p = c.download_pdf("99999999", docs[0])
    assert p.read_bytes().startswith(b"%PDF") and attempts["n"] == 2


def test_bad_key_is_a_clear_error(tmp_path):
    with _client(tmp_path, lambda r: httpx.Response(403)) as c, \
            pytest.raises(USPTOError, match="API key"):
        c.resolve("10123456")


def test_patent_missing_from_odp_explains_what_to_do(tmp_path):
    # ODP answers a search with no hits with a 404 (seen live for a 1998 filing).
    with _client(tmp_path, lambda r: httpx.Response(404)) as c, \
            pytest.raises(USPTOError, match="--folder"):
        c.resolve("6,219,730")
    assert not (tmp_path / "_lookup" / "6219730.json").exists()   # nothing bad cached
