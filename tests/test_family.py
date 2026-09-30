import json

import httpx
import pytest

from fh_analyzer.family import (
    OPSClient,
    OPSError,
    display_number,
    keys_from_text,
    match_key,
    parse_family,
)

from .conftest import FIXTURES

FAMILY = json.loads((FIXTURES / "ops_family.json").read_text())


@pytest.mark.parametrize("country,number,key", [
    ("US", "2005123456", "US2005123456"),       # DOCDB form
    ("US", "20050123456", "US2005123456"),      # as printed: 2005/0123456
    ("US", "09123456", "US9123456"),
    ("US", "RE049123", "USRE49123"),
    ("EP", "01111111", "EP1111111"),
])
def test_match_key(country, number, key):
    assert match_key(country, number) == key


def test_display_number():
    assert display_number("US", "2005123456", "A1") == "US 2005/0123456 A1"
    assert display_number("US", "9123456", "B2") == "US 9,123,456 B2"
    assert display_number("WO", "2014099999", "A1") == "WO 2014/099999 A1"


def test_keys_from_text():
    assert keys_from_text("Smith (US 1,234,567)") == {"US1234567"}
    assert keys_from_text("Jones US 2005/0123456 A1 in view of EP 1 111 111") == {
        "US2005123456", "EP1111111"}
    assert keys_from_text("Lee 9,876,543") == {"US9876543"}
    assert keys_from_text("Smith") == set()


def test_parse_family_members_and_citations():
    r = parse_family(FAMILY, "9999999")
    assert r.family_id == "111"
    assert r.offices == ["CN", "EP", "JP", "US", "WO"]
    assert r.members[0].country == "US"                       # US members first
    cn = next(m for m in r.members if m.country == "CN")
    assert cn.simple_family is False                          # extended INPADOC only
    ep = next(m for m in r.members if m.country == "EP")
    assert ep.register_url.startswith("https://register.epo.org/application?number=EP14999999")
    assert ep.citation_count == 3

    by_key = {c.key: c for c in r.citations}
    # Cited on the face of the US patent -> of record, whoever cited it.
    assert by_key["US1234567"].of_record_us is True
    assert by_key["US2005123456"].of_record_us is True        # IDS art, also Y in EP
    assert by_key["US2005123456"].categories == ["Y"]
    # EP X reference, cited in EP search and the ISR, never before the USPTO.
    x = by_key["EP1111111"]
    assert x.of_record_us is False and x.best_category == "X"
    assert x.categories == ["X", "Y"]
    assert {ci.member for ci in x.cited_in} == {"EP2999999A1", "WO2014099999A1"}
    assert x.cited_in[0].passages == ["paragraph [0021] - paragraph [0030]; figure 2"]
    assert x.cited_in[0].rel_claims == "1-15"
    # Single citation serialized as a dict (not a list) still parses.
    assert by_key["KR20100012345"].of_record_us is False
    # NPL can't be matched automatically.
    npl = next(c for c in r.citations if c.npl)
    assert npl.of_record_us is None and npl.best_category == "A"

    # Not-of-record art first, strongest category first.
    assert [c.display for c in r.new_art] == ["EP 1111111 A1", "JP 2010123456 A",
                                              "KR 20100012345 A"]


def test_rejection_refs_count_as_of_record():
    from fh_analyzer.family import mark_us_record
    r = mark_us_record(parse_family(FAMILY, "9999999"), ["Tanaka JP 2010-123456"])
    assert {c.key for c in r.new_art} == {"EP1111111", "KR20100012345"}


def _client(tmp_path, handler):
    return OPSClient("key", "secret", tmp_path, transport=httpx.MockTransport(handler),
                     max_retries=3)


def test_client_auth_fallback_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    seen = []

    def handler(req: httpx.Request):
        seen.append(req.url.path)
        if req.url.path.endswith("/auth/accesstoken"):
            assert req.headers["Authorization"].startswith("Basic ")
            return httpx.Response(200, json={"access_token": "tok", "expires_in": "1199"})
        assert req.headers["Authorization"] == "Bearer tok"
        if "US.9999999.B2" in req.url.path:
            return httpx.Response(404, text="SERVER.EntityNotFound")
        if "US.9999999.B1" in req.url.path:
            return httpx.Response(200, json=FAMILY)
        raise AssertionError(req.url)

    with _client(tmp_path, handler) as c:
        r = c.family("US 9,999,999 B1", ["Smith (US 1,234,567)"])
        assert len(r.members) == 6 and len(r.new_art) == 3
        n = len(seen)
        c.family("9999999")                    # served from disk cache
        assert len(seen) == n
    assert (tmp_path / "_family" / "US9999999.json").exists()


def test_expired_token_is_refreshed_once(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    tokens = iter(["old", "new"])

    def handler(req):
        if req.url.path.endswith("/auth/accesstoken"):
            return httpx.Response(200, json={"access_token": next(tokens), "expires_in": 1199})
        if req.headers["Authorization"] == "Bearer old":
            return httpx.Response(400, text="<code>invalid_access_token</code>")
        return httpx.Response(200, json=FAMILY)

    with _client(tmp_path, handler) as c:
        assert c.family("9999999").family_id == "111"


def test_missing_keys_and_quota_errors(tmp_path):
    with pytest.raises(OPSError, match="EPO_OPS_KEY"):
        OPSClient(None, None, tmp_path)

    def handler(req):
        if req.url.path.endswith("/auth/accesstoken"):
            return httpx.Response(200, json={"access_token": "t", "expires_in": 1199})
        return httpx.Response(403, headers={"X-Rejection-Reason": "RegisteredQuotaPerWeek"})

    with _client(tmp_path, handler) as c, pytest.raises(OPSError, match="QuotaPerWeek"):
        c.family("9999999")


def test_attach_family_uses_rejection_refs(app_info, docs, texts, fake_llm, settings,
                                           tmp_path, monkeypatch):
    from fh_analyzer.pipeline import analyze, attach_family, load, save

    def handler(req):
        if req.url.path.endswith("/auth/accesstoken"):
            return httpx.Response(200, json={"access_token": "t", "expires_in": 1199})
        return httpx.Response(200, json=FAMILY)

    a = analyze(app_info, docs, texts, fake_llm, settings)
    a.app.patent_number = "9999999"
    with _client(tmp_path, handler) as c:
        fam = attach_family(a, settings, client=c)
    # The fake rejection is over "Smith US 1,234,567" -> of record.
    assert "US1234567" in fam.us_record_keys
    p = save(a, tmp_path / "a.json")
    assert load(p).family.new_art[0].display == "EP 1111111 A1"


def test_large_family_falls_back_to_batched_biblio(tmp_path, monkeypatch):
    """OPS 413s family/biblio for big families; we fetch members, then biblio in batches."""
    import copy
    monkeypatch.setattr("time.sleep", lambda s: None)
    full = copy.deepcopy(FAMILY)
    members = full["ops:world-patent-data"]["ops:patent-family"]["ops:family-member"]
    # Bulk published-data shape: exchange-documents carrying @country/@doc-number/@kind.
    ex_docs = {}
    for fm in members:
        d = next(x for x in fm["publication-reference"]["document-id"]
                 if x["@document-id-type"] == "docdb")
        k = f'{d["country"]["$"]}.{d["doc-number"]["$"]}.{d["kind"]["$"]}'
        ex = fm.pop("exchange-document")
        ex_docs[k] = {"@country": d["country"]["$"], "@doc-number": d["doc-number"]["$"],
                      "@kind": d["kind"]["$"], "@family-id": fm["@family-id"], **ex}
    bare = full                                    # members without biblio
    posted = []

    def handler(req):
        p = req.url.path
        if p.endswith("/auth/accesstoken"):
            return httpx.Response(200, json={"access_token": "t", "expires_in": 1199})
        if p.endswith("/biblio") and "/family/" in p:
            return httpx.Response(413, text="<code>SERVER.LimitedServerResources</code>")
        if "/family/" in p:
            return httpx.Response(200, json=bare)
        if p.endswith("/published-data/publication/docdb/biblio"):
            ids = req.content.decode().split(",")
            posted.append(len(ids))
            if len(ids) > 2:                          # force the halving path
                return httpx.Response(413, text="SERVER.LimitedServerResources")
            docs = [{"exchange-document": ex_docs[i]} for i in ids]
            return httpx.Response(200, json={"ops:world-patent-data": {
                "exchange-documents": docs if len(docs) > 1 else docs[0]}})
        raise AssertionError(p)

    with _client(tmp_path, handler) as c:
        c.BATCH = 4
        r = c.family("9999999")
    assert max(posted) == 4 and min(posted) <= 2
    by_key = {x.key: x for x in r.citations}
    assert by_key["EP1111111"].best_category == "X"
    assert by_key["US1234567"].of_record_us is True
    assert len(r.members) == 6


def _with_gaps(family: dict) -> dict:
    """Copy of the fixture where OPS marked the EP member 'not found' (as it does for
    many members of large families) - its citations are then missing, not zero."""
    import copy

    data = copy.deepcopy(family)
    for fm in data["ops:world-patent-data"]["ops:patent-family"]["ops:family-member"]:
        if fm["publication-reference"]["document-id"][0]["country"]["$"] == "EP":
            real = fm["exchange-document"]
            fm["exchange-document"] = {"@country": "EP", "@doc-number": "2999999",
                                       "@kind": "A1", "@status": "not found",
                                       "bibliographic-data": {}}
            return data, real
    raise AssertionError("fixture has no EP member")


def test_not_found_members_are_unknown_not_zero():
    data, _ = _with_gaps(FAMILY)
    r = parse_family(data, "9999999")
    ep = next(m for m in r.members if m.country == "EP")
    assert ep.biblio_available is False and ep.citations_note.startswith("not available")
    assert [m.label for m in r.members_without_biblio] == ["EP2999999A1"]


def test_missing_biblio_is_fetched_in_bulk_and_cached(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    data, real = _with_gaps(FAMILY)
    real = {"@country": "EP", "@doc-number": "2999999", "@kind": "A1", **real}
    (tmp_path / "_family").mkdir()
    (tmp_path / "_family" / "US9999999.json").write_text(json.dumps(data))
    posts = []

    def handler(req):
        if req.url.path.endswith("/auth/accesstoken"):
            return httpx.Response(200, json={"access_token": "tok", "expires_in": "1199"})
        assert req.url.path.endswith("/published-data/publication/docdb/biblio")
        posts.append(req.content.decode())
        return httpx.Response(200, json={"ops:world-patent-data": {
            "exchange-documents": {"exchange-document": [real]}}})

    with _client(tmp_path, handler) as c:
        r = c.family("9999999")
        assert posts == ["EP.2999999.A1"]                 # only the gap was requested
        ep = next(m for m in r.members if m.country == "EP")
        assert ep.biblio_available and ep.citation_count == 3
        assert not r.members_without_biblio
        c.family("9999999")                               # filled data is cached
        assert len(posts) == 1


def test_a2_zero_citations_is_explained():
    from fh_analyzer.family import FamilyMember

    m = FamilyMember(country="EP", number="1999602", kind="A2", biblio_available=True)
    assert "A3" in m.citations_note


def test_reparse_upgrades_old_saved_report(tmp_path):
    from fh_analyzer.family import FamilyReport, reparse_cached

    data, _ = _with_gaps(FAMILY)
    (tmp_path / "_family").mkdir()
    (tmp_path / "_family" / "US9999999.json").write_text(json.dumps(data))
    old = parse_family(FAMILY, "9999999").model_dump()
    for m in old["members"]:
        m.pop("biblio_available")              # as saved by the previous version
    old = FamilyReport.model_validate(old)
    assert all(m.biblio_available is None for m in old.members)
    new = reparse_cached(old, tmp_path)
    assert [m.label for m in new.members_without_biblio] == ["EP2999999A1"]
    assert set(old.us_record_keys) <= set(new.us_record_keys)


def test_google_patents_links():
    from fh_analyzer.ui_helpers import google_patents_url, linkify_patents, reference_url

    assert reference_url("Smith (US 9,123,456)") == "https://patents.google.com/patent/US9123456"
    assert reference_url("Park US 2015/0123456 A1").endswith("/US20150123456")
    assert reference_url("EP 1 234 567").endswith("/EP1234567")
    assert reference_url("Smith et al.") is None
    assert google_patents_url("US2005123456").endswith("/US20050123456")
    md = linkify_patents("rejected over Lee (7,123,456) in view of Park (US 2015/0123456).")
    assert "[7,123,456](https://patents.google.com/patent/US7123456)" in md
    assert "[US 2015/0123456](https://patents.google.com/patent/US20150123456)" in md
    assert md.startswith("rejected over Lee (") and md.endswith(").")
