"""Patent family members (US and foreign) and their search-report citations, from EPO
Open Patent Services.

Why: European (and PCT/ISA) examiners often cite better art than the US examiner did, and
they grade it - X (novelty-destroying alone), Y (relevant in combination), A (background).
This module pulls the INPADOC family of a US patent, collects every citation made in any
family member's search/examination, and flags the art that was never of record in the US
case.

Docs: https://developers.epo.org  (OPS v3.2 reference guide)
Auth: OAuth2 client-credentials. Register an app at developers.epo.org and put the
consumer key/secret in EPO_OPS_KEY / EPO_OPS_SECRET. The free tier is ample: a family
lookup is one request.

One call does all the work: GET family/publication/docdb/US.<n>.<kind>/biblio returns every
family member with its bibliographic data, including <references-cited>.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

OPS_URL = "https://ops.epo.org/3.2"
AUTH_PATH = "/auth/accesstoken"

# Search-report categories, most to least significant for invalidity.
CATEGORY_RANK = {"X": 0, "Y": 1, "E": 2, "P": 3, "L": 4, "O": 5, "T": 6, "D": 7, "A": 8}
CATEGORY_HELP = {
    "X": "relevant alone (novelty / inventive step)",
    "Y": "relevant in combination with another Y document",
    "A": "general background / state of the art",
    "E": "earlier patent document published on or after the filing date",
    "P": "intermediate document (published between priority and filing date)",
    "L": "cited for other reasons (e.g. doubt on priority)",
    "O": "non-written disclosure",
    "T": "theory or principle underlying the invention",
    "D": "cited in the application itself",
}


class OPSError(RuntimeError):
    pass


class OPSTooLarge(OPSError):
    """413 SERVER.LimitedServerResources: ask for less data per request."""


# ------------------------------------------------------------------ models

class FamilyMember(BaseModel):
    country: str
    number: str
    kind: str
    pub_date: date | None = None
    family_id: str | None = None
    simple_family: bool = Field(
        default=True,
        description="Same DOCDB simple family as the US patent (same priorities). False = "
        "extended INPADOC member only (e.g. a CIP or a sibling sharing one priority).",
    )
    application: str | None = None       # docdb application number, e.g. EP 14123456
    title: str | None = None
    citation_count: int = 0
    # False when EPO OPS returned no bibliographic record for this publication (status
    # "not found"), so its citations are unknown - not zero. None = saved by an older
    # version that did not record this.
    biblio_available: bool | None = None

    @property
    def citations_note(self) -> str:
        """Why a member shows no citations, so '0' is never ambiguous."""
        if self.biblio_available is None:
            return "unknown (looked up with an older version) - refresh"
        if not self.biblio_available:
            return "not available: EPO OPS has no bibliographic record for this publication"
        if self.citation_count:
            return ""
        if self.country in ("EP", "WO") and self.kind == "A2":
            return "published without the search report; its citations are on the A3"
        return "no citations listed for this publication in DOCDB"

    @property
    def label(self) -> str:
        return f"{self.country}{self.number}{self.kind}"

    @property
    def espacenet_url(self) -> str:
        return f"https://worldwide.espacenet.com/patent/search?q=pn%3D{self.label}"

    @property
    def register_url(self) -> str | None:
        """EP Register (search opinion, exam reports, all documents) for EP members."""
        if self.country == "EP" and self.application:
            return (f"https://register.epo.org/application?number=EP{self.application}"
                    f"&lng=en&tab=doclist")
        return None


class CitingInstance(BaseModel):
    """One place a document was cited: which family member, at which phase, how graded."""

    member: str                         # e.g. EP2734567A1
    phase: str | None = None            # search | examination | opposition | isr | applicant ...
    cited_by: str | None = None         # examiner | applicant | third-party
    categories: list[str] = Field(default_factory=list)
    rel_claims: str | None = None
    passages: list[str] = Field(default_factory=list)


class ForeignCitation(BaseModel):
    key: str                            # normalized match key, e.g. "US2005123456"; NPL: "NPL:..."
    display: str                        # e.g. "US 2005/0123456 A1"
    country: str | None = None
    number: str | None = None
    kind: str | None = None
    npl: bool = False
    npl_text: str | None = None
    cited_in: list[CitingInstance] = Field(default_factory=list)
    # True = cited on the face of the US patent (examiner or IDS) or named in an extracted
    # rejection; False = not of record in the US case; None = could not tell (NPL).
    of_record_us: bool | None = None

    @property
    def categories(self) -> list[str]:
        cats = {c for ci in self.cited_in for c in ci.categories}
        return sorted(cats, key=lambda c: CATEGORY_RANK.get(c, 99))

    @property
    def best_category(self) -> str | None:
        return self.categories[0] if self.categories else None

    @property
    def rank(self) -> int:
        return CATEGORY_RANK.get(self.best_category or "", 50)

    @property
    def foreign_only(self) -> bool:
        """Cited somewhere other than a US publication of this family."""
        return any(not ci.member.startswith("US") for ci in self.cited_in)

    @property
    def espacenet_url(self) -> str | None:
        if self.npl or not self.country:
            return None
        return (f"https://worldwide.espacenet.com/patent/search?q=pn%3D"
                f"{self.country}{self.number}{self.kind or ''}")


class FamilyReport(BaseModel):
    subject: str                        # the US publication the family was built from
    family_id: str | None = None
    members: list[FamilyMember] = Field(default_factory=list)
    citations: list[ForeignCitation] = Field(default_factory=list)
    us_record_keys: list[str] = Field(default_factory=list)
    created: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def offices(self) -> list[str]:
        return sorted({m.country for m in self.members})

    @property
    def members_without_biblio(self) -> list[FamilyMember]:
        return [m for m in self.members if m.biblio_available is not True]

    @property
    def new_art(self) -> list[ForeignCitation]:
        """Patent art cited in some family member but never of record in the US case."""
        return [c for c in self.citations if c.of_record_us is False]


# ------------------------------------------------------------------ number handling

_US_PUB = re.compile(r"^(20\d\d)0(\d{6})$")   # 2005/0123456 -> docdb 2005123456


def match_key(country: str, number: str) -> str:
    """Kind-less key used to decide whether two citations are the same document.

    US pre-grant publications are written 2005/0123456 (11 digits) but stored in DOCDB as
    2005123456 (10); both reduce to the DOCDB form. Leading zeros are dropped."""
    country = country.upper().strip()
    digits = re.sub(r"[^0-9A-Z]", "", number.upper())
    if country == "US":
        m = _US_PUB.match(digits)
        if m:
            digits = m.group(1) + m.group(2)
        digits = re.sub(r"^(RE|D|PP)?0+", lambda x: x.group(1) or "", digits)
    else:
        digits = digits.lstrip("0")
    return f"{country}{digits}"


def display_number(country: str, number: str, kind: str | None) -> str:
    k = f" {kind}" if kind else ""
    if country == "US" and re.fullmatch(r"20\d{8}", number):
        return f"US {number[:4]}/0{number[4:]}{k}"
    if country == "US" and number.isdigit():
        return f"US {int(number):,}{k}"
    if country == "WO" and re.fullmatch(r"(19|20)\d{8}", number):
        return f"WO {number[:4]}/{number[4:]}{k}"
    return f"{country} {number}{k}"


# Patent numbers as examiners and applicants write them in OAs and remarks.
_REF_PATTERNS = [
    re.compile(r"\b(US|EP|WO|JP|CN|KR|DE|GB|FR|CA|AU|TW)\s*[-–]?\s*"
               r"((?:RE|D)?\d[\d,/.\-\s]{3,}\d)\b", re.I),
    re.compile(r"(?<![\w/])()(\d{1,2},\d{3},\d{3})\b"),        # bare "9,123,456" -> US
    re.compile(r"(?<![\w/])()((?:19|20)\d\d/0\d{6})\b"),       # bare "2015/0123456" -> US
]


def keys_from_text(text: str) -> set[str]:
    """Pull patent-number match keys out of free text like 'Smith (US 9,123,456)'."""
    out: set[str] = set()
    for pat in _REF_PATTERNS:
        for m in pat.finditer(text or ""):
            country = (m.group(1) or "US").upper()
            num = re.sub(r"[\s,.\-/]", "", m.group(2))
            if len(num) >= 5:
                out.add(match_key(country, num))
    return out


# ------------------------------------------------------------------ OPS JSON helpers
# OPS JSON is a mechanical conversion of the XML: attributes are "@name", text is "$",
# and a repeated element becomes a list only when it actually repeats.

def _as_list(v) -> list:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _text(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, dict):
        return v.get("$")
    if isinstance(v, list):
        parts = [_text(x) for x in v]
        return " ".join(p for p in parts if p) or None
    return str(v)


def _docdb(ref: dict | None) -> tuple[str, str, str | None, str | None] | None:
    """(country, number, kind, date) of the docdb-format document-id in a reference."""
    ids = _as_list((ref or {}).get("document-id"))
    d = next((x for x in ids if x.get("@document-id-type") == "docdb"), None)
    if d is None:
        return None
    c, n = _text(d.get("country")), _text(d.get("doc-number"))
    if not c or not n:
        return None
    return c, n, _text(d.get("kind")), _text(d.get("date"))


def _find_exchange_docs(node) -> list[dict]:
    """Every exchange-document in a published-data response, however it is nested."""
    out: list[dict] = []
    if isinstance(node, dict):
        if "bibliographic-data" in node and "@country" in node:
            return [node]
        for v in node.values():
            out += _find_exchange_docs(v)
    elif isinstance(node, list):
        for v in node:
            out += _find_exchange_docs(v)
    return out


def _biblio_missing(ed: dict | None) -> bool:
    """True if an exchange-document carries no real bibliographic data. OPS marks these
    with status="not found" - common in large family/biblio responses."""
    return not ed or ed.get("@status") == "not found" or "bibliographic-data" not in ed


def _parse_date(s: str | None) -> date | None:
    try:
        return datetime.strptime(s, "%Y%m%d").date() if s else None
    except ValueError:
        return None


def _title(bib: dict) -> str | None:
    titles = _as_list(bib.get("invention-title"))
    en = next((t for t in titles if isinstance(t, dict) and t.get("@lang") == "en"), None)
    return _text(en or (titles[0] if titles else None))


_CAT = re.compile(r"[XYAPELOTD]")


def _parse_citation(cit: dict, member: str) -> tuple[ForeignCitation, CitingInstance] | None:
    cats: list[str] = []
    for c in _as_list(cit.get("category")):
        cats += _CAT.findall((_text(c) or "").upper())
    passages = []
    for rp in _as_list(cit.get("rel-passage")):
        for p in _as_list(rp.get("passage") if isinstance(rp, dict) else rp):
            t = (_text(p) or "").strip(" *")
            if t:
                passages.append(t)
    inst = CitingInstance(
        member=member, phase=cit.get("@cited-phase"), cited_by=cit.get("@cited-by"),
        categories=list(dict.fromkeys(cats)), rel_claims=_text(cit.get("rel-claims")),
        passages=passages,
    )
    if "patcit" in cit:
        dd = _docdb(cit["patcit"])
        if dd is None:
            return None
        country, number, kind, _ = dd
        fc = ForeignCitation(key=match_key(country, number),
                             display=display_number(country, number, kind),
                             country=country, number=number, kind=kind)
        return fc, inst
    if "nplcit" in cit:
        txt = (_text(cit["nplcit"].get("text")) or "").strip()
        if not txt:
            return None
        key = "NPL:" + re.sub(r"\W+", " ", txt.lower()).strip()[:120]
        fc = ForeignCitation(key=key, display=txt[:160], npl=True, npl_text=txt)
        return fc, inst
    return None


def parse_family(data: dict, us_number: str) -> FamilyReport:
    """Turn a family/.../biblio response into a FamilyReport (US-record flags not yet set)."""
    wpd = data.get("ops:world-patent-data", data)
    fam = wpd.get("ops:patent-family", {})
    subject_key = match_key("US", us_number)
    members: list[FamilyMember] = []
    cited: dict[str, ForeignCitation] = {}
    us_record: set[str] = set()
    subject_family_id = None
    raw_members = _as_list(fam.get("ops:family-member"))

    # First pass: find the subject's DOCDB family id (its simple family).
    for fm in raw_members:
        dd = _docdb(fm.get("publication-reference"))
        if dd and dd[0] == "US" and match_key("US", dd[1]) == subject_key:
            subject_family_id = fm.get("@family-id")
            break

    for fm in raw_members:
        dd = _docdb(fm.get("publication-reference"))
        if dd is None:
            continue
        country, number, kind, pdate = dd
        appl = _docdb(fm.get("application-reference"))
        docs = _as_list(fm.get("exchange-document"))
        ed = docs[0] if docs else None
        bib = (ed or {}).get("bibliographic-data", {}) or {}
        m = FamilyMember(
            country=country, number=number, kind=kind or "", pub_date=_parse_date(pdate),
            family_id=fm.get("@family-id"),
            simple_family=(subject_family_id is None
                           or fm.get("@family-id") == subject_family_id),
            application=appl[1] if appl else None, title=_title(bib),
            biblio_available=not _biblio_missing(ed),
        )
        is_subject = country == "US" and match_key("US", number) == subject_key
        cits = _as_list((bib.get("references-cited") or {}).get("citation"))
        m.citation_count = len(cits)
        members.append(m)
        for cit in cits:
            parsed = _parse_citation(cit, m.label)
            if parsed is None:
                continue
            fc, inst = parsed
            if is_subject:
                # Everything on the face of the US patent was before the US examiner,
                # whether the examiner cited it or the applicant did (IDS).
                us_record.add(fc.key)
            existing = cited.setdefault(fc.key, fc)
            existing.cited_in.append(inst)

    members.sort(key=lambda x: (x.country != "US", x.country, x.pub_date or date.min))
    report = FamilyReport(subject=f"US{us_number}", family_id=subject_family_id,
                          members=members, citations=list(cited.values()),
                          us_record_keys=sorted(us_record))
    mark_us_record(report, [])
    return report


def mark_us_record(report: FamilyReport, extra_refs: list[str]) -> FamilyReport:
    """Set of_record_us on every citation. `extra_refs` are free-text reference names from
    the US prosecution (e.g. rejection references extracted by the LLM)."""
    keys = set(report.us_record_keys)
    for r in extra_refs:
        keys |= keys_from_text(r)
    report.us_record_keys = sorted(keys)
    for c in report.citations:
        c.of_record_us = None if c.npl else (c.key in keys)
    report.citations.sort(key=lambda c: (c.of_record_us is not False, c.npl, c.rank,
                                         -len(c.cited_in), c.display))
    return report


# ------------------------------------------------------------------ client

class OPSClient:
    def __init__(self, key: str | None, secret: str | None, cache_dir: Path, *,
                 transport: httpx.BaseTransport | None = None, max_retries: int = 4) -> None:
        if not key or not secret:
            raise OPSError("EPO_OPS_KEY / EPO_OPS_SECRET are not set. Register a free app at "
                           "https://developers.epo.org and add them to .env.")
        self._basic = base64.b64encode(f"{key}:{secret}".encode()).decode()
        self.cache_dir = Path(cache_dir)
        self.max_retries = max_retries
        self._token: str | None = None
        self._token_expiry = 0.0
        self._http = httpx.Client(base_url=OPS_URL, timeout=httpx.Timeout(60.0, connect=15.0),
                                  transport=transport, follow_redirects=True)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> OPSClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- auth ----------------------------------------------------------------------
    def _auth(self) -> str:
        if self._token and time.time() < self._token_expiry - 30:
            return self._token
        r = self._http.post(AUTH_PATH, data={"grant_type": "client_credentials"},
                            headers={"Authorization": f"Basic {self._basic}"})
        if r.status_code in (400, 401, 403):
            raise OPSError(f"EPO OPS rejected the key/secret ({r.status_code}). "
                           "Check EPO_OPS_KEY / EPO_OPS_SECRET.")
        r.raise_for_status()
        body = r.json()
        self._token = body["access_token"]
        self._token_expiry = time.time() + float(body.get("expires_in", 1200))
        return self._token

    # -- low-level GET with retry/backoff and one token refresh ---------------------
    def _get(self, path: str, *, post: str | None = None) -> httpx.Response | None:
        """GET (or POST a comma-separated number list, for bulk retrieval).
        Returns None on 404 (OPS: no such document); raises OPSTooLarge on 413."""
        delay, refreshed = 2.0, False
        for attempt in range(1, self.max_retries + 1):
            headers = {"Authorization": f"Bearer {self._auth()}",
                       "Accept": "application/json"}
            try:
                if post is None:
                    r = self._http.get(path, headers=headers)
                else:
                    r = self._http.post(path, content=post.encode(),
                                        headers={**headers, "Content-Type": "text/plain"})
            except httpx.TransportError as e:
                if attempt == self.max_retries:
                    raise OPSError(f"Network error calling EPO OPS: {e}") from e
                time.sleep(delay)
                delay *= 2
                continue
            body = r.text[:500]
            if r.status_code in (400, 401) and "access_token" in body.lower() and not refreshed:
                self._token, refreshed = None, True           # expired token: refresh once
                continue
            if r.status_code == 404:
                return None
            if r.status_code == 413:
                raise OPSTooLarge(f"EPO OPS: response too large for {path}")
            if r.status_code == 403:
                why = r.headers.get("X-Rejection-Reason") or body
                raise OPSError(f"EPO OPS refused the request (quota or throttling): {why}")
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == self.max_retries:
                    break
                wait = float(r.headers.get("Retry-After", delay))
                log.warning("OPS %s on %s; retrying in %.0fs", r.status_code, path, wait)
                time.sleep(min(wait, 60))
                delay *= 2
                continue
            if r.status_code >= 400:
                raise OPSError(f"EPO OPS error {r.status_code} on {path}: {body}")
            return r
        raise OPSError(f"EPO OPS request failed after {self.max_retries} attempts: {path}")

    # -- public API ----------------------------------------------------------------
    def family_raw(self, us_number: str, retry_missing: bool = False) -> dict:
        """INPADOC family with biblio (incl. citations) for a US patent number. Cached.
        `retry_missing` re-requests members whose biblio OPS could not supply last time."""
        n = _us_digits(us_number)
        cache = self.cache_dir / "_family" / f"US{n}.json"
        if cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
            if FILL_MARK not in data or (retry_missing and data[FILL_MARK]["unavailable"]):
                # Cached before gap-filling existed (or asked to retry): fill, re-cache.
                data = self._fill_missing_biblio(data)
                cache.write_text(json.dumps(data, indent=1), encoding="utf-8")
            return data
        # US grants since 2001 are B1 (no pre-grant pub) or B2; fall back to epodoc,
        # which needs no kind code.
        candidates = [f"docdb/{quote(f'US.{n}.{k}')}" for k in ("B2", "B1", "E")]
        candidates.append(f"epodoc/US{n}")
        for c in candidates:
            base = f"/rest-services/family/publication/{c}"
            try:
                r = self._get(f"{base}/biblio")
                data = r.json() if r is not None else None
            except OPSTooLarge:
                # Large family: OPS won't return every member's biblio in one response.
                # Get the bare member list, then the biblio in batches.
                log.info("Family of US%s too large for one request; fetching in batches", n)
                r = self._get(base)
                data = self._with_batched_biblio(r.json()) if r is not None else None
            if data is not None:
                if FILL_MARK not in data:
                    data = self._fill_missing_biblio(data)
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(data, indent=1), encoding="utf-8")
                return data
        raise OPSError(f"EPO OPS has no family record for US {us_number}")

    BATCH = 20
    MAX_SINGLE = 80   # cap on one-by-one retries, to stay well inside the OPS quota

    def _with_batched_biblio(self, data: dict) -> dict:
        """A bare family list (no biblio): fetch every member's biblio in bulk calls,
        producing the same shape the one-shot family/biblio call returns."""
        return self._fill_missing_biblio(data)

    def _fill_missing_biblio(self, data: dict) -> dict:
        """Fetch biblio for family members OPS left empty or marked status="not found".

        For large families the one-shot family/biblio response comes back with many
        members marked "not found" - typically EP A3/B1, WO A3 and granted US B2s, i.e.
        exactly the publications that carry the search-report citations. Re-request those
        through published-data: in bulk first, then one by one for any still missing."""
        fam = data.get("ops:world-patent-data", data).get("ops:patent-family", {})
        members = _as_list(fam.get("ops:family-member"))
        need: dict[str, list[dict]] = {}
        for fm in members:
            docs = _as_list(fm.get("exchange-document"))
            if not _biblio_missing(docs[0] if docs else None):
                continue
            dd = _docdb(fm.get("publication-reference"))
            if dd and dd[2]:
                need.setdefault(f"{dd[0]}.{dd[1]}.{dd[2]}", []).append(fm)
        found: dict[str, dict] = {}
        todo = list(need)
        for i in range(0, len(todo), self.BATCH):
            found.update(self._biblio_bulk(todo[i:i + self.BATCH]))
        for pid in [p for p in todo if p not in found][:self.MAX_SINGLE]:
            doc = self._biblio_one(pid)
            if doc is not None:
                found[pid] = doc
        for pid, fms in need.items():
            if pid in found:
                for fm in fms:
                    fm["exchange-document"] = found[pid]
        fam["ops:family-member"] = members
        data[FILL_MARK] = {"requested": len(need), "recovered": len(found),
                           "unavailable": sorted(set(need) - set(found))}
        if need:
            log.info("OPS biblio gap-fill: %d missing, %d recovered", len(need), len(found))
        return data

    def _biblio_bulk(self, ids: list[str]) -> dict[str, dict]:
        """docdb id -> exchange-document (only documents OPS actually found). Halves the
        batch on 413; a single document that is still too large is skipped."""
        try:
            r = self._get("/rest-services/published-data/publication/docdb/biblio",
                          post=",".join(ids))
        except OPSTooLarge:
            if len(ids) == 1:
                log.warning("OPS: biblio for %s too large; skipped", ids[0])
                return {}
            mid = len(ids) // 2
            return {**self._biblio_bulk(ids[:mid]), **self._biblio_bulk(ids[mid:])}
        if r is None:
            return {}
        return _docs_by_id(r.json())

    def _biblio_one(self, pid: str) -> dict | None:
        try:
            r = self._get(f"/rest-services/published-data/publication/docdb/{quote(pid)}"
                          "/biblio")
        except OPSTooLarge:
            return None
        return _docs_by_id(r.json()).get(pid) if r is not None else None

    def family(self, us_number: str, extra_refs: list[str] | None = None,
               retry_missing: bool = False) -> FamilyReport:
        n = _us_digits(us_number)
        report = parse_family(self.family_raw(n, retry_missing), n)
        return mark_us_record(report, extra_refs or [])


FILL_MARK = "fha:biblio-fill"   # set on cached family data once gaps have been filled


def _docs_by_id(payload) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for doc in _find_exchange_docs(payload):
        if _biblio_missing(doc):
            continue
        out.setdefault(f"{doc.get('@country')}.{doc.get('@doc-number')}.{doc.get('@kind')}",
                       doc)
    return out


def reparse_cached(report: FamilyReport, cache_dir: Path) -> FamilyReport:
    """Rebuild a saved FamilyReport from the raw OPS response on disk (no network), keeping
    the US-record matches it already had. Used to upgrade reports saved by older versions."""
    raw = Path(cache_dir) / "_family" / f"{report.subject}.json"
    if not raw.exists():
        return report
    n = report.subject.removeprefix("US")
    fresh = parse_family(json.loads(raw.read_text(encoding="utf-8")), n)
    fresh.us_record_keys = sorted(set(fresh.us_record_keys) | set(report.us_record_keys))
    return mark_us_record(fresh, [])


def _us_digits(us_number: str) -> str:
    """'US 8,046,721 B2' -> '8046721'; 'RE49,123' -> 'RE49123'."""
    s = re.sub(r"[\s,]", "", us_number.upper())
    s = re.sub(r"^US", "", s)
    return re.sub(r"(B1|B2|E)$", "", s)


def us_refs_from_findings(findings) -> list[str]:
    """Reference names the US prosecution mentions (rejections, distinguished art)."""
    out: list[str] = []
    for f in findings:
        out += f.meta.get("references", []) or []
        out += f.meta.get("distinguished_art", []) or []
    return out
