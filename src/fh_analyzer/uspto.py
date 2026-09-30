"""Thin client for the USPTO Open Data Portal (ODP) Patent File Wrapper API.

Docs: https://data.uspto.gov/apis/patent-file-wrapper/search
Auth: an `X-API-KEY` header. Coverage: applications filed on/after 2001-01-01.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import date
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from .doc_codes import DocCategory, classify

log = logging.getLogger(__name__)

BASE_URL = "https://api.uspto.gov"


class USPTOError(RuntimeError):
    pass


class FileWrapperDoc(BaseModel):
    """One document in the image file wrapper."""

    doc_id: str = Field(description="USPTO documentIdentifier")
    code: str
    description: str
    official_date: date | None
    direction: str | None = None  # INCOMING / OUTGOING / INTERNAL
    pdf_url: str | None = None
    page_count: int | None = None
    category: DocCategory = DocCategory.OTHER

    @property
    def label(self) -> str:
        d = self.official_date.isoformat() if self.official_date else "n.d."
        return f"{d} {self.code} – {self.description}"


class ApplicationInfo(BaseModel):
    application_number: str
    patent_number: str | None = None
    title: str | None = None
    filing_date: date | None = None
    grant_date: date | None = None
    examiner: str | None = None
    art_unit: str | None = None
    status: str | None = None


# ---------------------------------------------------------------- number handling

def normalize_number(raw: str) -> tuple[str, str]:
    """Return ("patent"|"application", digits) for user input.

    Accepts "US 10,123,456 B2", "10123456", "RE49,123", "16/123,456", "16123456".
    Application numbers are 8 digits and are usually written with a series slash;
    utility patent numbers are currently 7-8 digits, so an 8-digit bare number is
    ambiguous - we treat it as a patent unless it contains a "/".
    """
    s = raw.strip().upper()
    if "/" in s:
        digits = re.sub(r"\D", "", s)
        if len(digits) != 8:
            raise ValueError(f"Application number should have 8 digits: {raw!r}")
        return "application", digits
    s = re.sub(r"^US", "", s)
    s = re.sub(r"\s*[AB][12]$", "", s)  # kind code
    s = re.sub(r"[\s,\-]", "", s)
    if re.fullmatch(r"(RE|D|PP)?\d{5,8}", s):
        return "patent", s
    raise ValueError(f"Unrecognized patent/application number: {raw!r}")


def _parse_date(v: str | None) -> date | None:
    if not v:
        return None
    try:
        return date.fromisoformat(v[:10])
    except ValueError:
        return None


# ---------------------------------------------------------------- client

class ODPClient:
    def __init__(
        self,
        api_key: str,
        cache_dir: Path,
        *,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 5,
    ) -> None:
        if not api_key:
            raise USPTOError("USPTO_API_KEY is not set (see .env.example)")
        self.cache_dir = Path(cache_dir)
        self.max_retries = max_retries
        self._http = httpx.Client(
            base_url=BASE_URL,
            headers={"X-API-KEY": api_key, "Accept": "application/json"},
            timeout=httpx.Timeout(60.0, connect=15.0),
            follow_redirects=True,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ODPClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- low-level with retry/backoff ------------------------------------------------
    def _get(self, url: str, **kw) -> httpx.Response:
        delay = 2.0
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._http.get(url, **kw)
            except httpx.TransportError as e:
                if attempt == self.max_retries:
                    raise USPTOError(f"Network error calling {url}: {e}") from e
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == self.max_retries:
                    break
                wait = float(r.headers.get("Retry-After", delay))
                log.warning("ODP %s on %s; retrying in %.0fs", r.status_code, url, wait)
                time.sleep(min(wait, 60))
                delay *= 2
                continue
            if r.status_code == 403:
                raise USPTOError("ODP rejected the API key (403). Check USPTO_API_KEY.")
            if r.status_code == 404:
                raise USPTOError(f"Not found at USPTO ODP: {url}")
            r.raise_for_status()
            return r
        raise USPTOError(f"ODP request failed after {self.max_retries} attempts: {url}")

    def _cached_json(self, path: Path, url: str, **kw) -> dict:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        data = self._get(url, **kw).json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        return data

    # -- public API ----------------------------------------------------------------
    def resolve(self, number: str) -> ApplicationInfo:
        kind, digits = normalize_number(number)
        if kind == "application":
            data = self._cached_json(
                self.cache_dir / digits / "application.json",
                f"/api/v1/patent/applications/{digits}",
            )
        else:
            try:
                data = self._cached_json(
                    self.cache_dir / "_lookup" / f"{digits}.json",
                    "/api/v1/patent/applications/search",
                    params={"q": f"applicationMetaData.patentNumber:{digits}", "limit": 5},
                )
            except USPTOError as e:
                if "Not found" not in str(e):
                    raise
                data = {}          # ODP answers an empty search with 404
        bag = data.get("patentFileWrapperDataBag") or []
        if not bag:
            raise USPTOError(
                f"The USPTO Open Data Portal has no file wrapper for {number!r}. ODP's "
                "file-wrapper data mainly covers applications filed from 2001 on, so older "
                "patents are often missing. Try the application number instead, or download "
                "the file history PDFs from Patent Center and run: "
                "fh-analyze --folder <folder of PDFs>")
        return self._parse_app(bag[0])

    @staticmethod
    def _parse_app(rec: dict) -> ApplicationInfo:
        md = rec.get("applicationMetaData", {}) or {}
        return ApplicationInfo(
            application_number=rec.get("applicationNumberText", ""),
            patent_number=md.get("patentNumber"),
            title=md.get("inventionTitle"),
            filing_date=_parse_date(md.get("filingDate")),
            grant_date=_parse_date(md.get("grantDate")),
            examiner=md.get("examinerNameText"),
            art_unit=md.get("groupArtUnitNumber"),
            status=md.get("applicationStatusDescriptionText"),
        )

    def list_documents(self, app_no: str) -> list[FileWrapperDoc]:
        data = self._cached_json(
            self.cache_dir / app_no / "documents.json",
            f"/api/v1/patent/applications/{app_no}/documents",
        )
        return parse_documents(data)

    def continuity(self, app_no: str) -> dict:
        """Parent/child continuity for an application (CON / CIP / DIV / provisional)."""
        return self._cached_json(
            self.cache_dir / app_no / "continuity.json",
            f"/api/v1/patent/applications/{app_no}/continuity",
        )

    def download_pdf(self, app_no: str, doc: FileWrapperDoc) -> Path:
        out = self.cache_dir / app_no / "pdf" / f"{doc.doc_id}.pdf"
        if out.exists() and out.stat().st_size > 0:
            return out
        if not doc.pdf_url:
            raise USPTOError(f"No PDF download option for {doc.doc_id} ({doc.code})")
        r = self._get(doc.pdf_url, headers={"Accept": "application/pdf"})
        if not r.content.startswith(b"%PDF"):
            raise USPTOError(f"Download for {doc.doc_id} did not return a PDF")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(r.content)
        return out


def parse_documents(data: dict) -> list[FileWrapperDoc]:
    docs: list[FileWrapperDoc] = []
    for d in data.get("documentBag", []) or []:
        pdf = next(
            (o for o in d.get("downloadOptionBag", []) or []
             if (o.get("mimeTypeIdentifier") or "").upper() == "PDF"),
            None,
        )
        code = d.get("documentCode", "") or ""
        desc = d.get("documentCodeDescriptionText", "") or ""
        docs.append(
            FileWrapperDoc(
                doc_id=d.get("documentIdentifier", ""),
                code=code,
                description=desc,
                official_date=_parse_date(d.get("officialDate")),
                direction=d.get("directionCategory"),
                pdf_url=pdf.get("downloadUrl") if pdf else None,
                page_count=pdf.get("pageTotalQuantity") if pdf else None,
                category=classify(code, desc),
            )
        )
    docs.sort(key=lambda x: (x.official_date or date.min, x.doc_id))
    return docs
