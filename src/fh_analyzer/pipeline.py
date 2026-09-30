"""End-to-end orchestration: fetch -> OCR -> per-document extraction -> synthesis -> grounding."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from . import completeness, prompts
from .claims import ClaimHistory, ClaimSnapshot, build_histories
from .config import Settings
from .costs import RunCost, call_context
from .doc_codes import SUBSTANTIVE, DocCategory, classify
from .family import FamilyReport, OPSClient, us_refs_from_findings
from .grounding import Finding, GroundingReport, build_report, check_citation, judge_support
from .llm import LLM
from .models import (
    AllowanceExtraction,
    Citation,
    ClaimsExtraction,
    InterviewExtraction,
    OfficeActionExtraction,
    ResponseCoreExtraction,
    ResponseExtraction,
    Synthesis,
)
from .ocr import DocText, extract_document, render_for_llm
from .priority import PriorityReport
from .uspto import ApplicationInfo, FileWrapperDoc, ODPClient

log = logging.getLogger(__name__)

Progress = Callable[[str, float], None]

# category -> (system prompt, output schema)
ROUTES: dict[DocCategory, tuple[str, type[BaseModel]]] = {
    DocCategory.REJECTION: (prompts.OFFICE_ACTION, OfficeActionExtraction),
    DocCategory.ADVISORY: (prompts.OFFICE_ACTION, OfficeActionExtraction),
    DocCategory.RESTRICTION: (prompts.OFFICE_ACTION, OfficeActionExtraction),
    DocCategory.APPLICANT_RESPONSE: (prompts.RESPONSE, ResponseExtraction),
    DocCategory.APPEAL: (prompts.RESPONSE, ResponseExtraction),
    DocCategory.CLAIMS: (prompts.CLAIMS, ClaimsExtraction),
    DocCategory.ALLOWANCE: (prompts.ALLOWANCE, AllowanceExtraction),
    DocCategory.INTERVIEW: (prompts.INTERVIEW, InterviewExtraction),
}

# category -> which PromptSet field to use (so a different prompt version can be swapped in)
PROMPT_FIELD: dict[DocCategory, str] = {
    DocCategory.REJECTION: "office_action", DocCategory.ADVISORY: "office_action",
    DocCategory.RESTRICTION: "office_action", DocCategory.APPLICANT_RESPONSE: "response",
    DocCategory.APPEAL: "response", DocCategory.CLAIMS: "claims",
    DocCategory.ALLOWANCE: "allowance", DocCategory.INTERVIEW: "interview",
}

MAX_DOC_CHARS = 400_000  # ~100k tokens; a single IFW document essentially never exceeds this


class OCRStats(BaseModel):
    pages: int = 0
    ocr_pages: int = 0
    text_layer_pages: int = 0
    low_quality_pages: int = 0
    mean_ocr_confidence: float | None = None


class Analysis(BaseModel):
    app: ApplicationInfo
    documents: list[FileWrapperDoc]
    analyzed_doc_ids: list[str]
    extractions: dict[str, dict] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)
    # doc_id -> completeness issues still unresolved after the retry
    completeness: dict[str, list[str]] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)
    claim_histories: list[ClaimHistory] = Field(default_factory=list)
    synthesis: Synthesis | None = None
    grounding: GroundingReport | None = None
    ocr: OCRStats = Field(default_factory=OCRStats)
    usage: dict = Field(default_factory=dict)
    # Page texts are stored so the analysis is self-contained: grounding can be re-run and
    # every citation inspected from the JSON alone, without re-downloading or re-OCRing.
    texts: dict[str, DocText] = Field(default_factory=dict)
    pdf_dir: str | None = None
    # Patent family (US + foreign) + search-report citations from EPO OPS.
    # Optional; see family.py.
    family: FamilyReport | None = None
    # Claim-element support across the continuity chain (see priority.py). Optional.
    priority: PriorityReport | None = None
    # Dollar cost of the analysis run (see costs.py). Priority-step costs are kept on the
    # PriorityReport, since that step is run separately.
    cost: RunCost | None = None
    # Optional analyses switched on for this run, e.g. {"estoppel": False}. Empty for runs
    # made before the switch existed (those always included estoppel).
    features: dict[str, bool] = Field(default_factory=dict)
    model: str = ""
    prompt_version: str = prompts.PROMPT_VERSION
    created: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def doc(self, doc_id: str) -> FileWrapperDoc | None:
        return next((d for d in self.documents if d.doc_id == doc_id), None)


def _noop(msg: str, frac: float) -> None:
    log.info("[%3.0f%%] %s", frac * 100, msg)


# ------------------------------------------------------------------ acquisition

def fetch_from_uspto(number: str, settings: Settings, progress: Progress = _noop,
                     client: ODPClient | None = None):
    """Resolve the number, list the wrapper, download + OCR the substantive documents."""
    own = client is None
    client = client or ODPClient(settings.uspto_api_key or "", settings.cache_dir)
    try:
        progress(f"Looking up {number}", 0.0)
        app = client.resolve(number)
        docs = client.list_documents(app.application_number)
        todo = [d for d in docs if d.category in SUBSTANTIVE and d.pdf_url]
        progress(f"{len(docs)} documents in file wrapper; {len(todo)} substantive", 0.05)
        texts: dict[str, DocText] = {}
        for i, d in enumerate(todo):
            progress(f"Downloading + OCR {d.label}", 0.05 + 0.45 * i / max(len(todo), 1))
            pdf = client.download_pdf(app.application_number, d)
            texts[d.doc_id] = extract_document(pdf, d.doc_id, settings)
        return app, docs, texts
    finally:
        if own:
            client.close()


_FNAME = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2}).*?-(?P<code>[A-Z0-9.]+)\.pdf$", re.I)


def load_local_folder(folder: Path, settings: Settings, progress: Progress = _noop):
    """Offline mode: analyse a folder of file-history PDFs (e.g. a Patent Center ZIP).

    Filenames like '16123456-2019-05-01-00012-CTNF.pdf' give the date and doc code;
    anything else is classified by filename keywords.
    """
    pdfs = sorted(Path(folder).glob("*.pdf"))
    docs, texts = [], {}
    for i, pdf in enumerate(pdfs):
        m = _FNAME.search(pdf.name)
        code = m.group("code").upper() if m else pdf.stem.split("-")[-1].upper()
        dt = date.fromisoformat(m.group("date")) if m else None
        desc = pdf.stem.replace("_", " ")
        d = FileWrapperDoc(doc_id=pdf.stem, code=code, description=desc, official_date=dt,
                           category=classify(code, desc))
        docs.append(d)
        if d.category in SUBSTANTIVE:
            progress(f"OCR {pdf.name}", 0.5 * i / max(len(pdfs), 1))
            texts[d.doc_id] = extract_document(pdf, d.doc_id, settings,
                                               cache_dir=Path(folder) / ".fha_text")
    docs.sort(key=lambda x: (x.official_date or date.min, x.doc_id))
    app = ApplicationInfo(application_number=Path(folder).name)
    return app, docs, texts


# ------------------------------------------------------------------ analysis

def _extract_one(doc: FileWrapperDoc, text: DocText, llm: LLM, cache_dir: Path | None,
                 prompt_set: prompts.PromptSet | None = None, completeness_retry: bool = True,
                 estoppel: bool = True):
    """Extract one document. Returns (extraction, unresolved completeness issues)."""
    ps = prompt_set or prompts.CURRENT
    _, schema = ROUTES[doc.category]
    system = getattr(ps, PROMPT_FIELD[doc.category])
    core = schema is ResponseExtraction and not estoppel
    if core:                          # applicant filing without the estoppel analysis
        schema, system = ResponseCoreExtraction, ps.response_core
    cache = None
    if cache_dir:
        safe_model = re.sub(r"\W", "_", llm.model)
        safe_ver = re.sub(r"[^\w.-]", "_", ps.version) + (".noestoppel" if core else "")
        cache = cache_dir / f"{doc.doc_id}.{safe_model}.{safe_ver}.json"
        if cache.exists():
            if getattr(llm, "cost_log", None) is not None:
                llm.cost_log.note_cached()       # served from disk: no API call, $0
            out = schema.model_validate_json(cache.read_text(encoding="utf-8"))
            return out, completeness.check(out, text)
    user = render_for_llm(doc.doc_id, doc.label, text, MAX_DOC_CHARS)
    tag = dict(doc_id=doc.doc_id, doc_code=doc.code, prompt_version=ps.version)
    with call_context(stage="extract", **tag):
        out = llm.extract(system, user, schema)
    issues = completeness.check(out, text)
    if issues and completeness_retry:
        # One targeted retry: tell the model exactly what it appears to have missed.
        log.info("Completeness retry for %s: %s", doc.doc_id, issues)
        feedback = ("\n\nA previous pass over this document missed content:\n- "
                    + "\n- ".join(issues) + "\nRe-read the document and record it.")
        with call_context(stage="completeness_retry", **tag):
            retry = llm.extract(system, user + feedback, schema)
        retry_issues = completeness.check(retry, text)
        if len(retry_issues) <= len(issues):
            out, issues = retry, retry_issues
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(out.model_dump_json(indent=1), encoding="utf-8")
    return out, issues


def flatten(doc: FileWrapperDoc, ex: BaseModel, next_id: Callable[[], str]) -> list[Finding]:
    """Turn a per-document extraction into flat, individually checkable findings."""
    base = dict(doc_id=doc.doc_id, doc_label=doc.label,
                date=doc.official_date.isoformat() if doc.official_date else None)
    out: list[Finding] = []

    def add(kind, statement, citation, *, required=False, **meta):
        # Summaries are kept even when the model gave no citation, so they still show
        # in the UI; they are graded "uncited" instead of being silently dropped.
        if citation is None and required:
            citation = Citation(doc_id=doc.doc_id, page=0, quote="")
        if citation is not None:
            out.append(Finding(id=next_id(), kind=kind, statement=statement,
                               citation=citation, meta=meta, **base))

    if isinstance(ex, OfficeActionExtraction):
        add("action", f"{ex.action_type}: {ex.summary}", ex.summary_citation, required=True,
            action_type=ex.action_type)
        for r in ex.rejections:
            refs = f" over {', '.join(r.references)}" if r.references else ""
            add("rejection", f"Claims {r.claims} rejected under {r.basis}{refs}: {r.summary}",
                r.citation, claims=r.claims, basis=r.basis, references=r.references)
        if ex.allowable_claims:
            add("allowable", f"Claims {ex.allowable_claims} indicated allowable",
                ex.allowable_citation, claims=ex.allowable_claims)
    elif isinstance(ex, ResponseCoreExtraction):
        add("response", ex.summary, ex.summary_citation, required=True)
        for a in ex.amendments:
            add("amendment", f"Claim {a.claim}: {a.change}", a.citation, claims=[a.claim])
        for e in getattr(ex, "estoppel", []):
            add("estoppel", f"[{e.kind}] {e.limitation}: {e.statement}", e.citation,
                claims=e.claims, risk=e.risk.value, estoppel_type=e.kind, rationale=e.rationale,
                limitation=e.limitation, distinguished_art=e.distinguished_art)
    elif isinstance(ex, ClaimsExtraction):
        for c in ex.claims:
            add("claim", f"Claim {c.number} ({c.status})", c.citation,
                claims=[c.number], status=c.status, independent=c.independent)
    elif isinstance(ex, AllowanceExtraction):
        add("allowance", ex.summary, ex.summary_citation, required=True)
        for r in ex.reasons:
            add("allowance_reason", r.point, r.citation, claims=r.claims)
        if ex.examiners_amendment:
            add("examiners_amendment", ex.examiners_amendment, ex.examiners_amendment_citation)
    elif isinstance(ex, InterviewExtraction):
        add("interview", ex.summary, ex.summary_citation, required=True,
            agreement=ex.agreement_reached)
    return out


def analyze(app: ApplicationInfo, docs: list[FileWrapperDoc], texts: dict[str, DocText],
            llm: LLM, settings: Settings, *, progress: Progress = _noop, judge: LLM | None = None,
            workers: int = 4, extract_cache: Path | None = None,
            prompt_set: prompts.PromptSet | None = None,
            completeness_retry: bool = True, estoppel: bool | None = None) -> Analysis:
    ps = prompt_set or prompts.CURRENT
    estoppel = settings.estoppel if estoppel is None else estoppel
    analysis = Analysis(app=app, documents=docs, model=llm.model, texts=texts,
                        prompt_version=ps.version, features={"estoppel": estoppel},
                        analyzed_doc_ids=[d.doc_id for d in docs if d.doc_id in texts])
    todo = [d for d in docs if d.doc_id in texts and d.category in ROUTES]

    # --- OCR stats
    pages = [p for t in texts.values() for p in t.pages]
    ocr_p = [p for p in pages if p.method == "ocr"]
    analysis.ocr = OCRStats(
        pages=len(pages), ocr_pages=len(ocr_p), text_layer_pages=len(pages) - len(ocr_p),
        low_quality_pages=sum(p.is_low_quality(settings.low_conf_threshold) for p in pages),
        mean_ocr_confidence=round(sum(p.confidence or 0 for p in ocr_p) / len(ocr_p), 1)
        if ocr_p else None,
    )

    # --- map: one extraction per document, in parallel
    results: dict[str, BaseModel] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_extract_one, d, texts[d.doc_id], llm, extract_cache, ps,
                          completeness_retry, estoppel): d for d in todo}
        for i, fut in enumerate(as_completed(futs)):
            d = futs[fut]
            progress(f"Analyzed {d.label}", 0.5 + 0.35 * (i + 1) / max(len(todo), 1))
            try:
                results[d.doc_id], issues = fut.result()
                if issues:
                    analysis.completeness[d.doc_id] = issues
            except Exception as e:  # keep going; surface per-doc failures in the UI
                log.exception("Extraction failed for %s", d.doc_id)
                analysis.errors[d.doc_id] = f"{type(e).__name__}: {e}"

    # --- flatten in chronological order so finding ids read naturally
    counter = iter(range(1, 1_000_000))
    next_id = lambda: f"F{next(counter)}"  # noqa: E731
    for d in todo:
        if d.doc_id in results:
            analysis.extractions[d.doc_id] = results[d.doc_id].model_dump(mode="json")
            analysis.findings += flatten(d, results[d.doc_id], next_id)

    # --- claim evolution (deterministic)
    snaps = []
    for d in todo:
        r = results.get(d.doc_id)
        if isinstance(r, ClaimsExtraction):
            snaps += [ClaimSnapshot(number=c.number, doc_id=d.doc_id, date=d.official_date,
                                    status=c.status, text=c.text)
                      for c in r.claims if c.independent or c.status == "canceled"]
    analysis.claim_histories = build_histories(snaps)

    # --- reduce: synthesis over findings only (not raw text)
    if analysis.findings:
        progress("Synthesizing overview", 0.88)
        lines = [f"{f.id} | {f.date} | {f.kind} | {f.statement}" for f in analysis.findings]
        try:
            with call_context(stage="synthesis", prompt_version=ps.version):
                system = ps.synthesis if estoppel else \
                    ps.synthesis.replace(", validity and estoppel", " and validity")
                analysis.synthesis = llm.extract(system, "\n".join(lines),
                                                 Synthesis)
        except Exception as e:
            analysis.errors["synthesis"] = f"{type(e).__name__}: {e}"

    # --- evaluation
    progress("Verifying citations", 0.93)
    checks = {f.id: check_citation(f, texts, settings.low_conf_threshold)
              for f in analysis.findings}
    if judge is not None:
        progress("Judging support", 0.96)
        try:
            with call_context(stage="judge"):
                judge_support(analysis.findings, checks, texts, judge)
        except Exception as e:
            analysis.errors["judge"] = f"{type(e).__name__}: {e}"
    analysis.grounding = build_report(analysis.findings, checks, analysis.synthesis)

    usage = llm.usage.as_dict()
    if judge is not None and judge is not llm:
        usage["judge"] = judge.usage.as_dict()
    analysis.usage = usage
    if getattr(llm, "cost_log", None) is not None:
        analysis.cost = llm.cost_log.summary()
    progress("Done", 1.0)
    return analysis


def attach_family(analysis: Analysis, settings: Settings, progress: Progress = _noop,
                  client: OPSClient | None = None,
                  retry_missing: bool = False) -> FamilyReport:
    """Look up the patent's family (US + foreign) on EPO OPS and flag art not of record in the US.

    "Of record" = on the face of the US patent (examiner- or applicant-cited) or named in a
    rejection / distinguished-art finding extracted from this file history."""
    number = analysis.app.patent_number
    if not number:
        raise ValueError("Family lookup needs a granted US patent number.")
    own = client is None
    client = client or OPSClient(settings.epo_ops_key, settings.epo_ops_secret,
                                 settings.cache_dir)
    try:
        progress(f"EPO OPS: family of US {number}", 0.0)
        analysis.family = client.family(number, us_refs_from_findings(analysis.findings),
                                        retry_missing=retry_missing)
        progress("Family done", 1.0)
        return analysis.family
    finally:
        if own:
            client.close()


def save(analysis: Analysis, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(analysis.model_dump_json(indent=1), encoding="utf-8")
    return path


def load(path: Path) -> Analysis:
    return Analysis.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
