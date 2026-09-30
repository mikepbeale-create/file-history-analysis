"""Priority / effective-filing-date support analysis across a continuity chain.

For a patent that claims priority through continuations, CIPs and provisionals, each claim
is entitled to an earlier application's filing date only if that application - and every
application between it and the patent - provides written-description support for every
element of the claim (35 U.S.C. 112(a) via 119(e)/120). A CIP adds new matter, so claims
that rely on it get only the CIP's date.

Pipeline
--------
1. Chain: USPTO ODP continuity data -> a graph of parent/child applications.
2. Disclosure: for each application, the specification, claims and abstract *as filed*
   (earliest-dated SPEC/CLM/ABST documents in its file wrapper), OCR'd.
3. Claims: the patent's claims are parsed into verbatim text; the user picks which to
   analyze (independent claims by default). Each claim is split into its elements.
4. Support: one LLM pass per application (disclosure in a cached system prompt, elements
   in batches) grades each element explicit / implicit / none with verbatim quotes.
   Quotes are verified against the OCR text exactly like the main analysis; a "supported"
   verdict without any verified quote is downgraded to yellow.
5. Entitlement (deterministic): for each claim, the earliest application such that all of
   its elements (including those inherited from parent claims) are supported there and in
   every application on the path back from the patent.
6. CIP new matter (deterministic): each CIP's specification is diffed against its
   parent's, so support that exists only in added text is identified.

This is an analysis aid for a human, not a legal conclusion.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from collections.abc import Callable
from datetime import date, datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process

from .costs import RunCost, call_context
from .grounding import Finding, check_citation
from .models import Citation
from .ocr import DocText, extract_document, render_for_llm

log = logging.getLogger(__name__)

Progress = Callable[[str, float], None]
Level = Literal["explicit", "implicit", "none"]

PROVISIONAL_SERIES = ("60", "61", "62", "63")
RELATION_LABEL = {
    "CON": "continuation", "CIP": "continuation-in-part", "DIV": "divisional",
    "PRO": "claims benefit of provisional", "NST": "national stage", "REI": "reissue", "RCE": "RCE",
}


def _noop(msg: str, frac: float) -> None:
    log.info("[%3.0f%%] %s", frac * 100, msg)


def _parse_date(v) -> date | None:
    if not v:
        return None
    try:
        return date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


def fmt_app(n: str) -> str:
    return f"{n[:2]}/{n[2:5]},{n[5:]}" if len(n) == 8 and n.isdigit() else n


# ------------------------------------------------------------------ 1. chain

class ChainApp(BaseModel):
    app_no: str
    filing_date: date | None = None
    provisional: bool = False
    patent_number: str | None = None
    status: str | None = None

    @property
    def label(self) -> str:
        kind = "prov." if self.provisional else "appl."
        pn = f" (US {int(self.patent_number):,})" if (self.patent_number or "").isdigit() \
            else ""
        return f"{kind} {fmt_app(self.app_no)}{pn}"


class ChainEdge(BaseModel):
    child: str
    parent: str
    code: str                     # CON / CIP / DIV / PRO / ...
    description: str = ""


class Chain(BaseModel):
    subject: str
    apps: dict[str, ChainApp]
    edges: list[ChainEdge]

    def parents_of(self, app_no: str) -> list[ChainEdge]:
        return [e for e in self.edges if e.child == app_no]

    def path_to(self, ancestor: str) -> list[str] | None:
        """Applications from the subject back to `ancestor` (inclusive), following
        parent links. None if `ancestor` is not reachable."""
        prev: dict[str, str | None] = {self.subject: None}
        q = deque([self.subject])
        while q:
            cur = q.popleft()
            if cur == ancestor:
                path = [cur]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return list(reversed(path))
            for e in self.parents_of(cur):
                if e.parent not in prev:
                    prev[e.parent] = cur
                    q.append(e.parent)
        return None

    def edge(self, child: str, parent: str) -> ChainEdge | None:
        return next((e for e in self.edges if e.child == child and e.parent == parent), None)

    def ordered(self) -> list[ChainApp]:
        """Oldest first; undated last."""
        return sorted(self.apps.values(), key=lambda a: (a.filing_date or date.max, a.app_no))


def _walk_records(node) -> list[dict]:
    out: list[dict] = []
    if isinstance(node, dict):
        if "parentApplicationNumberText" in node or "childApplicationNumberText" in node:
            out.append(node)
        for v in node.values():
            out += _walk_records(v)
    elif isinstance(node, list):
        for v in node:
            out += _walk_records(v)
    return out


def _first(rec: dict, *keys):
    for k in keys:
        if rec.get(k):
            return rec[k]
    return None


def parse_chain(continuity: dict, subject: str, subject_filing: date | None,
                subject_patent: str | None = None) -> Chain:
    """Build the ancestor graph from an ODP continuity response.

    ODP lists every ancestor relationship as a (child, parent, type) record, so the whole
    chain comes back in one call. Only parent-side records are used: children of the
    patent are irrelevant to its own priority date."""
    apps = {subject: ChainApp(app_no=subject, filing_date=subject_filing,
                              patent_number=subject_patent)}
    edges: list[ChainEdge] = []
    for rec in _walk_records(continuity):
        parent = _first(rec, "parentApplicationNumberText")
        child = _first(rec, "childApplicationNumberText") or subject
        if not parent or parent == subject:
            continue
        code = (_first(rec, "claimParentageTypeCode") or "").upper()
        desc = _first(rec, "claimParentageTypeCodeDescriptionText",
                      "claimParentageTypeCodeDescription") or RELATION_LABEL.get(code, code)
        pdate = _parse_date(_first(rec, "parentApplicationFilingDate", "filingDate"))
        pa = apps.setdefault(parent, ChainApp(app_no=parent))
        pa.filing_date = pa.filing_date or pdate
        pa.provisional = pa.provisional or code == "PRO" or parent[:2] in PROVISIONAL_SERIES
        pa.patent_number = pa.patent_number or _first(rec, "parentPatentNumber")
        pa.status = pa.status or _first(rec, "parentApplicationStatusDescriptionText",
                                        "applicationStatusDescriptionText")
        apps.setdefault(child, ChainApp(app_no=child))
        if not any(e.child == child and e.parent == parent for e in edges):
            edges.append(ChainEdge(child=child, parent=parent, code=code, description=desc))
    chain = Chain(subject=subject, apps=apps, edges=edges)
    # Drop anything not actually an ancestor of the subject (e.g. sibling branches).
    keep = {a for a in apps if chain.path_to(a) is not None}
    chain.apps = {k: v for k, v in apps.items() if k in keep}
    chain.edges = [e for e in edges if e.child in keep and e.parent in keep]
    return chain


# ------------------------------------------------------------------ 2. disclosures

DISCLOSURE_CODES = {"SPEC": "specification", "CLM": "claims as filed", "ABST": "abstract"}


class Disclosure(BaseModel):
    app_no: str
    doc_ids: list[str] = Field(default_factory=list)
    parts: dict[str, str] = Field(default_factory=dict)   # doc_id -> "specification" ...
    texts: dict[str, DocText] = Field(default_factory=dict)
    available: bool = False
    note: str = ""

    @property
    def pages(self) -> int:
        return sum(len(t.pages) for t in self.texts.values())


def select_as_filed(docs) -> list:
    """Earliest-dated SPEC / CLM / ABST documents = the disclosure as filed. Later
    documents with the same codes are amendments, which cannot add support."""
    chosen = []
    for code in DISCLOSURE_CODES:
        same = [d for d in docs if d.code == code and d.pdf_url]
        if not same:
            continue
        first = min((d.official_date or date.max) for d in same)
        chosen += [d for d in same if (d.official_date or date.max) == first]
    return chosen


def fetch_disclosure(client, app_no: str, settings, progress: Progress = _noop) -> Disclosure:
    out = Disclosure(app_no=app_no)
    try:
        docs = client.list_documents(app_no)
    except Exception as e:  # e.g. pre-2001 applications are not in ODP
        out.note = f"file wrapper unavailable ({type(e).__name__}: {e})"
        return out
    chosen = select_as_filed(docs)
    if not any(d.code == "SPEC" for d in chosen):
        out.note = "no specification (SPEC) document found in the file wrapper"
    for d in chosen:
        try:
            pdf = client.download_pdf(app_no, d)
            out.texts[d.doc_id] = extract_document(pdf, d.doc_id, settings)
            out.doc_ids.append(d.doc_id)
            out.parts[d.doc_id] = DISCLOSURE_CODES[d.code]
        except Exception as e:
            out.note += f"; {d.code} {d.doc_id} failed: {e}"
    out.available = any(out.parts.get(i) == "specification" for i in out.doc_ids)
    return out


def render_disclosure(disc: Disclosure, label: str) -> str:
    return "\n\n".join(
        render_for_llm(did, f"{label} - {disc.parts.get(did, '')}", disc.texts[did], 600_000)
        for did in disc.doc_ids)


# ------------------------------------------------------------------ 3. claims + elements

class ParsedClaim(BaseModel):
    number: int
    depends_on: int | None = Field(
        default=None, description="Number of the claim it depends from; null if independent.")
    text: str = Field(description="The full claim text, verbatim, without the number.")


class ClaimSet(BaseModel):
    claims: list[ParsedClaim]


class Element(BaseModel):
    id: str = Field(description="Label such as '1[pre]', '1[a]', '1[b]', '5[a]'.")
    claim: int
    text: str = Field(description="Verbatim segment of the claim text.")


class ClaimElements(BaseModel):
    elements: list[Element]


CLAIMS_PROMPT = """\
You are given the claims of a US patent (OCR text of a claim listing, or pasted text).
Return every claim that is NOT canceled, with its number, the claim it depends from (null
for independent claims), and its full text VERBATIM - no paraphrasing, no abbreviation.
In an amended listing, drop text shown as deleted ([[double brackets]] or struck through)
and keep added text. Ignore status identifiers such as "(Currently Amended)".
"""

ELEMENTS_PROMPT = """\
Split each patent claim below into its claim elements, the way a claim chart would:
the preamble as '<n>[pre]', then each limitation as '<n>[a]', '<n>[b]', ... . Split at the
natural boundaries (semicolons, 'wherein' / 'such that' clauses, each recited step or
component). For a dependent claim, only its ADDED limitation(s): do not repeat the parent.
Every element's text must be a VERBATIM, contiguous segment of the claim - never paraphrase.
"""


def element_text_ok(el: Element, claim_text: str) -> bool:
    """Guard: an element must be a verbatim piece of its claim (OCR-tolerant)."""
    return fuzz.partial_ratio(el.text.lower(), claim_text.lower()) >= 90


# ------------------------------------------------------------------ 4. support

class ElementSupport(BaseModel):
    element_id: str
    level: Level = Field(description=(
        "explicit = the disclosure describes this limitation in words that reasonably "
        "convey possession of it; implicit = arguably supported but it takes explanation "
        "(inherent, spread across passages, species vs genus, a figure only); none = no "
        "support found."))
    explanation: str = Field(description="One or two sentences, specific to this document.")
    citations: list[Citation] = Field(default_factory=list, description=(
        "Up to 3 verbatim quotes that provide the support. Empty when level is none."))


class SupportExtraction(BaseModel):
    items: list[ElementSupport]


SUPPORT_PROMPT = """\
You are a US patent attorney assessing WRITTEN-DESCRIPTION SUPPORT (35 U.S.C. 112(a)) for
claim elements in ONE earlier application, to decide whether a later patent's claims are
entitled to that application's filing date.

Standard: the application as filed must reasonably convey to a skilled person that the
inventor possessed the claimed subject matter. Word-for-word identity is not required, but
obviousness is NOT support, and a genus is not supported by a single unrelated species.
Judge only from the documents below (specification, claims as filed, abstract). Do not use
outside knowledge or other family members.

For each element return:
- level: explicit / implicit / none (be conservative; if you are unsure, use implicit),
- explanation: why, specific to this application,
- citations: up to 3 VERBATIM quotes (8-60 words) copied exactly from a single <page>,
  with the document id and page number. The text is OCR: copy errors as they appear.
Return one item for EVERY element id you are given, in the same order.

The application's disclosure:
"""


class SupportCell(BaseModel):
    element_id: str
    app_no: str
    level: Level
    explanation: str = ""
    citations: list[Citation] = Field(default_factory=list)
    verified: list[bool] = Field(default_factory=list)
    new_matter: bool = False        # support found only in text a CIP added over its parent

    @property
    def color(self) -> Literal["green", "yellow", "red"]:
        if self.level == "none":
            return "red"
        if self.level == "explicit" and any(self.verified):
            return "green"
        return "yellow"   # implicit, or 'explicit' without any verified quote


def _verify(cell: SupportCell, texts: dict[str, DocText]) -> None:
    cell.verified = []
    for i, c in enumerate(cell.citations):
        f = Finding(id=f"S{i}", kind="support", doc_id=c.doc_id, doc_label=c.doc_id,
                    date=None, statement=cell.element_id, citation=c)
        cell.verified.append(check_citation(f, texts).status in ("verified", "wrong_page"))


def assess_support(llm, disc: Disclosure, label: str, elements: list[Element],
                   batch: int = 12) -> list[SupportCell]:
    system = SUPPORT_PROMPT + render_disclosure(disc, label)
    cells: list[SupportCell] = []
    for i in range(0, len(elements), batch):
        chunk = elements[i:i + batch]
        user = "Claim elements to assess:\n" + "\n".join(f"{e.id}: {e.text}" for e in chunk)
        with call_context(stage="priority_support", doc_id=disc.app_no, doc_code="SPEC"):
            out = llm.extract(system, user, SupportExtraction)
        got = {s.element_id: s for s in out.items}
        for e in chunk:
            s = got.get(e.id)
            cell = SupportCell(element_id=e.id, app_no=disc.app_no,
                               level=s.level if s else "none",
                               explanation=s.explanation if s else "No assessment returned.",
                               citations=(s.citations[:3] if s else []))
            _verify(cell, disc.texts)
            cells.append(cell)
    return cells


# ------------------------------------------------------------------ 6. CIP new matter

def sentences(dt: DocText) -> list[tuple[int, str]]:
    """(page, sentence) pairs. Comparing sentences rather than paragraphs keeps the diff
    aligned even when OCR loses paragraph breaks or a CIP inserts text mid-paragraph."""
    out: list[tuple[int, str]] = []
    for p in dt.pages:
        text = re.sub(r"\[\s*0?\d{3,4}\s*\]", " ", re.sub(r"\s+", " ", p.text))
        for sent in re.split(r"(?<=[.;:])\s+(?=[A-Z(])", text):
            if len(sent.strip()) > 30:
                out.append((p.page, sent.strip()))
    return out


class NewMatter(BaseModel):
    child: str
    parent: str
    child_sentences: int
    new_passages: list[tuple[int, str]]      # (page, text) in the child's specification
    new_sentences: int = 0

    @property
    def new_fraction(self) -> float:
        return self.new_sentences / self.child_sentences if self.child_sentences else 0.0


def diff_specs(child: Disclosure, parent: Disclosure, threshold: int = 85) -> NewMatter | None:
    """Sentences of the child's specification with no close counterpart in the parent's.
    Consecutive new sentences on a page are merged into one passage."""
    def spec_sents(d: Disclosure):
        return [ss for did in d.doc_ids if d.parts.get(did) == "specification"
                for ss in sentences(d.texts[did])]
    cs, ps = spec_sents(child), spec_sents(parent)
    if not cs or not ps:
        return None
    parent_texts = [t.lower() for _, t in ps]
    passages: list[tuple[int, str]] = []
    n_new, prev_new = 0, False
    for page, t in cs:
        best = process.extractOne(t.lower(), parent_texts, scorer=fuzz.ratio)
        is_new = best is None or best[1] < threshold
        if is_new:
            n_new += 1
            if prev_new and passages and passages[-1][0] == page:
                passages[-1] = (page, passages[-1][1] + " " + t)
            else:
                passages.append((page, t))
        prev_new = is_new
    return NewMatter(child=child.app_no, parent=parent.app_no, child_sentences=len(cs),
                     new_passages=passages, new_sentences=n_new)


def quote_in_new_matter(quote: str, nm: NewMatter) -> bool:
    return any(fuzz.partial_ratio(quote.lower(), t.lower()) >= 90 for _, t in nm.new_passages)


# ------------------------------------------------------------------ 5. entitlement

class ClaimPriority(BaseModel):
    claim: int
    element_ids: list[str]
    strict_app: str | None            # earliest app where every element is green
    strict_date: date | None
    lenient_app: str | None           # earliest where every element is green or yellow
    lenient_date: date | None
    breaks: list[str] = Field(default_factory=list)   # "why not earlier" notes


def claim_element_ids(claim: int, claims: dict[int, ParsedClaim],
                      elements: list[Element]) -> list[str]:
    """A dependent claim includes every element of the claims it depends from."""
    ids: list[str] = []
    seen: set[int] = set()
    cur: int | None = claim
    chain: list[int] = []
    while cur is not None and cur not in seen and cur in claims:
        seen.add(cur)
        chain.append(cur)
        cur = claims[cur].depends_on
    for c in reversed(chain):
        ids += [e.id for e in elements if e.claim == c]
    return ids


def entitlement(chain: Chain, grid: dict[tuple[str, str], SupportCell], ids: list[str],
                claim: int, analyzed: set[str]) -> ClaimPriority:
    def ok(app: str, lenient: bool) -> bool:
        cells = [grid.get((app, i)) for i in ids]
        if not cells or any(c is None for c in cells):
            return False
        return all(c.color != "red" if lenient else c.color == "green" for c in cells)

    def best(lenient: bool) -> str | None:
        cands = []
        for a in chain.ordered():
            if a.app_no not in analyzed:
                continue
            path = chain.path_to(a.app_no)
            if path and all(p in analyzed and ok(p, lenient) for p in path):
                cands.append(a)
        return cands[0].app_no if cands else None

    s, lw = best(False), best(True)
    breaks = []
    sdate = chain.apps[s].filing_date if s else None
    for a in chain.ordered():
        if a.app_no not in analyzed or (sdate and a.filing_date and a.filing_date >= sdate):
            continue        # only explain the applications older than the date obtained
        missing = [i for i in ids if (c := grid.get((a.app_no, i))) is None or c.color == "red"]
        if missing:
            breaks.append(f"{a.label} ({a.filing_date or 'n.d.'}): no support for "
                          + ", ".join(missing))
    return ClaimPriority(
        claim=claim, element_ids=ids,
        strict_app=s, strict_date=chain.apps[s].filing_date if s else None,
        lenient_app=lw, lenient_date=chain.apps[lw].filing_date if lw else None,
        breaks=breaks)


# ------------------------------------------------------------------ report + driver

class PriorityReport(BaseModel):
    subject: str
    chain: Chain
    claims: list[ParsedClaim] = Field(default_factory=list)
    claims_source: str = ""
    selected: list[int] = Field(default_factory=list)
    elements: list[Element] = Field(default_factory=list)
    element_warnings: list[str] = Field(default_factory=list)
    disclosures: dict[str, Disclosure] = Field(default_factory=dict)
    cells: list[SupportCell] = Field(default_factory=list)
    results: list[ClaimPriority] = Field(default_factory=list)
    new_matter: list[NewMatter] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)
    usage: dict = Field(default_factory=dict)
    # One RunCost per button press (step 1, step 2, re-runs); see total_usd.
    costs: list[RunCost] = Field(default_factory=list)
    created: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def stage(self) -> str:
        return "done" if self.results else ("claims" if self.claims else "chain")

    @property
    def total_usd(self) -> float:
        return round(sum(c.total_usd for c in self.costs), 4)

    def grid(self) -> dict[tuple[str, str], SupportCell]:
        return {(c.app_no, c.element_id): c for c in self.cells}

    def claim(self, n: int) -> ParsedClaim | None:
        return next((c for c in self.claims if c.number == n), None)


def parse_claims(llm, text: str) -> list[ParsedClaim]:
    with call_context(stage="priority_claims"):
        out = llm.extract(CLAIMS_PROMPT, text, ClaimSet)
    seen, claims = set(), []
    for c in sorted(out.claims, key=lambda c: c.number):
        if c.number not in seen and c.text.strip():
            seen.add(c.number)
            claims.append(c)
    return claims


def prepare(client, analysis, llm, *, claims_text: str | None = None,
            progress: Progress = _noop) -> PriorityReport:
    """Step 1 (cheap): continuity chain + the patent's claims, for the user to pick from."""
    app = analysis.app
    progress("Reading continuity data", 0.1)
    chain = parse_chain(client.continuity(app.application_number), app.application_number,
                        app.filing_date, app.patent_number)
    report = PriorityReport(subject=app.application_number, chain=chain)
    if claims_text:
        source, text = "pasted claims", claims_text
    else:
        clm = [d for d in analysis.documents if d.code == "CLM" and d.doc_id in analysis.texts]
        if not clm:
            report.errors["claims"] = ("No claim listing in this analysis. Paste the issued "
                                       "claims instead.")
            return report
        last = max(clm, key=lambda d: (d.official_date or date.min, d.doc_id))
        source = f"latest claim listing ({last.label})"
        text = render_for_llm(last.doc_id, last.label, analysis.texts[last.doc_id])
    progress("Parsing claims", 0.6)
    report.claims = parse_claims(llm, text)
    report.claims_source = source
    report.selected = [c.number for c in report.claims if c.depends_on is None]
    _add_cost(report, llm)
    return report


def _add_cost(report: PriorityReport, llm) -> None:
    if getattr(llm, "cost_log", None) is not None:
        report.costs.append(llm.cost_log.summary())


def run(client, report: PriorityReport, llm, settings, selected: list[int],
        progress: Progress = _noop) -> PriorityReport:
    """Step 2: elements -> disclosures -> support grid -> entitlement -> CIP diff."""
    claims = {c.number: c for c in report.claims}
    # Include every claim a selected dependent relies on, so inherited elements exist.
    needed: set[int] = set()
    for n in selected:
        cur = n
        while cur is not None and cur in claims and cur not in needed:
            needed.add(cur)
            cur = claims[cur].depends_on
    report.selected = sorted(selected)

    progress("Splitting claims into elements", 0.02)
    todo = [claims[n] for n in sorted(needed)]
    body = "\n\n".join(f"Claim {c.number}"
                       + (f" (depends on claim {c.depends_on})" if c.depends_on else "")
                       + f": {c.text}" for c in todo)
    with call_context(stage="priority_elements"):
        els = llm.extract(ELEMENTS_PROMPT, body, ClaimElements).elements
    report.elements = [e for e in els if e.claim in needed]
    report.element_warnings = [f"{e.id} is not a verbatim segment of claim {e.claim}"
                               for e in report.elements
                               if not element_text_ok(e, claims[e.claim].text)]

    apps = report.chain.ordered()
    report.cells = []
    for i, a in enumerate(apps):
        frac = 0.05 + 0.85 * i / max(len(apps), 1)
        progress(f"Disclosure of {a.label}", frac)
        disc = report.disclosures.get(a.app_no)
        if disc is None or not disc.available:
            disc = fetch_disclosure(client, a.app_no, settings)
            report.disclosures[a.app_no] = disc
        if not disc.available:
            report.errors[a.app_no] = disc.note or "disclosure unavailable"
            continue
        progress(f"Assessing support in {a.label}", frac + 0.4 / max(len(apps), 1))
        try:
            report.cells += assess_support(llm, disc, a.label, report.elements)
        except Exception as e:
            log.exception("support failed for %s", a.app_no)
            report.errors[a.app_no] = f"{type(e).__name__}: {e}"

    progress("Comparing CIP specifications", 0.92)
    report.new_matter = []
    for e in report.chain.edges:
        if e.code == "CIP" and e.child in report.disclosures and e.parent in report.disclosures:
            nm = diff_specs(report.disclosures[e.child], report.disclosures[e.parent])
            if nm:
                report.new_matter.append(nm)
    for c in report.cells:
        nms = [nm for nm in report.new_matter if nm.child == c.app_no]
        c.new_matter = bool(nms) and c.level != "none" and bool(c.citations) and all(
            any(quote_in_new_matter(q.quote, nm) for nm in nms) for q in c.citations)

    analyzed = {a for a, d in report.disclosures.items() if d.available
                and a not in report.errors}
    grid = report.grid()
    report.results = [entitlement(report.chain, grid,
                                  claim_element_ids(n, claims, report.elements), n, analyzed)
                      for n in report.selected]
    report.usage = llm.usage.as_dict()
    _add_cost(report, llm)
    progress("Done", 1.0)
    return report
