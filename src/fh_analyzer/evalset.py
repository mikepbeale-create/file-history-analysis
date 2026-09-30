"""Expert-labeled evaluation set ("gold labels") and the scorer that uses it.

Workflow
--------
1. Run the analyzer on a patent (produces data/cache/<app>/analysis.json).
2. Open `label_app.py`. Every model finding is shown as a pre-filled label; the expert
   marks it correct / fixes it / marks it wrong, and adds anything the model missed.
   Labels are saved to gold/<app>.json after every click.
3. `fh-score` compares any analysis run (another model, another prompt version) against
   the gold file and reports precision / recall / F1 per finding type, field accuracy,
   and agreement on estoppel risk.

Only documents the expert marked "reviewed" are scored. On an unreviewed document we
cannot tell a model miss from a labeling gap, so including it would distort recall.

Caveat, stated in the README too: pre-filling labels from model output speeds labeling
up a lot but can anchor the labeler toward the model. The "add missed item" step and
per-document review are there to counter that; for a stricter set, label some
documents blind first.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from .grounding import Finding

# ------------------------------------------------------------------ what gets labeled

# kind -> editable fields (name, widget type). Summary-type findings ("action",
# "response", ...) are narrative and not scored, so they are not labeled.
KIND_FIELDS: dict[str, list[tuple[str, str]]] = {
    "rejection": [("claims", "claims"), ("basis", "basis"), ("references", "list"),
                  ("summary", "text")],
    "allowable": [("claims", "claims")],
    "amendment": [("claims", "claims"), ("change", "text")],
    "estoppel": [("claims", "claims"), ("limitation", "str"), ("estoppel_type", "etype"),
                 ("risk", "risk"), ("distinguished_art", "list"), ("summary", "text")],
    "allowance_reason": [("claims", "claims"), ("summary", "text")],
    "examiners_amendment": [("summary", "text")],
    "claim": [("claims", "claims"), ("status", "cstatus"), ("text", "text")],
}
KIND_LABELS = {
    "rejection": "Rejection / objection", "allowable": "Allowable claims",
    "amendment": "Claim amendment", "estoppel": "Estoppel / disclaimer",
    "allowance_reason": "Reason for allowance", "examiners_amendment": "Examiner's amendment",
    "claim": "Independent claim (text)",
}
CHOICES = {
    "basis": ["101", "102", "103", "112(a)", "112(b)", "112(d)", "112(f)",
              "double patenting", "objection", "other"],
    "etype": ["argument-based disclaimer", "amendment-based estoppel",
              "definition/lexicography"],
    "risk": ["high", "medium", "low"],
    "cstatus": ["original", "currently amended", "previously presented", "canceled",
                "withdrawn", "new", "unknown"],
}
# Why a model item was wrong or needed fixing - the basis for error analysis.
ERROR_TAGS = [
    "not in document (hallucinated)", "boilerplate / form text", "duplicate",
    "wrong claims", "wrong statutory basis", "wrong / missing references",
    "wrong risk level", "overstated", "wrong claim text (OCR/markup)", "wrong page / quote",
    "incomplete", "other",
]

Verdict = Literal["correct", "corrected", "incorrect"]


# ------------------------------------------------------------------ claim-list helpers

def parse_claims(s: str) -> list[int]:
    """'1-3, 5, 7–8' -> [1, 2, 3, 5, 7, 8]"""
    out: set[int] = set()
    for part in re.split(r"[,\s;]+", s.replace("–", "-").replace("—", "-")):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a <= b and b - a < 500:
                out.update(range(a, b + 1))
        elif part.isdigit():
            out.add(int(part))
    return sorted(out)


def fmt_claims(nums: list[int]) -> str:
    """[1, 2, 3, 5] -> '1-3, 5'"""
    nums = sorted(set(nums))
    runs, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        runs.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(runs)


# ------------------------------------------------------------------ gold schema

class GoldItem(BaseModel):
    id: str                                   # G1, G2 ...
    origin: Literal["model", "added"]
    model_finding_id: str | None = None       # F-id in the analysis it was created from
    kind: str
    doc_id: str
    verdict: Verdict | None = None            # None = not reviewed yet
    fields: dict = Field(default_factory=dict)   # the TRUE values (after any fix)
    model_values: dict = Field(default_factory=dict)  # what the model said (for diffs)
    page: int | None = None
    quote: str = ""
    error_tags: list[str] = Field(default_factory=list)
    note: str = ""

    @property
    def is_gold(self) -> bool:
        """True items: model items confirmed/fixed, plus items the expert added."""
        return self.origin == "added" or self.verdict in ("correct", "corrected")


class GoldDoc(BaseModel):
    reviewed: bool = False
    opened_at: str | None = None
    reviewed_at: str | None = None
    note: str = ""


class GoldSet(BaseModel):
    application_number: str
    patent_number: str | None = None
    title: str | None = None
    labeler: str = ""
    source_model: str = ""
    source_prompt_version: str = ""
    source_analysis_created: str = ""
    created: str = Field(default_factory=lambda: _now())
    updated: str = Field(default_factory=lambda: _now())
    docs: dict[str, GoldDoc] = Field(default_factory=dict)
    items: list[GoldItem] = Field(default_factory=list)

    def next_id(self) -> str:
        n = max((int(i.id[1:]) for i in self.items if i.id[1:].isdigit()), default=0)
        return f"G{n + 1}"

    def for_doc(self, doc_id: str) -> list[GoldItem]:
        return [i for i in self.items if i.doc_id == doc_id]

    def progress(self) -> dict:
        model = [i for i in self.items if i.origin == "model"]
        return {
            "docs_reviewed": sum(d.reviewed for d in self.docs.values()),
            "docs_total": len(self.docs),
            "items_reviewed": sum(i.verdict is not None for i in model),
            "items_total": len(model),
            "added": sum(i.origin == "added" for i in self.items),
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ build from analysis

def _summary_part(statement: str) -> str:
    return statement.split(": ", 1)[1] if ": " in statement else statement


def fields_from_finding(f: Finding, extractions: dict | None = None) -> dict:
    m, claims = f.meta, list(f.meta.get("claims", []))
    if f.kind == "rejection":
        return {"claims": claims, "basis": m.get("basis", "other"),
                "references": list(m.get("references", [])),
                "summary": _summary_part(f.statement)}
    if f.kind == "allowable":
        return {"claims": claims}
    if f.kind == "amendment":
        return {"claims": claims, "change": _summary_part(f.statement)}
    if f.kind == "estoppel":
        return {"claims": claims, "limitation": m.get("limitation", ""),
                "estoppel_type": m.get("estoppel_type", CHOICES["etype"][0]),
                "risk": m.get("risk", "low"),
                "distinguished_art": list(m.get("distinguished_art", [])),
                "summary": _summary_part(f.statement)}
    if f.kind in ("allowance_reason", "examiners_amendment"):
        return {"claims": claims, "summary": f.statement} if f.kind == "allowance_reason" \
            else {"summary": f.statement}
    if f.kind == "claim":
        text = ""
        ex = (extractions or {}).get(f.doc_id, {})
        for c in ex.get("claims", []):
            if claims and c.get("number") == claims[0]:
                text = c.get("text", "")
                break
        return {"claims": claims, "status": m.get("status", "unknown"), "text": text}
    return {}


def gold_from_analysis(analysis) -> GoldSet:
    """Start a gold set pre-filled with the model's findings (all unreviewed)."""
    g = GoldSet(
        application_number=analysis.app.application_number,
        patent_number=analysis.app.patent_number, title=analysis.app.title,
        source_model=analysis.model, source_prompt_version=analysis.prompt_version,
        source_analysis_created=analysis.created,
    )
    for d in analysis.documents:
        if d.doc_id in analysis.extractions:
            g.docs[d.doc_id] = GoldDoc()
    for f in analysis.findings:
        if f.kind not in KIND_FIELDS:
            continue
        flds = fields_from_finding(f, analysis.extractions)
        g.items.append(GoldItem(
            id=g.next_id(), origin="model", model_finding_id=f.id, kind=f.kind,
            doc_id=f.doc_id, fields=dict(flds), model_values=dict(flds),
            page=f.citation.page or None, quote=f.citation.quote,
        ))
    return g


def gold_path(gold_dir: Path, app_no: str) -> Path:
    return Path(gold_dir) / f"{app_no}.json"


def load_gold(path: Path) -> GoldSet | None:
    p = Path(path)
    return GoldSet.model_validate_json(p.read_text(encoding="utf-8")) if p.exists() else None


def _replace(tmp: str, dest: Path, attempts: int = 8) -> None:
    """os.replace with retries. On Windows the rename fails with "Access is denied" (or
    PermissionError) while another program briefly has the target open: antivirus
    scanning the file just written, OneDrive / Dropbox sync, a search indexer, or an
    editor. Those locks last milliseconds to a second, so wait and try again."""
    delay = 0.05
    for i in range(attempts):
        try:
            os.replace(tmp, dest)
            return
        except PermissionError:
            if i == attempts - 1:
                break
            time.sleep(delay)
            delay = min(delay * 2, 1.0)
    # Still locked (~3 s): fall back to writing in place so the label is not lost.
    # Not atomic, but the temp copy is kept until this succeeds.
    data = Path(tmp).read_text(encoding="utf-8")
    dest.write_text(data, encoding="utf-8")
    try:
        os.remove(tmp)
    except OSError:
        pass


def save_gold(g: GoldSet, path: Path) -> None:
    """Atomic write, so a crash mid-save never corrupts hours of labeling."""
    g.updated = _now()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(g.model_dump_json(indent=1))
    _replace(tmp, p)


# ------------------------------------------------------------------ matching

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def _ref_key(r: str) -> str:
    """Reduce 'Smith et al. (US 2015/0139045)' to a comparable key: the patent/pub
    number digits if present, else the first word (the inventor name)."""
    digits = re.sub(r"\D", "", r)
    if len(digits) >= 6:
        return digits[-7:]
    w = re.findall(r"[A-Za-z]+", r)
    return w[0].lower() if w else _norm(r)


def match_score(kind: str, gold: dict, pred: dict, gold_quote: str = "",
                pred_quote: str = "") -> float:
    """0..1 similarity between a gold item and a predicted item of the same kind and
    document. 0 means 'cannot be the same item'."""
    cj = _jaccard(gold.get("claims", []), pred.get("claims", []))
    q = fuzz.partial_ratio(_norm(gold_quote), _norm(pred_quote)) / 100 \
        if gold_quote and pred_quote else 0.0
    if kind == "rejection":
        if gold.get("basis") != pred.get("basis"):
            return 0.0
        return 0.7 * cj + 0.3 * max(q, _jaccard(map(_ref_key, gold.get("references", [])),
                                                 map(_ref_key, pred.get("references", []))))
    if kind in ("amendment", "claim", "allowable"):
        return cj if cj > 0 else 0.0
    if kind == "estoppel":
        lim = fuzz.token_set_ratio(_norm(gold.get("limitation", "")),
                                   _norm(pred.get("limitation", ""))) / 100
        return max(q, 0.5 * lim + 0.5 * cj) if (cj > 0 or lim > 0.6 or q > 0.8) else 0.0
    # free-text kinds: allowance_reason, examiners_amendment
    txt = fuzz.token_set_ratio(_norm(gold.get("summary", "")),
                               _norm(pred.get("summary", ""))) / 100
    return max(q, txt)


MATCH_THRESHOLD = 0.5


def _greedy_match(golds: list, preds: list, kind: str) -> list[tuple[int, int, float]]:
    pairs = []
    for gi, g in enumerate(golds):
        for pi, p in enumerate(preds):
            s = match_score(kind, g[0], p[0], g[1], p[1])
            if s >= MATCH_THRESHOLD:
                pairs.append((s, gi, pi))
    pairs.sort(reverse=True)
    used_g, used_p, out = set(), set(), []
    for s, gi, pi in pairs:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            out.append((gi, pi, s))
    return out


# ------------------------------------------------------------------ scoring

class KindScore(BaseModel):
    gold: int
    predicted: int
    matched: int
    precision: float | None
    recall: float | None
    f1: float | None


class ScoreReport(BaseModel):
    application_number: str
    run_model: str
    run_prompt_version: str
    docs_scored: int
    by_kind: dict[str, KindScore]
    overall: KindScore
    field_accuracy: dict[str, float]            # e.g. "rejection.references": 0.8
    risk_confusion: dict[str, dict[str, int]]   # gold risk -> predicted risk -> n
    risk_kappa: float | None
    misses: list[dict]                          # gold items not found (for error analysis)
    false_positives: list[dict]


def _prf(g: int, p: int, m: int) -> KindScore:
    prec = m / p if p else None
    rec = m / g if g else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if p or g else None)
    r = lambda x: None if x is None else round(x, 3)  # noqa: E731
    return KindScore(gold=g, predicted=p, matched=m, precision=r(prec), recall=r(rec), f1=r(f1))


def cohen_kappa(pairs: list[tuple[str, str]], labels: list[str]) -> float | None:
    """Linear-weighted Cohen's kappa for ordinal labels (e.g. high/medium/low)."""
    n = len(pairs)
    if n < 2:
        return None
    k = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    w = [[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    obs = [[0] * k for _ in range(k)]
    for a, b in pairs:
        obs[idx[a]][idx[b]] += 1
    ra = [sum(row) for row in obs]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    po = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k)) / n
    pe = sum(w[i][j] * ra[i] * cb[j] for i in range(k) for j in range(k)) / (n * n)
    return None if pe == 0 else round(1 - po / pe, 3)


def scored_kinds(analysis) -> set[str]:
    """Finding types the run was asked to produce. Estoppel is left out of scoring when
    the run had it switched off, so it doesn't count as missed."""
    kinds = set(KIND_FIELDS)
    if (getattr(analysis, "features", None) or {}).get("estoppel") is False:
        kinds.discard("estoppel")
    return kinds


def score(gold: GoldSet, analysis) -> ScoreReport:
    reviewed = {d for d, s in gold.docs.items() if s.reviewed}
    kinds = scored_kinds(analysis)
    golds_by = defaultdict(list)    # (kind, doc) -> [(fields, quote, item)]
    for it in gold.items:
        if it.is_gold and it.doc_id in reviewed and it.kind in kinds:
            golds_by[(it.kind, it.doc_id)].append((it.fields, it.quote, it))
    preds_by = defaultdict(list)
    for f in analysis.findings:
        if f.kind in kinds and f.doc_id in reviewed:
            preds_by[(f.kind, f.doc_id)].append(
                (fields_from_finding(f, analysis.extractions), f.citation.quote, f))

    counts = defaultdict(lambda: [0, 0, 0])
    field_hits: dict[str, list[int]] = defaultdict(list)
    risk_pairs: list[tuple[str, str]] = []
    misses, fps = [], []
    for key in set(golds_by) | set(preds_by):
        kind, _doc = key
        gs, ps = golds_by.get(key, []), preds_by.get(key, [])
        matches = _greedy_match(gs, ps, kind)
        c = counts[kind]
        c[0] += len(gs)
        c[1] += len(ps)
        c[2] += len(matches)
        for gi, pi, _s in matches:
            gf, pf = gs[gi][0], ps[pi][0]
            field_hits[f"{kind}.claims"].append(set(gf.get("claims", [])) ==
                                                set(pf.get("claims", [])))
            if kind == "rejection":
                field_hits["rejection.references"].append(
                    set(map(_ref_key, gf.get("references", []))) ==
                    set(map(_ref_key, pf.get("references", []))))
            if kind == "estoppel":
                field_hits["estoppel.risk"].append(gf.get("risk") == pf.get("risk"))
                field_hits["estoppel.type"].append(
                    gf.get("estoppel_type") == pf.get("estoppel_type"))
                if gf.get("risk") in CHOICES["risk"] and pf.get("risk") in CHOICES["risk"]:
                    risk_pairs.append((gf["risk"], pf["risk"]))
            if kind == "claim":
                field_hits["claim.text>=0.95"].append(
                    fuzz.ratio(_norm(gf.get("text", "")), _norm(pf.get("text", ""))) >= 95)
        mg = {gi for gi, _, _ in matches}
        mp = {pi for _, pi, _ in matches}
        misses += [{"kind": kind, "doc_id": g[2].doc_id, "gold_id": g[2].id,
                    "fields": g[0]} for i, g in enumerate(gs) if i not in mg]
        fps += [{"kind": kind, "doc_id": p[2].doc_id, "finding_id": p[2].id,
                 "statement": p[2].statement[:200]} for i, p in enumerate(ps) if i not in mp]

    by_kind = {k: _prf(*v) for k, v in sorted(counts.items())}
    tot = [sum(v[i] for v in counts.values()) for i in range(3)]
    conf: dict[str, dict[str, int]] = {g: dict(Counter(p for gg, p in risk_pairs if gg == g))
                                       for g in CHOICES["risk"]}
    return ScoreReport(
        application_number=gold.application_number, run_model=analysis.model,
        run_prompt_version=analysis.prompt_version, docs_scored=len(reviewed),
        by_kind=by_kind, overall=_prf(*tot),
        field_accuracy={k: round(sum(v) / len(v), 3) for k, v in sorted(field_hits.items())},
        risk_confusion=conf, risk_kappa=cohen_kappa(risk_pairs, CHOICES["risk"]),
        misses=misses, false_positives=fps,
    )


def verdict_stats(gold: GoldSet) -> dict:
    """Direct stats from the labeling itself (for the run the labels were built from)."""
    model = [i for i in gold.items if i.origin == "model" and i.verdict]
    return {
        "verdicts": dict(Counter(i.verdict for i in model)),
        "error_tags": dict(Counter(t for i in model for t in i.error_tags).most_common()),
        "added_by_kind": dict(Counter(i.kind for i in gold.items if i.origin == "added")),
    }


def to_json(obj) -> str:
    return json.dumps(obj, indent=1, default=str)
