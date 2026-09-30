"""Experiment runner: compare models, prompt versions and pipeline options on the gold set.

    fh-experiment experiments/model_sweep.yaml            # run every configuration, then score
    fh-experiment experiments/model_sweep.yaml --dry-run  # list configurations + rough cost

A grid file names the axes to vary; every combination is one *configuration*:

    name: model-sweep
    grid:
      model: [claude-sonnet-5, claude-haiku-4-5]
      prompt: [2026-09-25.2, 2026-09-25.2-no-rule3]
    fixed:
      judge: false

Each configuration runs over the labeled patents using the OCR text already saved in
data/cache/<app>/analysis.json (no USPTO download, no OCR), and writes

    runs/<experiment>/<configuration>/<app>.json     # the analysis, same format as the app's
    runs/<experiment>/<configuration>/config.json

The runs are then scored against gold/<app>.json with the same scorer as fh-score and
combined into runs/<experiment>/results.{md,csv} plus a quality-vs-cost chart. The
configurations that nothing else beats on both F1 and dollars per patent (the Pareto
frontier) are marked: those are the real choices; everything else is dominated.

Design notes
------------
* Resumable: a finished (configuration, patent) is skipped on re-run, so a crash or a
  budget stop loses at most one patent's work. Use --force to redo.
* No extraction cache is shared between runs. Every run pays for its own API calls, so
  dollars per patent are measured, not reduced by earlier runs.
* Every call is also written to the main cost log (fh-costs) with run_id
  "exp:<experiment>/<configuration>/<app>".
* `repeats: N` runs each configuration N times. The models are not deterministic, so
  this shows how much F1 moves between identical runs, which tells you whether a gap
  between two configurations is real.
* The F1 interval is a bootstrap over patents (resample patents with replacement). With a
  handful of patents it is wide, and it should be.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import random
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from . import prompts
from .config import Settings
from .costs import BudgetExceeded, CostLog, Pricing
from .evalset import KIND_FIELDS, GoldSet, ScoreReport, load_gold, score
from .pipeline import ROUTES, Analysis, analyze, load, save

log = logging.getLogger(__name__)

# axis -> (default, short label used in configuration names)
AXES: dict[str, tuple[Any, str]] = {
    "model": (None, ""),                 # None -> Settings.model
    "prompt": (None, "p-"),              # None -> prompts.CURRENT (a version name or a folder)
    "completeness_retry": (True, "retry-"),
    "strict": (True, "strict-"),         # strict tool use (API-enforced schema)
    "judge": (False, "judge-"),          # false | true (same model) | a model id
    "max_tokens": (8000, "maxtok-"),
}


# ------------------------------------------------------------------ spec

class ExperimentSpec(BaseModel):
    name: str
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    fixed: dict[str, Any] = Field(default_factory=dict)
    patents: Literal["all"] | list[str] = "all"
    # "all": analyze every document, so $ per patent is the real cost of a full run.
    # "reviewed": only documents marked reviewed in the gold file (the only ones scored);
    #             much cheaper, but $ per patent then covers only those documents.
    docs: Literal["all", "reviewed"] = "all"
    repeats: int = Field(1, ge=1, le=10)
    budget_usd: float | None = None       # stop the whole experiment once this is spent
    workers: int = Field(4, ge=1, le=16)
    source_dir: str | None = None         # default Settings.cache_dir (data/cache)
    gold_dir: str | None = None           # default Settings.gold_dir (gold)

    @field_validator("name")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        if not re.fullmatch(r"[\w.-]+", v):
            raise ValueError("name may contain only letters, digits, '.', '_' and '-'")
        return v

    @field_validator("grid", "fixed")
    @classmethod
    def _known_axes(cls, v: dict) -> dict:
        bad = set(v) - set(AXES)
        if bad:
            raise ValueError(f"unknown option(s) {sorted(bad)}; known: {', '.join(AXES)}")
        return v

    @field_validator("patents", mode="before")
    @classmethod
    def _patents_as_str(cls, v):
        return [str(x) for x in v] if isinstance(v, list) else v


def load_spec(path: Path) -> ExperimentSpec:
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".json":
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    data.setdefault("name", Path(path).stem)
    for k, v in list((data.get("grid") or {}).items()):
        if not isinstance(v, list):
            data["grid"][k] = [v]
    return ExperimentSpec.model_validate(data)


class RunConfig(BaseModel):
    name: str
    model: str
    prompt: str                 # prompt version name (or folder path)
    completeness_retry: bool = True
    strict: bool = True
    judge: bool | str = False
    max_tokens: int = 8000

    @property
    def prompt_set(self) -> prompts.PromptSet:
        return prompts.get_prompt_set(self.prompt)


def _short(axis: str, value: Any) -> str:
    if axis == "model":
        return str(value).removeprefix("claude-")
    if axis == "prompt":
        return AXES[axis][1] + Path(str(value)).name
    if isinstance(value, bool):
        return AXES[axis][1] + ("on" if value else "off")
    if axis == "judge":
        return AXES[axis][1] + str(value).removeprefix("claude-")
    return AXES[axis][1] + str(value)


def expand(spec: ExperimentSpec, settings: Settings) -> list[RunConfig]:
    """Every combination of the grid axes, with fixed options and defaults filled in.
    Configuration names mention only the axes that vary, e.g. 'haiku-4-5__retry-off'."""
    base = {k: d for k, (d, _) in AXES.items()}
    base["model"] = settings.model
    base["prompt"] = prompts.CURRENT.version
    base.update(spec.fixed)
    axes = [a for a in AXES if a in spec.grid]      # stable, documented order
    out, seen = [], set()
    for combo in itertools.product(*(spec.grid[a] for a in axes)):
        opts = {**base, **dict(zip(axes, combo, strict=True))}
        name = "__".join(_short(a, v) for a, v in zip(axes, combo, strict=True)
                         if len(spec.grid[a]) > 1) or "base"
        name = re.sub(r"[^\w.+-]", "_", name)
        if name in seen:
            raise ValueError(f"duplicate configuration {name!r} in grid")
        seen.add(name)
        cfg = RunConfig(name=name, **opts)
        prompts.get_prompt_set(cfg.prompt)          # fail early on an unknown prompt
        out.append(cfg)
    return out


# ------------------------------------------------------------------ inputs

class PatentInput(BaseModel):
    app_no: str
    gold: GoldSet
    source: Path

    def reviewed(self) -> set[str]:
        return {d for d, s in self.gold.docs.items() if s.reviewed}


def find_patents(spec: ExperimentSpec, settings: Settings) -> tuple[list[PatentInput],
                                                                     list[str]]:
    """Labeled patents that have reviewed documents and saved OCR text.
    Returns (usable, reasons-for-skipping)."""
    gold_dir = Path(spec.gold_dir) if spec.gold_dir else settings.gold_dir
    src = Path(spec.source_dir) if spec.source_dir else settings.cache_dir
    wanted = spec.patents if spec.patents != "all" else \
        sorted(p.stem for p in gold_dir.glob("*.json"))
    usable, skipped = [], []
    for app in wanted:
        g = load_gold(gold_dir / f"{app}.json")
        a = src / app / "analysis.json"
        if g is None:
            skipped.append(f"{app}: no gold file in {gold_dir}")
        elif not any(d.reviewed for d in g.docs.values()):
            skipped.append(f"{app}: no documents marked reviewed yet")
        elif not a.exists():
            skipped.append(f"{app}: no saved analysis with OCR text at {a}")
        else:
            usable.append(PatentInput(app_no=app, gold=g, source=a))
    return usable, skipped


def _inputs(p: PatentInput, docs_mode: str):
    base = load(p.source)
    texts = base.texts
    if docs_mode == "reviewed":
        texts = {k: v for k, v in texts.items() if k in p.reviewed()}
    return base, texts


# ------------------------------------------------------------------ running

LLMFactory = Callable[[RunConfig, str, CostLog], Any]   # (cfg, model, cost_log) -> LLM


def anthropic_factory(settings: Settings) -> LLMFactory:
    from .llm import AnthropicLLM

    def make(cfg: RunConfig, model: str, cost_log: CostLog):
        return AnthropicLLM(settings.anthropic_api_key or "", model, max_tokens=cfg.max_tokens,
                            strict=cfg.strict, cost_log=cost_log)
    return make


def run_path(exp_dir: Path, cfg_name: str, app: str, rep: int) -> Path:
    return exp_dir / cfg_name / (f"{app}.json" if rep == 1 else f"{app}.r{rep}.json")


class RunOutcome(BaseModel):
    config: str
    app: str
    repeat: int
    status: Literal["done", "skipped", "failed", "budget"]
    cost_usd: float = 0.0
    message: str = ""


def run_one(cfg: RunConfig, p: PatentInput, spec: ExperimentSpec, settings: Settings,
            exp_dir: Path, rep: int, factory: LLMFactory,
            budget_left: float | None) -> RunOutcome:
    out = run_path(exp_dir, cfg.name, p.app_no, rep)
    base, texts = _inputs(p, spec.docs)
    log_ = CostLog(settings.cost_log, app_no=p.app_no, max_usd=budget_left,
                   run_id=f"exp:{spec.name}/{cfg.name}/{p.app_no}"
                          + (f"/r{rep}" if rep > 1 else ""))
    llm = factory(cfg, cfg.model, log_)
    judge = None
    if cfg.judge is True:
        judge = llm
    elif isinstance(cfg.judge, str) and cfg.judge:
        judge = factory(cfg, cfg.judge, log_)
    a = analyze(base.app, base.documents, texts, llm, settings, judge=judge,
                workers=spec.workers, extract_cache=None, prompt_set=cfg.prompt_set,
                completeness_retry=cfg.completeness_retry)
    a.cost = log_.summary()
    a.texts = {}                    # the OCR text stays in the source analysis; keep runs small
    a.pdf_dir = str(p.source)
    usd = log_.total_usd
    if any("BudgetExceeded" in e for e in a.errors.values()):
        save(a, out.with_suffix(".partial.json"))
        return RunOutcome(config=cfg.name, app=p.app_no, repeat=rep, status="budget",
                          cost_usd=usd, message="experiment budget reached")
    save(a, out)
    msg = f"{len(a.errors)} step(s) failed" if a.errors else ""
    return RunOutcome(config=cfg.name, app=p.app_no, repeat=rep, status="done",
                      cost_usd=usd, message=msg)


def _had_errors(path: Path) -> bool:
    """A finished run with failed steps (e.g. every call rejected by the API) is redone on
    resume, so fixing the cause and re-running the grid retries only those runs."""
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("errors"))
    except (OSError, ValueError):
        return True


def run_experiment(spec: ExperimentSpec, settings: Settings, runs_dir: Path | None = None,
                   *, factory: LLMFactory | None = None, force: bool = False,
                   progress: Callable[[str], None] = print) -> list[RunOutcome]:
    exp_dir = (runs_dir or settings.runs_dir) / spec.name
    configs = expand(spec, settings)
    patents, skipped = find_patents(spec, settings)
    for s in skipped:
        progress(f"skip {s}")
    if not patents:
        raise RuntimeError("No usable labeled patents (see the skip messages above).")
    factory = factory or anthropic_factory(settings)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "spec.json").write_text(spec.model_dump_json(indent=1), encoding="utf-8")

    spent, outcomes = 0.0, []
    jobs = [(c, p, r) for c in configs for p in patents for r in range(1, spec.repeats + 1)]
    for i, (cfg, p, rep) in enumerate(jobs, 1):
        cdir = exp_dir / cfg.name
        cdir.mkdir(exist_ok=True)
        (cdir / "config.json").write_text(cfg.model_dump_json(indent=1), encoding="utf-8")
        tag = f"[{i}/{len(jobs)}] {cfg.name} · {p.app_no}" + (f" · r{rep}" if rep > 1 else "")
        done = run_path(exp_dir, cfg.name, p.app_no, rep)
        if done.exists() and not force:
            if not _had_errors(done):
                outcomes.append(RunOutcome(config=cfg.name, app=p.app_no, repeat=rep,
                                           status="skipped", message="already done"))
                progress(f"{tag}: already done")
                continue
            progress(f"{tag}: previous run had failed steps; running it again")
        left = None if spec.budget_usd is None else spec.budget_usd - spent
        if left is not None and left <= 0:
            progress(f"Budget of ${spec.budget_usd:.2f} reached; stopping. Re-run to resume.")
            break
        progress(f"{tag}: running")
        try:
            o = run_one(cfg, p, spec, settings, exp_dir, rep, factory, left)
        except BudgetExceeded as e:
            o = RunOutcome(config=cfg.name, app=p.app_no, repeat=rep, status="budget",
                           message=str(e))
        except Exception as e:      # one bad configuration must not sink the experiment
            log.exception("run failed")
            o = RunOutcome(config=cfg.name, app=p.app_no, repeat=rep, status="failed",
                           message=f"{type(e).__name__}: {e}")
        spent += o.cost_usd
        outcomes.append(o)
        progress(f"{tag}: {o.status} ${o.cost_usd:.2f}" + (f" ({o.message})" if o.message
                                                          else ""))
        if o.status == "budget":
            progress(f"Budget of ${spec.budget_usd:.2f} reached; stopping. Re-run to resume.")
            break
    return outcomes


# ------------------------------------------------------------------ dry-run estimate

def estimate_usd(cfg: RunConfig, p: PatentInput, spec: ExperimentSpec,
                 pricing: Pricing) -> float | None:
    """Rough: re-price the token counts of the saved baseline run at this model's prices,
    scaled by the share of text that will be sent. Judge and retries are not modelled."""
    base = load(p.source)
    u = base.usage or {}
    if not u.get("input_tokens") and not u.get("cache_read_tokens"):
        chars = sum(len(pg.text) for t in base.texts.values() for pg in t.pages)
        u = {"input_tokens": int(chars / 3.8), "output_tokens": int(chars / 3.8 * 0.15)}
    routed = {d.doc_id for d in base.documents if d.category in ROUTES}
    size = lambda ids: sum(len(pg.text) for k, t in base.texts.items()  # noqa: E731
                           if k in ids and k in routed for pg in t.pages)
    share = 1.0
    if spec.docs == "reviewed":
        share = size(p.reviewed()) / max(size(set(base.texts)), 1)
    c = pricing.cost(cfg.model, u.get("input_tokens", 0), u.get("output_tokens", 0),
                     u.get("cache_write_tokens", 0), u.get("cache_read_tokens", 0))
    return None if c is None else c * share


# ------------------------------------------------------------------ scoring

class ConfigResult(BaseModel):
    config: str
    model: str
    prompt: str
    options: dict[str, Any]
    patents: int
    runs: int
    gold: int
    predicted: int
    matched: int
    precision: float | None
    recall: float | None
    f1: float | None
    f1_lo: float | None = None           # bootstrap interval over patents
    f1_hi: float | None = None
    f1_repeat_sd: float | None = None    # run-to-run noise (repeats > 1)
    f1_by_kind: dict[str, float | None] = Field(default_factory=dict)
    usd_per_patent: float | None
    usd_total: float
    unpriced: bool = False
    verified_rate: float | None = None   # citation grounding: exact quote on cited page
    failed_steps: int = 0
    pareto: bool = False


def _f1(g: int, p: int, m: int) -> float | None:
    if not g and not p:
        return None
    return 2 * m / (g + p) if (g + p) else 0.0


def bootstrap_f1(per_patent: list[tuple[int, int, int]], n: int = 2000,
                 seed: int = 0, level: float = 0.9) -> tuple[float | None, float | None]:
    """Percentile interval of micro-F1 when the set of patents is resampled."""
    if len(per_patent) < 2:
        return None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        s = [rng.choice(per_patent) for _ in per_patent]
        f = _f1(sum(x[0] for x in s), sum(x[1] for x in s), sum(x[2] for x in s))
        if f is not None:
            vals.append(f)
    if not vals:
        return None, None
    vals.sort()
    lo = vals[int((1 - level) / 2 * (len(vals) - 1))]
    hi = vals[int((1 + level) / 2 * (len(vals) - 1))]
    return round(lo, 3), round(hi, 3)


def pareto_front(points: list[tuple[str, float | None, float | None]]) -> set[str]:
    """(name, cost, quality) -> names not dominated (another point at least as cheap AND at
    least as good, and strictly better on one). Points without cost or quality are left out."""
    pts = [(n, c, q) for n, c, q in points if c is not None and q is not None]
    front = set()
    for n, c, q in pts:
        if not any((c2 <= c and q2 >= q) and (c2 < c or q2 > q) for _, c2, q2 in pts):
            front.add(n)
    return front


def _run_files(cdir: Path) -> list[tuple[str, int, Path]]:
    out = []
    for f in sorted(cdir.glob("*.json")):
        m = re.fullmatch(r"(\w+?)(?:\.r(\d+))?\.json", f.name)
        if m and f.name != "config.json":
            out.append((m.group(1), int(m.group(2) or 1), f))
    return out


def score_experiment(exp_dir: Path, gold_dir: Path | None = None
                     ) -> tuple[list[ConfigResult], list[dict], list[ScoreReport]]:
    """Score every run in runs/<experiment>/ against the gold set.
    Returns (one row per configuration, one row per run, the full score reports)."""
    spec_file = exp_dir / "spec.json"
    spec = ExperimentSpec.model_validate_json(spec_file.read_text(encoding="utf-8")) \
        if spec_file.exists() else None
    if gold_dir is None:
        from .config import get_settings

        gold_dir = spec.gold_dir if spec and spec.gold_dir else get_settings().gold_dir
    gold_dir = Path(gold_dir)
    golds: dict[str, GoldSet | None] = {}
    results, rows, reports = [], [], []
    for cdir in sorted(d for d in exp_dir.iterdir() if (d / "config.json").exists()):
        cfg = RunConfig.model_validate_json((cdir / "config.json").read_text("utf-8"))
        per_patent: dict[str, list[int]] = {}
        by_rep: dict[int, list[int]] = {}
        kinds: dict[str, list[int]] = {}
        usd, unpriced, verified, failed = [], False, [], 0
        for app, rep, f in _run_files(cdir):
            if app not in golds:
                golds[app] = load_gold(gold_dir / f"{app}.json")
            g = golds[app]
            if g is None:
                continue
            a: Analysis = load(f)
            r = score(g, a)
            reports.append(r)
            o = r.overall
            for acc in (per_patent.setdefault(app, [0, 0, 0]),
                        by_rep.setdefault(rep, [0, 0, 0])):
                acc[0] += o.gold
                acc[1] += o.predicted
                acc[2] += o.matched
            for k, ks in r.by_kind.items():
                acc = kinds.setdefault(k, [0, 0, 0])
                acc[0] += ks.gold
                acc[1] += ks.predicted
                acc[2] += ks.matched
            cost = a.cost.total_usd if a.cost else None
            unpriced |= bool(a.cost and a.cost.unpriced_models)
            usd.append(cost or 0.0)
            if a.grounding:
                verified.append(a.grounding.rate_verified)
            failed += len(a.errors)
            rows.append({"config": cfg.name, "app": app, "repeat": rep, "gold": o.gold,
                         "predicted": o.predicted, "matched": o.matched,
                         "precision": o.precision, "recall": o.recall, "f1": o.f1,
                         "usd": cost, "calls": a.cost.calls if a.cost else None,
                         "failed_steps": len(a.errors),
                         "verified_rate": a.grounding.rate_verified if a.grounding else None})
        if not usd:
            continue                                  # no scorable runs yet
        g, p, m = (sum(v[i] for v in per_patent.values()) for i in range(3))
        f1 = _f1(g, p, m)
        lo, hi = bootstrap_f1([tuple(v) for v in per_patent.values()])
        rep_f1 = [x for x in (_f1(*v) for v in by_rep.values()) if x is not None]
        sd = None
        if len(rep_f1) > 1:
            mu = sum(rep_f1) / len(rep_f1)
            sd = round(math.sqrt(sum((x - mu) ** 2 for x in rep_f1) / (len(rep_f1) - 1)), 3)
        rnd = lambda x: None if x is None else round(x, 3)  # noqa: E731
        results.append(ConfigResult(
            config=cfg.name, model=cfg.model, prompt=cfg.prompt_set.version,
            options=cfg.model_dump(exclude={"name", "model", "prompt"}),
            patents=len(per_patent), runs=len(usd), gold=g, predicted=p, matched=m,
            precision=rnd(m / p if p else None), recall=rnd(m / g if g else None), f1=rnd(f1),
            f1_lo=lo, f1_hi=hi, f1_repeat_sd=sd,
            f1_by_kind={k: rnd(_f1(*v)) for k, v in sorted(kinds.items())},
            usd_per_patent=None if unpriced else round(sum(usd) / len(usd), 4),
            usd_total=round(sum(usd), 4), unpriced=unpriced,
            verified_rate=rnd(sum(verified) / len(verified)) if verified else None,
            failed_steps=failed,
        ))
    # A configuration with failed steps is not a real choice: its $ and F1 describe calls that
    # errored, not the model (a run where every call fails costs $0 and would otherwise sit
    # on the frontier). Such configurations are reported, but kept off the frontier and chart.
    front = pareto_front([(r.config, r.usd_per_patent, r.f1) for r in results
                          if not r.failed_steps])
    for r in results:
        r.pareto = r.config in front
    results.sort(key=lambda r: (bool(r.failed_steps), -(r.f1 or 0), r.usd_per_patent or 0))
    return results, rows, reports


# ------------------------------------------------------------------ reporting

def _pct(x) -> str:
    return "–" if x is None else f"{x:.0%}"


def results_markdown(name: str, results: list[ConfigResult], chart: str | None) -> str:
    kinds = [k for k in KIND_FIELDS if any(k in r.f1_by_kind for r in results)]
    lines = [f"# Experiment: {name}", ""]
    if chart:
        lines += [f"![F1 vs cost]({chart})", ""]
    lines += ["★ = on the Pareto frontier (no other configuration is both cheaper and more "
              "accurate). ⚠ = some steps failed; the numbers describe errors, not the model, so "
              "the configuration is left off the frontier and the chart. F1 is micro-averaged "
              "over all scored items; the interval is a 90% bootstrap over patents.", ""]
    head = ["", "configuration", "F1", "F1 90% CI", "precision", "recall", "$ / patent",
            "citations verified", "patents", "failed steps"]
    if any(r.f1_repeat_sd is not None for r in results):
        head.insert(4, "run-to-run SD")
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(["---"] * 2 + ["---:"] * (len(head) - 2)) + "|")
    for r in results:
        ci = f"{_pct(r.f1_lo)}–{_pct(r.f1_hi)}" if r.f1_lo is not None else "–"
        cost = "unpriced" if r.unpriced else f"${r.usd_per_patent:,.2f}"
        mark = "★" if r.pareto else ("⚠" if r.failed_steps else "")
        row = [mark, f"`{r.config}`", _pct(r.f1), ci, _pct(r.precision),
               _pct(r.recall), cost, _pct(r.verified_rate), str(r.patents),
               str(r.failed_steps)]
        if "run-to-run SD" in head:
            row.insert(4, "–" if r.f1_repeat_sd is None else f"{r.f1_repeat_sd:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    if kinds:
        lines += ["", "## F1 by finding type", "",
                  "| configuration | " + " | ".join(kinds) + " |",
                  "|---|" + "---:|" * len(kinds)]
        for r in results:
            lines.append(f"| `{r.config}` | " +
                         " | ".join(_pct(r.f1_by_kind.get(k)) for k in kinds) + " |")
    lines += ["", "## Configurations", "",
              "| configuration | model | prompt | options |", "|---|---|---|---|"]
    for r in results:
        opts = ", ".join(f"{k}={v}" for k, v in r.options.items())
        lines.append(f"| `{r.config}` | {r.model} | {r.prompt} | {opts} |")
    return "\n".join(lines) + "\n"


def plot(results: list[ConfigResult], path: Path, title: str) -> Path | None:
    """F1 against $ per patent, one point per configuration; the frontier is highlighted."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the chart (pip install matplotlib)",
              file=sys.stderr)
        return None
    pts = [r for r in results if r.usd_per_patent is not None and r.f1 is not None
           and not r.failed_steps]
    if not pts:
        return None
    ink, muted, accent, grid = "#1f2328", "#8c959f", "#0969da", "#e6e8eb"
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(muted)
    ax.tick_params(colors=ink, labelsize=9)
    ax.grid(True, color=grid, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.margins(x=0.18, y=0.08)
    costs = [r.usd_per_patent for r in pts]
    if min(costs) > 0 and max(costs) / min(costs) > 8:
        ax.set_xscale("log")
    front = sorted((r for r in pts if r.pareto), key=lambda r: r.usd_per_patent)
    if len(front) > 1:
        ax.step([r.usd_per_patent for r in front], [r.f1 for r in front], where="post",
                color=accent, linewidth=1.2, alpha=0.5, zorder=1)
    for r in pts:
        on = r.pareto
        if r.f1_lo is not None:
            ax.errorbar(r.usd_per_patent, r.f1, yerr=[[r.f1 - r.f1_lo], [r.f1_hi - r.f1]],
                        fmt="none", ecolor=accent if on else muted, alpha=0.45,
                        elinewidth=1, capsize=2, zorder=2)
        ax.scatter(r.usd_per_patent, r.f1, s=56 if on else 34, zorder=3,
                   color=accent if on else "white", edgecolor=accent if on else muted,
                   linewidth=1.3)
        ax.annotate(r.config, (r.usd_per_patent, r.f1), xytext=(6, 5),
                    textcoords="offset points", fontsize=7.5,
                    color=ink if on else muted, fontweight="bold" if on else "normal")
    ax.set_xlabel("Dollars per patent (measured API spend)", color=ink, fontsize=10)
    ax.set_ylabel("F1 vs. expert gold labels", color=ink, fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.2f}"))
    ax.set_title(title, loc="left", color=ink, fontsize=12, fontweight="bold", pad=14)
    ax.text(0, 1.01, "Filled = Pareto frontier (nothing else is both cheaper and better). "
            "Bars: 90% bootstrap over patents.", transform=ax.transAxes, fontsize=8,
            color=muted)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_reports(exp_dir: Path, gold_dir: Path | None = None) -> list[ConfigResult]:
    import pandas as pd

    results, rows, reports = score_experiment(exp_dir, gold_dir)
    if not results:
        return results
    flat = []
    for r in results:
        d = r.model_dump(exclude={"options", "f1_by_kind"})
        d.update({f"opt_{k}": v for k, v in r.options.items()})
        d.update({f"f1_{k}": v for k, v in r.f1_by_kind.items()})
        flat.append(d)
    pd.DataFrame(flat).to_csv(exp_dir / "results.csv", index=False)
    pd.DataFrame(rows).to_csv(exp_dir / "runs.csv", index=False)
    (exp_dir / "scores.json").write_text(
        json.dumps([r.model_dump() for r in reports], indent=1, default=str), encoding="utf-8")
    chart = plot(results, exp_dir / "frontier.png", f"Quality vs. cost: {exp_dir.name}")
    (exp_dir / "results.md").write_text(
        results_markdown(exp_dir.name, results, chart.name if chart else None),
        encoding="utf-8")
    return results
