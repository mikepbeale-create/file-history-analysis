"""Experiment runner: grid expansion, runs over saved OCR text, scoring, Pareto frontier."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace as NS

import pytest

from fh_analyzer import prompts
from fh_analyzer.evalset import gold_from_analysis, save_gold
from fh_analyzer.experiment import (
    ExperimentSpec,
    bootstrap_f1,
    expand,
    load_spec,
    pareto_front,
    run_experiment,
    write_reports,
)
from fh_analyzer.pipeline import analyze, save

from .conftest import FakeLLM


def _usage(i, o):
    return NS(input_tokens=i, output_tokens=o, cache_creation_input_tokens=0,
              cache_read_input_tokens=0)


class PricedFake(FakeLLM):
    """FakeLLM that logs a priced call. 'haiku' misses the estoppel + amendment items."""

    def __init__(self, model, cost_log, systems):
        super().__init__()
        self.model, self.cost_log, self.systems = model, cost_log, systems

    def extract(self, system, user, schema):
        self.systems.append(system)
        self.cost_log.check_budget()
        self.cost_log.record(model=self.model, usage=_usage(20_000, 2_000))
        out = super().extract(system, user, schema)
        if "haiku" in self.model and schema.__name__.startswith("Response"):
            out.amendments = []
            if hasattr(out, "estoppel"):
                out.estoppel = []
        return out


@pytest.fixture
def lab(app_info, docs, texts, settings, tmp_path):
    """A saved analysis with OCR text (the experiment's input) plus a fully reviewed gold
    file built from it, so the plain fake scores F1 = 1."""
    a = analyze(app_info, docs, texts, FakeLLM(), settings)
    src = tmp_path / "cache"
    save(a, src / app_info.application_number / "analysis.json")
    g = gold_from_analysis(a)
    for d in g.docs.values():
        d.reviewed = True
    for it in g.items:
        it.verdict = "correct"
    save_gold(g, tmp_path / "gold" / f"{app_info.application_number}.json")
    s = dataclasses.replace(settings, cost_log=tmp_path / "calls.jsonl",
                            model="claude-sonnet-5")
    systems: list[str] = []
    made: list[str] = []

    def factory(cfg, model, cost_log):
        made.append(model)
        return PricedFake(model, cost_log, systems)

    return NS(settings=s, factory=factory, systems=systems, made=made, root=tmp_path,
              src=src, gold=tmp_path / "gold")


def _spec(lab, **kw):
    base = dict(name="t", source_dir=str(lab.src), gold_dir=str(lab.gold), workers=1)
    return ExperimentSpec.model_validate({**base, **kw})


def test_expand_names_only_the_varying_axes(settings):
    spec = ExperimentSpec(name="x", grid={"model": ["claude-sonnet-5", "claude-haiku-4-5"],
                                          "completeness_retry": [True, False],
                                          "judge": [False]})
    cfgs = expand(spec, settings)
    assert [c.name for c in cfgs] == ["sonnet-5__retry-on", "sonnet-5__retry-off",
                                      "haiku-4-5__retry-on", "haiku-4-5__retry-off"]
    assert all(c.judge is False and c.prompt == prompts.PROMPT_VERSION for c in cfgs)
    assert expand(ExperimentSpec(name="x"), settings)[0].name == "base"
    with pytest.raises(ValueError, match="unknown option"):
        ExperimentSpec(name="x", grid={"temperature": [0, 1]})
    with pytest.raises(KeyError, match="Unknown prompt version"):
        expand(ExperimentSpec(name="x", grid={"prompt": ["nope"]}), settings)


def test_prompt_versions_from_registry_and_folder(tmp_path):
    abl = prompts.get_prompt_set(prompts.PROMPT_VERSION + "-no-rule3")
    assert "report ALL" in prompts.CURRENT.office_action
    assert "report ALL" not in abl.office_action and abl.synthesis == prompts.SYNTHESIS
    d = tmp_path / "v3-terse"
    d.mkdir()
    (d / "response.txt").write_text("TERSE RESPONSE PROMPT", encoding="utf-8")
    ps = prompts.get_prompt_set(str(d))
    assert ps.version == "v3-terse" and ps.response == "TERSE RESPONSE PROMPT"
    assert ps.claims == prompts.CLAIMS                    # untouched prompts fall back


def test_run_score_and_pareto(lab):
    spec = _spec(lab, grid={"model": ["claude-sonnet-5", "claude-haiku-4-5",
                                      "claude-opus-5-5"]})
    outs = run_experiment(spec, lab.settings, lab.root / "runs", factory=lab.factory,
                          progress=lambda m: None)
    assert [o.status for o in outs] == ["done"] * 3
    exp = lab.root / "runs" / "t"
    assert (exp / "sonnet-5" / "99999999.json").exists()
    saved = json.loads((exp / "sonnet-5" / "99999999.json").read_text())
    assert saved["texts"] == {} and saved["cost"]["total_usd"] > 0    # small, and priced

    res = {r.config: r for r in write_reports(exp)}
    assert res["sonnet-5"].f1 == 1.0 and res["opus-5-5"].f1 == 1.0
    assert res["haiku-4-5"].f1 < 1.0 and res["haiku-4-5"].recall < 1.0
    assert res["haiku-4-5"].usd_per_patent < res["sonnet-5"].usd_per_patent \
        < res["opus-5-5"].usd_per_patent
    # Opus is as good as Sonnet but dearer -> dominated. The other two are real choices.
    assert {k for k, r in res.items() if r.pareto} == {"sonnet-5", "haiku-4-5"}
    assert res["haiku-4-5"].f1_by_kind["estoppel"] == 0.0
    md = (exp / "results.md").read_text()
    assert "★" in md and "frontier.png" in md and (exp / "frontier.png").stat().st_size > 5000
    assert (exp / "results.csv").exists() and (exp / "runs.csv").exists()
    # Every call also went to the main cost log, tagged with the experiment.
    log = (lab.root / "calls.jsonl").read_text()
    assert '"run_id":"exp:t/haiku-4-5/99999999"' in log


def test_resume_skips_finished_runs_and_force_reruns(lab):
    spec = _spec(lab)
    run_experiment(spec, lab.settings, lab.root / "runs", factory=lab.factory,
                   progress=lambda m: None)
    n = len(lab.made)
    outs = run_experiment(spec, lab.settings, lab.root / "runs", factory=lab.factory,
                          progress=lambda m: None)
    assert [o.status for o in outs] == ["skipped"] and len(lab.made) == n
    run_experiment(spec, lab.settings, lab.root / "runs", factory=lab.factory, force=True,
                   progress=lambda m: None)
    assert len(lab.made) == 2 * n


def test_prompt_and_retry_options_reach_the_pipeline(lab):
    spec = _spec(lab, grid={"prompt": [prompts.PROMPT_VERSION + "-no-rule3"]},
                 fixed={"completeness_retry": False})
    run_experiment(spec, lab.settings, lab.root / "runs", factory=lab.factory,
                   progress=lambda m: None)
    assert lab.systems and all("report ALL" not in s for s in lab.systems)
    a = json.loads((lab.root / "runs" / "t" / "base" / "99999999.json").read_text())
    assert a["prompt_version"].endswith("-no-rule3")
    assert "completeness_retry" not in a["cost"]["by_stage"]


def test_budget_stops_the_experiment(lab):
    spec = _spec(lab, grid={"model": ["claude-sonnet-5", "claude-haiku-4-5"]},
                 budget_usd=0.05)
    msgs = []
    outs = run_experiment(spec, lab.settings, lab.root / "runs", factory=lab.factory,
                          progress=msgs.append)
    assert outs[-1].status == "budget" and len(outs) == 1
    assert any("Budget" in m for m in msgs)
    assert (lab.root / "runs" / "t" / "sonnet-5" / "99999999.partial.json").exists()
    assert not (lab.root / "runs" / "t" / "sonnet-5" / "99999999.json").exists()


def test_repeats_report_run_to_run_spread(lab):
    spec = _spec(lab, repeats=2)
    run_experiment(spec, lab.settings, lab.root / "runs", factory=lab.factory,
                   progress=lambda m: None)
    assert (lab.root / "runs" / "t" / "base" / "99999999.r2.json").exists()
    (r,) = write_reports(lab.root / "runs" / "t")
    assert r.runs == 2 and r.f1_repeat_sd == 0.0


def test_pareto_and_bootstrap_helpers():
    pts = [("a", 1.0, 0.8), ("b", 2.0, 0.8), ("c", 0.5, 0.6), ("d", 3.0, 0.9),
           ("e", None, 0.99)]
    assert pareto_front(pts) == {"a", "c", "d"}
    lo, hi = bootstrap_f1([(10, 10, 9), (10, 10, 5), (10, 10, 7)])
    assert lo <= 0.7 <= hi and lo >= 0.5 and hi <= 0.9
    assert bootstrap_f1([(10, 10, 9)]) == (None, None)


def test_cli_dry_run_and_score_only(lab, tmp_path, capsys, monkeypatch):
    from fh_analyzer import experiment_cli

    grid = tmp_path / "g.yaml"
    grid.write_text(f"name: t\nsource_dir: {lab.src.as_posix()}\n"
                    f"gold_dir: {lab.gold.as_posix()}\n"
                    "grid:\n  model: [claude-sonnet-5, claude-haiku-4-5]\n", encoding="utf-8")
    assert load_spec(grid).grid["model"][1] == "claude-haiku-4-5"
    monkeypatch.setattr(experiment_cli, "get_settings", lambda: lab.settings)
    assert experiment_cli.main([str(grid), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "2 configurations × 1 patents" in out and "haiku-4-5" in out

    run_experiment(load_spec(grid), lab.settings, lab.root / "runs", factory=lab.factory,
                   progress=lambda m: None)
    assert experiment_cli.main([str(grid), "--score-only", "--runs-dir",
                                str(lab.root / "runs")]) == 0
    assert "# Experiment: t" in capsys.readouterr().out

    from fh_analyzer.score_cli import main as score_main
    assert score_main(["--experiment", str(lab.root / "runs" / "t")]) == 0


def test_workspace_env_moves_every_output(monkeypatch, tmp_path):
    from fh_analyzer.config import get_settings

    for v in ("FHA_CACHE_DIR", "FHA_COST_LOG", "FHA_GOLD_DIR"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.delenv("FHA_WORKSPACE", raising=False)
    s = get_settings()
    assert not s.is_private and s.gold_dir.as_posix() == "gold"
    monkeypatch.setenv("FHA_WORKSPACE", str(tmp_path / "private"))
    s = get_settings()
    assert s.is_private
    assert s.cache_dir == tmp_path / "private" / "data" / "cache"
    assert s.cost_log == tmp_path / "private" / "data" / "costs" / "calls.jsonl"
    assert s.gold_dir == tmp_path / "private" / "gold"
    assert s.runs_dir == tmp_path / "private" / "runs"


def test_failed_configuration_is_kept_off_the_frontier(lab):
    """A model that rejects every call costs $0 and scores 0; it must not look like a choice."""

    class Broken(PricedFake):
        def extract(self, system, user, schema):
            raise RuntimeError("tool_choice not supported for this model")

    def factory(cfg, model, cost_log):
        cls = Broken if "opus" in model else PricedFake
        lab.made.append(model)
        return cls(model, cost_log, lab.systems)

    spec = _spec(lab, grid={"model": ["claude-sonnet-5", "claude-opus-5-5"]})
    run_experiment(spec, lab.settings, lab.root / "runs", factory=factory,
                   progress=lambda m: None)
    res = {r.config: r for r in write_reports(lab.root / "runs" / "t")}
    assert res["opus-5-5"].failed_steps > 0 and res["opus-5-5"].usd_per_patent == 0
    # Resume re-runs only the configuration whose steps failed (e.g. after a client fix).
    n = len(lab.made)
    outs = run_experiment(spec, lab.settings, lab.root / "runs", factory=lab.factory,
                          progress=lambda m: None)
    assert {o.config: o.status for o in outs} == {"sonnet-5": "skipped", "opus-5-5": "done"}
    assert len(lab.made) == n + 1
    assert res["sonnet-5"].pareto and not res["opus-5-5"].pareto
    md = (lab.root / "runs" / "t" / "results.md").read_text()
    assert "| ⚠ | `opus-5-5`" in md
