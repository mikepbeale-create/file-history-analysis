"""Run a model / prompt / option grid over the gold-labeled patents and compare F1 vs cost.

    fh-experiment experiments/model_sweep.yaml             # run (resumes), score, chart
    fh-experiment experiments/model_sweep.yaml --dry-run   # configurations + rough cost only
    fh-experiment experiments/model_sweep.yaml --score-only
    fh-experiment experiments/model_sweep.yaml --force     # redo finished runs

Results: runs/<name>/results.md, results.csv, runs.csv, frontier.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import get_settings
from .costs import Pricing
from .experiment import (
    estimate_usd,
    expand,
    find_patents,
    load_spec,
    run_experiment,
    write_reports,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("grid", type=Path, help="grid file (.yaml or .json)")
    ap.add_argument("--runs-dir", type=Path, help="default: <workspace>/runs")
    ap.add_argument("--dry-run", action="store_true",
                    help="list configurations and a rough cost estimate; no API calls")
    ap.add_argument("--score-only", action="store_true", help="re-score existing runs")
    ap.add_argument("--force", action="store_true", help="re-run finished runs")
    ap.add_argument("--yes", "-y", action="store_true", help="don't ask before spending")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    try:
        spec = load_spec(a.grid)
    except Exception as e:
        print(f"Invalid grid file {a.grid}: {e}", file=sys.stderr)
        return 2
    s = get_settings()
    runs_dir = a.runs_dir or s.runs_dir
    exp_dir = runs_dir / spec.name

    if not a.score_only:
        configs = expand(spec, s)
        patents, skipped = find_patents(spec, s)
        for msg in skipped:
            print(f"skip {msg}")
        print(f"\n{len(configs)} configurations × {len(patents)} patents × "
              f"{spec.repeats} repeat(s) = {len(configs) * len(patents) * spec.repeats} runs"
              f" (documents: {spec.docs})\n")
        pricing = Pricing.load()
        total, unknown = 0.0, False
        for c in configs:
            est = [estimate_usd(c, p, spec, pricing) for p in patents]
            if any(e is None for e in est):
                unknown = True
            usd = sum(e or 0 for e in est) * spec.repeats
            total += usd
            print(f"  {c.name:<45} {c.model:<20} prompt {c.prompt_set.version:<24} "
                  f"~${usd:,.2f}")
        print(f"\nRough estimate: ~${total:,.2f}" + (" (+ unpriced models)" if unknown else "")
              + (f"; budget ${spec.budget_usd:,.2f}" if spec.budget_usd else "")
              + ". Based on the token counts of each patent's saved run; judge calls and "
              "retries are not included.")
        if a.dry_run:
            return 0
        if not patents:
            return 1
        if not a.yes:
            if not sys.stdin.isatty():
                print("Not a terminal: pass --yes to run.", file=sys.stderr)
                return 1
            if input("\nRun it? [y/N] ").strip().lower() not in ("y", "yes"):
                return 1
        outcomes = run_experiment(spec, s, runs_dir, force=a.force)
        spent = sum(o.cost_usd for o in outcomes)
        bad = [o for o in outcomes if o.status in ("failed", "budget")]
        print(f"\nSpent ${spent:,.2f} on {sum(o.status == 'done' for o in outcomes)} runs"
              + (f"; {len(bad)} did not finish" if bad else ""))

    if not exp_dir.exists():
        print(f"No runs in {exp_dir}", file=sys.stderr)
        return 1
    results = write_reports(exp_dir, Path(spec.gold_dir) if spec.gold_dir else s.gold_dir)
    if not results:
        print("Nothing to score yet.", file=sys.stderr)
        return 1
    print((exp_dir / "results.md").read_text(encoding="utf-8"))
    print(f"Written: {exp_dir / 'results.md'}, results.csv, runs.csv"
          + (", frontier.png" if (exp_dir / "frontier.png").exists() else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
