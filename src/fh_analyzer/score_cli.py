"""Score analysis runs against expert gold labels.

    fh-score                                   # every gold/*.json vs data/cache/<app>/analysis.json
    fh-score --runs runs/sonnet.json runs/haiku.json --gold gold/<app>.json
    fh-score --json report.json                # also write the full report (misses, FPs)
    fh-score --experiment runs/model-sweep     # score every run of an fh-experiment grid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import get_settings
from .evalset import KIND_LABELS, load_gold, score, to_json, verdict_stats
from .pipeline import load


def _pct(x) -> str:
    return "  –  " if x is None else f"{x:5.0%}"


def _table(reports) -> str:
    kinds = sorted({k for r in reports for k in r.by_kind})
    lines = []
    for r in reports:
        lines.append(f"\n### App {r.application_number} · {r.run_model} · prompt "
                     f"{r.run_prompt_version} · {r.docs_scored} reviewed docs\n")
        lines.append("| type | gold | pred | matched | precision | recall | F1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for k in kinds + ["overall"]:
            s = r.overall if k == "overall" else r.by_kind.get(k)
            if s is None:
                continue
            name = "**overall**" if k == "overall" else KIND_LABELS.get(k, k)
            lines.append(f"| {name} | {s.gold} | {s.predicted} | {s.matched} | "
                         f"{_pct(s.precision)} | {_pct(s.recall)} | {_pct(s.f1)} |")
        if r.field_accuracy:
            lines.append("\nField accuracy on matched items: " + ", ".join(
                f"{k} {v:.0%}" for k, v in r.field_accuracy.items()))
        if r.risk_kappa is not None:
            lines.append(f"Estoppel risk agreement (weighted Cohen's kappa): {r.risk_kappa}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--gold", type=Path, nargs="*", help="gold files (default: gold/*.json)")
    ap.add_argument("--runs", type=Path, nargs="*",
                    help="analysis.json files to score (default: latest run per gold app)")
    ap.add_argument("--json", type=Path, help="write full reports as JSON")
    ap.add_argument("--experiment", type=Path,
                    help="an fh-experiment folder (runs/<name>): score all its runs and "
                         "write results.md / results.csv / frontier.png there")
    a = ap.parse_args(argv)
    s = get_settings()

    if a.experiment:
        from .experiment import write_reports

        gold_dir = a.gold[0].parent if a.gold else None   # -> the experiment's own setting
        if not write_reports(a.experiment, gold_dir):
            print(f"No scorable runs in {a.experiment}", file=sys.stderr)
            return 1
        print((a.experiment / "results.md").read_text(encoding="utf-8"))
        return 0

    gold_files = a.gold or sorted(s.gold_dir.glob("*.json"))
    if not gold_files:
        print("No gold files found. Label some documents with: "
              "python -m streamlit run label_app.py", file=sys.stderr)
        return 1
    golds = {g.application_number: g for g in map(load_gold, gold_files) if g}

    runs = a.runs or [s.cache_dir / app / "analysis.json" for app in golds]
    reports = []
    for rp in runs:
        if not Path(rp).exists():
            print(f"skip {rp}: not found", file=sys.stderr)
            continue
        an = load(rp)
        g = golds.get(an.app.application_number)
        if g is None:
            print(f"skip {rp}: no gold labels for app {an.app.application_number}",
                  file=sys.stderr)
            continue
        if not any(d.reviewed for d in g.docs.values()):
            print(f"skip {rp}: no documents marked reviewed yet", file=sys.stderr)
            continue
        reports.append(score(g, an))

    if not reports:
        return 1
    print(_table(reports))
    for g in golds.values():
        vs = verdict_stats(g)
        if vs["error_tags"]:
            print(f"\nApp {g.application_number} error tags: {vs['error_tags']}")
    if a.json:
        a.json.write_text(to_json([r.model_dump() for r in reports]), encoding="utf-8")
        print(f"\nFull report: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
