"""Summarize the LLM cost log.

    fh-costs                          # everything in data/costs/calls.jsonl
    fh-costs --since 2026-09-01       # from a date
    fh-costs --app <application no.>  # one application
    fh-costs --by model,prompt_version
    fh-costs --reprice                # recompute $ from tokens with the current pricing.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import get_settings
from .costs import Pricing, read_log, reprice

GROUPINGS = {
    "day": ["day"], "app": ["app_no"], "stage": ["stage"], "doc": ["doc_code"],
    "model": ["model", "prompt_version"], "run": ["run_id", "app_no"],
}


def frame(records) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in records])
    if df.empty:
        return df
    df["day"] = df["ts"].str[:10]
    df["cost_usd"] = df["cost_usd"].astype(float)
    df["input_side"] = df.input_tokens + df.cache_write_tokens + df.cache_read_tokens
    return df


def table(df: pd.DataFrame, by: list[str]) -> str:
    g = df.groupby(by, dropna=False).agg(
        calls=("cost_usd", "size"), usd=("cost_usd", "sum"),
        failed=("ok", lambda s: int((~s).sum())), retries=("retry", "sum"),
        cache_read=("cache_read_tokens", "sum"), input_side=("input_side", "sum"),
        p50_s=("latency_ms", lambda s: s.median() / 1000),
    ).reset_index()
    g["cache_hit"] = (g.cache_read / g.input_side.where(g.input_side > 0)).fillna(0)
    g = g.drop(columns=["cache_read", "input_side"]).sort_values("usd", ascending=False)
    return g.to_markdown(index=False, floatfmt=".3f")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--log", type=Path, help="cost log (default from FHA_COST_LOG)")
    ap.add_argument("--since", help="YYYY-MM-DD")
    ap.add_argument("--app", help="application number")
    ap.add_argument("--by", default="day,app,stage,doc,model",
                    help=f"comma-separated groupings: {', '.join(GROUPINGS)}")
    ap.add_argument("--reprice", action="store_true",
                    help="recompute dollars from tokens with the current price table")
    a = ap.parse_args(argv)

    s = get_settings()
    records = read_log(a.log or s.cost_log)
    if not records:
        print(f"No cost records in {a.log or s.cost_log}.", file=sys.stderr)
        return 1
    pricing = Pricing.load()
    if a.reprice:
        reprice(records, pricing)
    df = frame(records)
    if a.since:
        df = df[df.day >= a.since]
    if a.app:
        df = df[df.app_no == a.app]
    if df.empty:
        print("No records match.", file=sys.stderr)
        return 1

    unpriced = sorted(df[df.cost_usd.isna()].model.unique())
    total = df.cost_usd.sum()
    runs = df.run_id.nunique()
    print(f"# LLM cost report\n\n{len(df)} calls in {runs} runs · **${total:,.2f}** · "
          f"prices as of {pricing.as_of}" + (f" · unpriced models: {unpriced}" if unpriced
                                             else ""))
    per_app = df.groupby("app_no").cost_usd.sum()
    if len(per_app):
        print(f"\nPer application: median ${per_app.median():,.2f}, "
              f"min ${per_app.min():,.2f}, max ${per_app.max():,.2f} "
              f"({len(per_app)} applications)")
    for key in [k.strip() for k in a.by.split(",") if k.strip()]:
        if key not in GROUPINGS:
            print(f"\n(unknown grouping {key!r})")
            continue
        print(f"\n## By {key}\n\n{table(df, GROUPINGS[key])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
