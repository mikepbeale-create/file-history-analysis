"""Command line entry point.

    fh-analyze 10123456                 # fetch from USPTO ODP and analyse
    fh-analyze --folder ./my_wrapper    # analyse a folder of already-downloaded PDFs
    fh-analyze 10123456 --judge         # also run the LLM-as-judge support check
    fh-analyze 10123456 --family        # also pull patent family + citations (EPO OPS)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import get_settings
from .llm import AnthropicLLM
from .pipeline import analyze, attach_family, fetch_from_uspto, load_local_folder, save


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("number", nargs="?", help="US patent number or application number (16/123,456)")
    ap.add_argument("--folder", type=Path, help="Analyse local PDFs instead of calling USPTO")
    ap.add_argument("--judge", action="store_true", help="Run the LLM support check")
    ap.add_argument("--family", action="store_true",
                    help="Patent family (US + foreign) + search-report citations "
                         "from EPO OPS")
    ap.add_argument("--out", type=Path, help="Output JSON path")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    if not a.number and not a.folder:
        ap.error("give a patent number or --folder")

    s = get_settings()

    def progress(msg: str, frac: float) -> None:
        print(f"[{frac:4.0%}] {msg}", file=sys.stderr)

    if a.folder:
        app, docs, texts = load_local_folder(a.folder, s, progress)
    else:
        app, docs, texts = fetch_from_uspto(a.number, s, progress)
    llm = AnthropicLLM(s.anthropic_api_key or "", s.model,
                       cost_log=s.new_cost_log(app.application_number))
    result = analyze(app, docs, texts, llm, s, progress=progress,
                     judge=llm if a.judge else None,
                     extract_cache=s.cache_dir / app.application_number / "extract")
    result.pdf_dir = str(a.folder or s.cache_dir / app.application_number / "pdf")
    if a.family:
        try:
            attach_family(result, s, progress)
        except Exception as e:
            result.errors["family"] = f"{type(e).__name__}: {e}"
    out = a.out or s.cache_dir / app.application_number / "analysis.json"
    save(result, out)

    g = result.grounding
    print(f"\n{app.patent_number or app.application_number}: {len(result.findings)} findings "
          f"from {len(result.extractions)} documents")
    if g:
        print(f"Citations verified: {g.rate_verified:.0%}   located anywhere: "
              f"{g.rate_grounded:.0%}   breakdown: {g.totals}")
    if result.family:
        fam = result.family
        xy = [c for c in fam.new_art if c.best_category in ("X", "Y")]
        print(f"Family: {len(fam.members)} members ({', '.join(fam.offices)}); "
              f"{len(fam.new_art)} cited references not of record in the US, "
              f"{len(xy)} graded X/Y")
        for c in xy[:10]:
            print(f"  [{''.join(c.categories)}] {c.display}  cited in "
                  f"{', '.join(dict.fromkeys(ci.member for ci in c.cited_in))}")
    if result.errors:
        print(f"Errors: {result.errors}")
    if result.cost:
        c = result.cost
        print(f"Cost: {c.label} over {c.calls} calls ({c.cached_extractions} cached), "
              f"cache hits {c.cache_hit_share or 0:.0%}; by stage {c.by_stage}")
    print(f"Tokens: {result.usage}")
    print(f"Saved {out}\nView it:  streamlit run app.py  (then load the JSON)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
