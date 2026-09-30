"""Runtime configuration, read from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    uspto_api_key: str | None
    anthropic_api_key: str | None
    model: str
    cache_dir: Path
    tesseract_cmd: str | None
    ocr_dpi: int = 300
    # Pages whose embedded text layer has fewer chars than this are treated as image-only.
    min_text_layer_chars: int = 200
    # Mean Tesseract word confidence below this marks a page "low quality".
    low_conf_threshold: float = 70.0
    # EPO Open Patent Services (patent family + search-report citations). Optional.
    epo_ops_key: str | None = None
    epo_ops_secret: str | None = None
    # Cost accounting (costs.py): every API call is appended to this JSONL log.
    cost_log: Path = Path("data/costs/calls.jsonl")
    # Optional spending cap per run (one Analyze / Priority button press), in USD.
    max_usd_per_run: float | None = None
    # Workspace: the root for everything a run produces (data/cache, data/costs, gold,
    # runs). "." is the public repo. Point FHA_WORKSPACE at a git-ignored folder such as
    # "private" to keep client or confidential matters out of the repository entirely.
    workspace: Path = Path(".")
    gold_dir: Path = Path("gold")
    runs_dir: Path = Path("runs")
    # Estoppel / disclaimer analysis. Off for now to keep the project focused; set
    # FHA_ESTOPPEL=1 to extract it again and show the Estoppel tab. Nothing was removed.
    estoppel: bool = False

    @property
    def is_private(self) -> bool:
        return self.workspace.resolve() != Path(".").resolve()

    def new_cost_log(self, app_no: str | None = None):
        from .costs import CostLog

        return CostLog(self.cost_log, app_no=app_no, max_usd=self.max_usd_per_run)


def _float(v: str | None) -> float | None:
    try:
        return float(v) if v else None
    except ValueError:
        return None


def get_settings() -> Settings:
    ws = Path(os.getenv("FHA_WORKSPACE") or ".")
    env_path = lambda var, rel: Path(os.getenv(var) or ws / rel)  # noqa: E731
    return Settings(
        uspto_api_key=os.getenv("USPTO_API_KEY") or None,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        model=os.getenv("FHA_MODEL", "claude-sonnet-5"),
        cache_dir=env_path("FHA_CACHE_DIR", "data/cache"),
        tesseract_cmd=os.getenv("TESSERACT_CMD") or None,
        epo_ops_key=os.getenv("EPO_OPS_KEY") or None,
        epo_ops_secret=os.getenv("EPO_OPS_SECRET") or None,
        cost_log=env_path("FHA_COST_LOG", "data/costs/calls.jsonl"),
        max_usd_per_run=_float(os.getenv("FHA_MAX_USD_PER_RUN")),
        workspace=ws,
        gold_dir=env_path("FHA_GOLD_DIR", "gold"),
        runs_dir=ws / "runs",
        estoppel=(os.getenv("FHA_ESTOPPEL") or "").strip().lower() in ("1", "true", "yes", "on"),
    )
