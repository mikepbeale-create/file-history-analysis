"""Cost accounting for every LLM call.

Design
------
* One record per API call (including failed validations and retries), appended to a
  JSONL log (default data/costs/calls.jsonl). Tokens are stored as the source of truth;
  dollars are derived from pricing.json, which carries its own as_of date, so costs can
  be recomputed if prices change.
* Context (which stage, which document) comes from `call_context(...)`, a contextvar set
  by the pipeline around each call. The LLM interface itself does not change, so fakes
  and tests are unaffected.
* A run summary (RunCost) is stored in analysis.json: total $, $ by stage / document type /
  model, cache-hit share, latency percentiles, retries, failures, and how many
  extractions were served from the disk cache for free.
* An optional budget (FHA_MAX_USD_PER_RUN) stops further calls once a run has spent it.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import math
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

PRICING_FILE = Path(__file__).with_name("pricing.json")


# ------------------------------------------------------------------ pricing

class ModelPrice(BaseModel):
    input: float           # $ per million tokens
    output: float
    cache_write: float
    cache_read: float


class Pricing(BaseModel):
    as_of: str
    source: str = ""
    batch_discount: float = 0.5
    models: dict[str, ModelPrice]

    @classmethod
    def load(cls, path: Path | None = None) -> Pricing:
        data = json.loads(Path(path or PRICING_FILE).read_text(encoding="utf-8"))
        data.pop("_comment", None)
        return cls.model_validate(data)

    def for_model(self, model: str) -> ModelPrice | None:
        """Exact id, else the longest known id that prefixes it
        (e.g. 'claude-haiku-4-5-20251001' -> 'claude-haiku-4-5')."""
        if model in self.models:
            return self.models[model]
        best = max((k for k in self.models if model.startswith(k + "-")), key=len,
                   default=None)
        return self.models[best] if best else None

    def cost(self, model: str, input_tokens: int = 0, output_tokens: int = 0,
             cache_write_tokens: int = 0, cache_read_tokens: int = 0,
             batch: bool = False) -> float | None:
        p = self.for_model(model)
        if p is None:
            return None
        usd = (input_tokens * p.input + output_tokens * p.output
               + cache_write_tokens * p.cache_write + cache_read_tokens * p.cache_read) / 1e6
        return usd * (self.batch_discount if batch else 1.0)


# ------------------------------------------------------------------ call context

_CTX: contextvars.ContextVar[dict | None] = contextvars.ContextVar("fha_call_ctx",
                                                                  default=None)


@contextlib.contextmanager
def call_context(**kw):
    """Tag every LLM call made inside this block, e.g.
    `with call_context(stage="extract", doc_id=..., doc_code="CTNF"): llm.extract(...)`."""
    token = _CTX.set({**(_CTX.get() or {}), **{k: v for k, v in kw.items() if v is not None}})
    try:
        yield
    finally:
        _CTX.reset(token)


def current_context() -> dict:
    return dict(_CTX.get() or {})


# ------------------------------------------------------------------ records + summary

class CallRecord(BaseModel):
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_id: str
    app_no: str | None = None
    stage: str = "other"
    doc_id: str | None = None
    doc_code: str | None = None
    model: str
    prompt_version: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    latency_ms: int = 0
    retry: bool = False
    ok: bool = True
    error: str | None = None
    cost_usd: float | None = None      # None = model not in the price table


class RunCost(BaseModel):
    run_id: str
    pricing_as_of: str
    total_usd: float
    calls: int
    failed_calls: int
    retries: int
    cached_extractions: int            # served from disk cache: no API call, $0
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    cache_hit_share: float | None      # cache_read / all input-side tokens
    latency_p50_ms: int | None
    latency_p95_ms: int | None
    by_stage: dict[str, float]
    by_doc_code: dict[str, float]
    by_model: dict[str, float]
    unpriced_models: list[str] = Field(default_factory=list)

    @property
    def label(self) -> str:
        s = f"${self.total_usd:,.2f}"
        return s + " (+ unpriced calls)" if self.unpriced_models else s


def _pct(values: list[int], q: float) -> int | None:
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, max(0, math.ceil(q * len(v)) - 1))]


def summarize(records: list[CallRecord], run_id: str = "", pricing_as_of: str = "",
              cached_extractions: int = 0) -> RunCost:
    by_stage: dict[str, float] = defaultdict(float)
    by_doc: dict[str, float] = defaultdict(float)
    by_model: dict[str, float] = defaultdict(float)
    for r in records:
        c = r.cost_usd or 0.0
        by_stage[r.stage] += c
        by_doc[r.doc_code or "-"] += c
        by_model[r.model] += c
    inp = sum(r.input_tokens for r in records)
    cw = sum(r.cache_write_tokens for r in records)
    cr = sum(r.cache_read_tokens for r in records)
    lat = [r.latency_ms for r in records if r.ok]
    rnd = lambda d: {k: round(v, 4) for k, v in sorted(d.items(), key=lambda kv: -kv[1])}  # noqa: E731
    return RunCost(
        run_id=run_id, pricing_as_of=pricing_as_of,
        total_usd=round(sum(r.cost_usd or 0.0 for r in records), 4),
        calls=len(records), failed_calls=sum(not r.ok for r in records),
        retries=sum(r.retry for r in records), cached_extractions=cached_extractions,
        input_tokens=inp, output_tokens=sum(r.output_tokens for r in records),
        cache_write_tokens=cw, cache_read_tokens=cr,
        cache_hit_share=round(cr / (inp + cw + cr), 3) if (inp + cw + cr) else None,
        latency_p50_ms=_pct(lat, 0.5), latency_p95_ms=_pct(lat, 0.95),
        by_stage=rnd(by_stage), by_doc_code=rnd(by_doc), by_model=rnd(by_model),
        unpriced_models=sorted({r.model for r in records if r.cost_usd is None}),
    )


class BudgetExceeded(RuntimeError):
    pass


class CostLog:
    """Collects CallRecords for one run; appends each to a JSONL file as it happens."""

    def __init__(self, path: Path | None = None, *, app_no: str | None = None,
                 max_usd: float | None = None, pricing: Pricing | None = None,
                 run_id: str | None = None) -> None:
        self.path = Path(path) if path else None
        self.app_no = app_no
        self.max_usd = max_usd
        self.pricing = pricing or Pricing.load()
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-") + \
            uuid.uuid4().hex[:6]
        self.records: list[CallRecord] = []
        self.cached_extractions = 0
        self._lock = threading.Lock()

    @property
    def total_usd(self) -> float:
        return sum(r.cost_usd or 0.0 for r in self.records)

    def check_budget(self) -> None:
        if self.max_usd is not None and self.total_usd >= self.max_usd:
            raise BudgetExceeded(
                f"Run budget of ${self.max_usd:.2f} reached (spent ${self.total_usd:.2f}). "
                "Raise FHA_MAX_USD_PER_RUN in .env to continue.")

    def record(self, *, model: str, usage=None, latency_ms: int = 0, retry: bool = False,
               ok: bool = True, error: str | None = None,
               prompt_version: str | None = None) -> CallRecord:
        ctx = current_context()
        g = lambda k: int(getattr(usage, k, 0) or 0) if usage is not None else 0  # noqa: E731
        rec = CallRecord(
            run_id=self.run_id, app_no=ctx.get("app_no", self.app_no),
            stage=ctx.get("stage", "other"), doc_id=ctx.get("doc_id"),
            doc_code=ctx.get("doc_code"), model=model,
            prompt_version=prompt_version or ctx.get("prompt_version"),
            input_tokens=g("input_tokens"), output_tokens=g("output_tokens"),
            cache_write_tokens=g("cache_creation_input_tokens"),
            cache_read_tokens=g("cache_read_input_tokens"),
            latency_ms=latency_ms, retry=retry, ok=ok, error=error,
        )
        rec.cost_usd = self.pricing.cost(model, rec.input_tokens, rec.output_tokens,
                                         rec.cache_write_tokens, rec.cache_read_tokens)
        with self._lock:
            self.records.append(rec)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(rec.model_dump_json() + "\n")
        return rec

    def note_cached(self, n: int = 1) -> None:
        with self._lock:
            self.cached_extractions += n

    def summary(self) -> RunCost:
        with self._lock:
            return summarize(list(self.records), self.run_id, self.pricing.as_of,
                             self.cached_extractions)


def read_log(path: Path) -> list[CallRecord]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(CallRecord.model_validate_json(line))
    return out


def reprice(records: list[CallRecord], pricing: Pricing) -> list[CallRecord]:
    """Recompute dollars from tokens with a (possibly newer) price table."""
    for r in records:
        r.cost_usd = pricing.cost(r.model, r.input_tokens, r.output_tokens,
                                  r.cache_write_tokens, r.cache_read_tokens)
    return records


# ------------------------------------------------------------------ estimates

CHARS_PER_TOKEN = 3.8      # English legal / technical text, roughly


def estimate_priority_usd(model: str, disclosure_chars: list[int], n_elements: int,
                          pricing: Pricing | None = None, batch: int = 12,
                          out_tokens_per_element: int = 220,
                          prompt_overhead_tokens: int = 1_500) -> float | None:
    """Estimate for the Priority step: per application, the disclosure is written to the
    prompt cache on the first batch of elements and read from it on the others."""
    pricing = pricing or Pricing.load()
    if pricing.for_model(model) is None:
        return None
    calls = max(1, math.ceil(n_elements / batch))
    total = 0.0
    for chars in disclosure_chars:
        disc = int(chars / CHARS_PER_TOKEN) + prompt_overhead_tokens
        total += pricing.cost(model, cache_write_tokens=disc,
                              cache_read_tokens=disc * (calls - 1),
                              input_tokens=300 * calls,
                              output_tokens=out_tokens_per_element * n_elements) or 0.0
    return round(total, 2)
