from types import SimpleNamespace as NS

import pytest

from fh_analyzer import costs
from fh_analyzer.costs import (
    BudgetExceeded,
    CostLog,
    Pricing,
    call_context,
    current_context,
    estimate_priority_usd,
    read_log,
    reprice,
)
from fh_analyzer.models import InterviewExtraction


def _usage(i=0, o=0, cw=0, cr=0):
    return NS(input_tokens=i, output_tokens=o, cache_creation_input_tokens=cw,
              cache_read_input_tokens=cr)


def test_pricing_table_and_math():
    p = Pricing.load()
    assert p.as_of and p.for_model("claude-sonnet-5")
    # Dated ids resolve to their family entry.
    assert p.for_model("claude-haiku-4-5-20251001") == p.models["claude-haiku-4-5"]
    assert p.for_model("gpt-4") is None
    m = p.models["claude-sonnet-5"]
    got = p.cost("claude-sonnet-5", input_tokens=1_000_000, output_tokens=100_000,
                 cache_write_tokens=200_000, cache_read_tokens=2_000_000)
    want = m.input + m.output * 0.1 + m.cache_write * 0.2 + m.cache_read * 2
    assert got == pytest.approx(want)
    assert p.cost("claude-sonnet-5", input_tokens=1_000_000, batch=True) == \
        pytest.approx(m.input * p.batch_discount)
    # Cache reads must be much cheaper than fresh input, writes a bit dearer.
    assert m.cache_read < m.input < m.cache_write


def test_cost_log_records_context_and_writes_jsonl(tmp_path):
    log = CostLog(tmp_path / "calls.jsonl", app_no="16000001")
    with call_context(stage="extract", doc_id="D1", doc_code="CTNF"):
        assert current_context()["stage"] == "extract"
        log.record(model="claude-sonnet-5", usage=_usage(10_000, 2_000, 0, 0), latency_ms=900)
        with call_context(stage="completeness_retry"):
            log.record(model="claude-sonnet-5", usage=_usage(0, 1_000, 0, 10_000),
                       retry=True, latency_ms=700)
    assert current_context() == {}
    log.record(model="mystery-model", usage=_usage(5, 5), ok=False, error="boom")
    log.note_cached(3)

    recs = read_log(tmp_path / "calls.jsonl")
    assert [r.stage for r in recs] == ["extract", "completeness_retry", "other"]
    assert recs[1].doc_code == "CTNF" and recs[0].app_no == "16000001"

    s = log.summary()
    assert s.calls == 3 and s.failed_calls == 1 and s.retries == 1
    assert s.cached_extractions == 3
    assert s.unpriced_models == ["mystery-model"]          # not silently $0
    assert s.total_usd == pytest.approx(sum(r.cost_usd or 0 for r in recs), abs=1e-4)
    assert set(s.by_stage) == {"extract", "completeness_retry", "other"}
    assert s.cache_hit_share == pytest.approx(10_000 / 20_005, abs=1e-3)
    assert s.label.endswith("(+ unpriced calls)")


def test_budget_guard(tmp_path):
    log = CostLog(None, max_usd=0.01)
    log.check_budget()
    log.record(model="claude-sonnet-5", usage=_usage(1_000_000))   # well over a cent
    with pytest.raises(BudgetExceeded, match="FHA_MAX_USD_PER_RUN"):
        log.check_budget()


def test_reprice_uses_tokens_not_stored_dollars(tmp_path):
    log = CostLog(tmp_path / "c.jsonl")
    log.record(model="claude-sonnet-5", usage=_usage(1_000_000))
    recs = read_log(tmp_path / "c.jsonl")
    p = Pricing.load()
    p.models["claude-sonnet-5"].input *= 2
    assert reprice(recs, p)[0].cost_usd == pytest.approx(2 * log.records[0].cost_usd)


# -------------------------------------------------------------- LLM integration

class _Msgs:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def create(self, **kw):
        block = NS(type="tool_use", id="t1", input=self.outputs.pop(0))
        return NS(content=[block], usage=_usage(1_000, 200, 5_000, 0))


def _llm(outputs, log):
    import anthropic

    from fh_analyzer.llm import AnthropicLLM, Usage

    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm.model, llm.max_tokens, llm.strict = "claude-sonnet-5", 100, True
    llm.force_tool = True
    llm._anthropic, llm.usage, llm.cost_log = anthropic, Usage(), log
    llm._client = NS(messages=_Msgs(outputs))
    return llm


GOOD = {"agreement_reached": True, "summary": "s",
        "summary_citation": {"doc_id": "D", "page": 1, "quote": "q"}}


def test_every_attempt_is_logged_including_failed_validation(tmp_path):
    log = CostLog(tmp_path / "c.jsonl")
    llm = _llm([{"summary": 123}, GOOD], log)          # first output fails validation
    with call_context(stage="extract", doc_code="EXIN"):
        llm.extract("sys", "user", InterviewExtraction)
    a, b = log.records
    assert (a.ok, a.retry, b.ok, b.retry) == (False, False, True, True)
    assert a.error.startswith("ValidationError") and a.cost_usd > 0   # failures cost money
    assert a.cache_write_tokens == 5_000 and b.doc_code == "EXIN"
    assert llm.usage.cache_write_tokens == 10_000


def test_llm_stops_at_budget(tmp_path):
    log = CostLog(None, max_usd=0.0)
    llm = _llm([GOOD], log)
    with pytest.raises(BudgetExceeded):
        llm.extract("sys", "user", InterviewExtraction)
    assert not log.records


# -------------------------------------------------------------- pipeline integration

def test_pipeline_tags_stages_and_counts_cache_hits(app_info, docs, texts, fake_llm,
                                                    settings, tmp_path):
    from fh_analyzer.pipeline import analyze

    stages = []
    real = fake_llm.extract

    def spy(system, user, schema):
        stages.append(current_context().get("stage"))
        fake_llm.cost_log.record(model="claude-sonnet-5", usage=_usage(1_000, 100))
        return real(system, user, schema)

    fake_llm.extract = spy
    fake_llm.cost_log = CostLog(None)
    a = analyze(app_info, docs, texts, fake_llm, settings, judge=fake_llm,
                extract_cache=tmp_path / "x")
    assert stages.count("extract") == 5 and "synthesis" in stages and "judge" in stages
    assert a.cost.calls == len(stages) and a.cost.by_stage["extract"] > 0

    fake_llm.cost_log = CostLog(None)
    b = analyze(app_info, docs, texts, fake_llm, settings, extract_cache=tmp_path / "x")
    assert b.cost.cached_extractions == 5                  # re-run: extractions free
    assert set(b.cost.by_stage) == {"synthesis"}


def test_priority_estimate_reflects_caching():
    one = estimate_priority_usd("claude-sonnet-5", [80_000], 12)
    many = estimate_priority_usd("claude-sonnet-5", [80_000], 48)
    two_apps = estimate_priority_usd("claude-sonnet-5", [80_000, 80_000], 12)
    assert 0 < one < many < 4 * one                        # extra batches hit the cache
    assert two_apps == pytest.approx(2 * one, abs=0.02)   # estimate is rounded to cents
    assert estimate_priority_usd("unknown-model", [1], 1) is None


def test_fh_costs_report(tmp_path, capsys, monkeypatch):
    from fh_analyzer.costs_cli import main

    log = CostLog(tmp_path / "c.jsonl", app_no="16000001")
    with call_context(stage="extract", doc_code="REM"):
        log.record(model="claude-sonnet-5", usage=_usage(20_000, 3_000), latency_ms=5_000)
    log2 = CostLog(tmp_path / "c.jsonl", app_no="14000002")
    with call_context(stage="synthesis"):
        log2.record(model="claude-haiku-4-5", usage=_usage(5_000, 500))
    assert main(["--log", str(tmp_path / "c.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "2 calls in 2 runs" in out and "## By stage" in out and "REM" in out
    assert main(["--log", str(tmp_path / "c.jsonl"), "--app", "14000002",
                 "--by", "model"]) == 0
    assert "claude-haiku-4-5" in capsys.readouterr().out
    assert costs.PRICING_FILE.exists()
