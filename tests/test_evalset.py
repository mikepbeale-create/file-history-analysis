from fh_analyzer.evalset import (
    GoldItem,
    cohen_kappa,
    fmt_claims,
    gold_from_analysis,
    load_gold,
    parse_claims,
    save_gold,
    score,
    verdict_stats,
)
from fh_analyzer.pipeline import analyze


def _run(app_info, docs, texts, fake_llm, settings):
    return analyze(app_info, docs, texts, fake_llm, settings)


def test_claim_ranges_round_trip():
    assert parse_claims("1-3, 5; 7–8") == [1, 2, 3, 5, 7, 8]
    assert fmt_claims([1, 2, 3, 5, 7, 8]) == "1-3, 5, 7-8"
    assert parse_claims("") == []


def test_gold_prefill_and_perfect_score(app_info, docs, texts, fake_llm, settings, tmp_path):
    a = _run(app_info, docs, texts, fake_llm, settings)
    g = gold_from_analysis(a)
    kinds = {i.kind for i in g.items}
    assert {"rejection", "amendment", "estoppel", "claim", "allowance_reason"} <= kinds
    assert "action" not in kinds and "response" not in kinds     # narrative: not labeled
    assert all(i.verdict is None for i in g.items)

    # Nothing reviewed yet -> nothing is scored.
    assert score(g, a).docs_scored == 0

    for d in g.docs.values():
        d.reviewed = True
    for i in g.items:
        i.verdict = "correct"
    r = score(g, a)
    assert r.overall.precision == 1.0 and r.overall.recall == 1.0

    save_gold(g, tmp_path / "g.json")
    assert load_gold(tmp_path / "g.json").items[0].id == g.items[0].id


def test_wrong_items_hurt_precision_added_items_hurt_recall(
        app_info, docs, texts, fake_llm, settings):
    a = _run(app_info, docs, texts, fake_llm, settings)
    g = gold_from_analysis(a)
    for d in g.docs.values():
        d.reviewed = True
    for i in g.items:
        i.verdict = "correct"
    # Expert: the model's amendment item is wrong, and it missed a 103 rejection.
    amend = next(i for i in g.items if i.kind == "amendment")
    amend.verdict, amend.error_tags = "incorrect", ["not in document (hallucinated)"]
    g.items.append(GoldItem(id=g.next_id(), origin="added", kind="rejection",
                            doc_id="OA-1", fields={"claims": [1], "basis": "103",
                                                   "references": ["Jones"]}))
    r = score(g, a)
    assert r.by_kind["amendment"].precision == 0.0
    assert r.by_kind["rejection"].recall == 0.5
    assert any(m["kind"] == "rejection" for m in r.misses)
    assert verdict_stats(g)["error_tags"] == {"not in document (hallucinated)": 1}


def test_fixed_fields_show_up_in_field_accuracy(app_info, docs, texts, fake_llm, settings):
    a = _run(app_info, docs, texts, fake_llm, settings)
    g = gold_from_analysis(a)
    for d in g.docs.values():
        d.reviewed = True
    for i in g.items:
        i.verdict = "correct"
    est = next(i for i in g.items if i.kind == "estoppel")
    est.fields = {**est.fields, "risk": "medium"}          # expert disagrees on risk
    est.verdict = "corrected"
    r = score(g, a)
    assert r.by_kind["estoppel"].recall == 1.0             # still the same item
    assert r.field_accuracy["estoppel.risk"] == 0.0
    assert r.risk_confusion["medium"] == {"high": 1}


def test_cohen_kappa():
    same = [("high", "high"), ("low", "low"), ("medium", "medium")]
    assert cohen_kappa(same, ["high", "medium", "low"]) == 1.0
    assert cohen_kappa([("high", "low")], ["high", "medium", "low"]) is None


def test_save_gold_survives_a_briefly_locked_file(tmp_path, monkeypatch):
    """Windows: os.replace raises PermissionError while antivirus/sync holds the file."""
    import os

    from fh_analyzer import evalset
    from fh_analyzer.evalset import GoldSet, load_gold, save_gold

    monkeypatch.setattr(evalset.time, "sleep", lambda s: None)
    real, calls = os.replace, {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(5, "Access is denied")
        real(src, dst)

    monkeypatch.setattr(evalset.os, "replace", flaky)
    p = tmp_path / "gold" / "1.json"
    save_gold(GoldSet(application_number="1"), p)
    assert calls["n"] == 3 and load_gold(p).application_number == "1"

    monkeypatch.setattr(evalset.os, "replace",
                        lambda s, d: (_ for _ in ()).throw(PermissionError(5, "denied")))
    save_gold(GoldSet(application_number="2"), p)        # never unlocks -> write in place
    assert load_gold(p).application_number == "2"
    assert not list(p.parent.glob("*.tmp"))
