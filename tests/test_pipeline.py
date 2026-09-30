from fh_analyzer.pipeline import analyze, load, save


def test_end_to_end_with_fake_llm(app_info, docs, texts, fake_llm, settings, tmp_path):
    a = analyze(app_info, docs, texts, fake_llm, settings, judge=fake_llm, workers=2)

    assert not a.errors
    kinds = [f.kind for f in a.findings]
    assert kinds.count("claim") == 2 and "estoppel" in kinds and "allowance_reason" in kinds

    # Finding ids are assigned in chronological document order.
    assert a.findings[0].doc_id == "CLM-A"

    # Claim evolution computed in code from the two claim listings.
    (h,) = a.claim_histories
    assert "coil" in h.changes[1].added

    g = a.grounding
    st = {f.kind: g.checks[f.id].status for f in a.findings}
    assert st["rejection"] == "wrong_page"      # scripted mistake is caught
    assert st["amendment"] == "unverified"      # scripted fabrication is caught
    assert st["estoppel"] == "verified"
    assert g.totals["verified"] >= 6

    # Synthesis traceability: F999 does not exist.
    assert g.synthesis["dangling_ids"] == ["F999"]
    assert g.synthesis["points_without_valid_support"] == 1

    # The judge only sees findings whose text was located (not the fabricated one).
    amend = next(f for f in a.findings if f.kind == "amendment")
    assert g.checks[amend.id].support is None
    assert g.support_totals["supported"] == len(a.findings) - 1

    # Round-trips through JSON for the UI.
    p = save(a, tmp_path / "a.json")
    assert load(p).grounding.totals == g.totals


def test_extraction_cache(app_info, docs, texts, fake_llm, settings, tmp_path):
    analyze(app_info, docs, texts, fake_llm, settings, extract_cache=tmp_path / "x")
    first = fake_llm.usage.calls
    analyze(app_info, docs, texts, fake_llm, settings, extract_cache=tmp_path / "x")
    # Only the synthesis call repeats; per-document extractions come from cache.
    assert fake_llm.usage.calls - first == 1


def test_estoppel_switched_off(app_info, docs, texts, fake_llm, settings, tmp_path):
    """FHA_ESTOPPEL off: filings are extracted without the estoppel list (smaller schema and
    prompt), nothing is flagged, and scoring leaves estoppel out instead of counting misses."""
    import dataclasses

    from fh_analyzer.evalset import gold_from_analysis, score
    from fh_analyzer.prompts import RESPONSE_CORE

    full = analyze(app_info, docs, texts, fake_llm, settings)
    systems = []
    real = fake_llm.extract

    def spy(system, user, schema):
        systems.append((schema.__name__, system))
        return real(system, user, schema)

    fake_llm.extract = spy
    off = analyze(app_info, docs, texts, fake_llm,
                  dataclasses.replace(settings, estoppel=False), extract_cache=tmp_path / "x")
    assert ("ResponseCoreExtraction", RESPONSE_CORE) in systems
    assert "ResponseExtraction" not in {n for n, _ in systems}
    assert not [f for f in off.findings if f.kind == "estoppel"]
    assert [f for f in off.findings if f.kind == "amendment"]        # amendments still there
    assert off.features == {"estoppel": False} and full.features == {"estoppel": True}
    assert list((tmp_path / "x").glob("REM-1.*.noestoppel.json"))    # separate cache entry

    g = gold_from_analysis(full)
    for d in g.docs.values():
        d.reviewed = True
    for it in g.items:
        it.verdict = "correct"
    assert "estoppel" in score(g, full).by_kind
    r = score(g, off)
    assert "estoppel" not in r.by_kind and r.overall.recall == 1.0
