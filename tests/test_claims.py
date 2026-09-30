from datetime import date

from fh_analyzer.claims import ClaimSnapshot, build_histories, diff_claim


def test_diff_claim_finds_added_limitation():
    added, removed, html, ratio = diff_claim(
        "a housing; and a spring disposed within the housing.",
        "a housing; and a coil spring disposed within the housing, wherein it is preloaded.",
    )
    assert "coil" in added
    assert any("preloaded" in a for a in added)
    assert '<ins class="add">coil</ins>' in html
    assert 0.5 < ratio < 1


def _snap(doc, d, status, text, n=1):
    return ClaimSnapshot(number=n, doc_id=doc, date=d, status=status, text=text)


def test_histories_track_amend_cancel_and_noise():
    h = build_histories([
        _snap("A", date(2020, 1, 1), "original", "a spring in a housing."),
        _snap("B", date(2020, 6, 1), "currently amended", "a coil spring in a housing."),
        # Identical claim re-listed: no change recorded.
        _snap("C", date(2020, 9, 1), "previously presented", "a coil spring in a housing."),
        # OCR noise on an unchanged claim: recorded but flagged.
        _snap("D", date(2021, 1, 1), "previously presented", "a coil spring in a hous1ng."),
        _snap("X", date(2020, 1, 1), "original", "a method.", n=2),
        _snap("Y", date(2020, 6, 1), "canceled", "", n=2),
    ])
    c1, c2 = h
    assert [c.to_doc for c in c1.changes] == ["A", "B", "D"]
    assert c1.changes[1].added == ["coil"] and not c1.changes[1].noise_suspect
    assert c1.changes[2].noise_suspect
    assert c2.canceled and c2.final_text == "a method."
