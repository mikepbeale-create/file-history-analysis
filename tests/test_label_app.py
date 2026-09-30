"""Smoke test for the labeling UI: start labeling, check items, mark a doc reviewed."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from fh_analyzer.pipeline import analyze, save

APP = str(Path(__file__).parents[1] / "label_app.py")


@pytest.fixture
def workdir(tmp_path, monkeypatch, app_info, docs, texts, fake_llm, settings):
    a = analyze(app_info, docs, texts, fake_llm, settings)
    save(a, tmp_path / "data" / "cache" / app_info.application_number / "analysis.json")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FHA_CACHE_DIR", "data/cache")
    return tmp_path


def _btn(at, prefix):
    return next(b for b in at.button if b.label.startswith(prefix))


def test_label_flow(workdir):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    _btn(at.sidebar, "Start labeling").click().run()
    gold_file = workdir / "gold" / "99999999.json"
    assert gold_file.exists()

    # Go to the remarks (2 items), confirm one item, reject the other, mark reviewed.
    while not at.subheader or at.subheader[0].value != "Remarks":
        _btn(at, "▶").click().run()
    _btn(at, "✅ Correct").click().run()
    _btn(at, "❌ Wrong").click().run()
    at.multiselect[0].set_value(["duplicate"]).run()
    _btn(at, "Save").click().run()
    _btn(at, "✔ Mark document reviewed").click().run()
    assert not at.exception

    g = json.loads(gold_file.read_text())
    rem = [i for i in g["items"] if i["doc_id"] == "REM-1"]
    assert sorted(i["verdict"] for i in rem) == ["correct", "incorrect"]
    assert g["docs"]["REM-1"]["reviewed"] is True
