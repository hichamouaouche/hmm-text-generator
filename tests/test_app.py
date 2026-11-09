"""Smoke tests of the Streamlit app with streamlit.testing (no browser needed)."""
import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def run():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    return at


def test_app_starts_with_notice():
    at = run()
    assert any("The corpus is synthetic" in w.value for w in at.warning)


def test_generate_reproduces_guide_metrics():
    at = run()
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    values = {m.label: m.value for m in at.metric}
    assert values["Valid tag pattern"] == "100.0%"
    assert values["Plausible subject-verb"] == "60.6%"


def test_tagger_handles_unknown_and_empty_input():
    at = run()
    box = [t for t in at.text_input if t.label == "Type a sentence"][0]
    box.input("the zorgs watch qwerty").run()
    assert not at.exception, at.exception
    assert any("Out-of-vocabulary" in w.value for w in at.warning)
    box = [t for t in at.text_input if t.label == "Type a sentence"][0]
    box.input("   ").run()
    assert not at.exception, at.exception
