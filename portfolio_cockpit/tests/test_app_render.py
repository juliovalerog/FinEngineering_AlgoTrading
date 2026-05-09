from __future__ import annotations

import os

from streamlit.testing.v1 import AppTest


def test_app_renders_six_tabs_and_reporting_without_manual_repair_button() -> None:
    os.environ.pop("GEMINI_API_KEY", None)
    os.environ.pop("GOOGLE_API_KEY", None)
    app = AppTest.from_file("app.py")

    app.run(timeout=60)

    assert len(app.tabs) == 6
    assert [tab.label for tab in app.tabs] == [
        "Home / Portfolio Cockpit",
        "Data & Controls",
        "Portfolio Analysis",
        "Performance & Risk",
        "New Deal",
        "Reporting & Roadmap",
    ]
    assert not app.exception
    assert all(button.label != "Repair S&P 500 benchmark scale" for button in app.button)
    assert "Deterministic report" in "\n".join(subheader.value for subheader in app.subheader)
