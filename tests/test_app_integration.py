"""
Integration test for Streamlit app execution using Streamlit AppTest framework.
"""

import pytest
from streamlit.testing.v1 import AppTest


def test_streamlit_app_loads_and_runs():
    """Verify that app.py executes cleanly without uncaught exceptions."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"App threw uncaught exception: {at.exception}"
    assert len(at.tabs) == 6
    assert at.tabs[0].label == "1. Research Overview"
    assert at.tabs[1].label == "2. Data Explorer"
    assert at.tabs[2].label == "3. Forecast Evaluation"
    assert at.tabs[3].label == "4. Macro Regimes"
    assert at.tabs[4].label == "5. Rates Scenario Lab"
    assert at.tabs[5].label == "6. Research Notes"
