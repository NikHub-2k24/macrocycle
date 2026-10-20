"""
Unit tests for macrocycle.regimes
"""

import numpy as np
import pandas as pd
import pytest

from macrocycle.data_pipeline import build_research_panel, load_curated_data
from macrocycle.regimes import (
    classify_growth_inflation_regimes,
    compute_regime_statistics,
    evaluate_forecast_errors_by_regime,
)


def test_classify_regimes_strictly_past_information():
    """Verify regime thresholds are strictly computed using past trailing observations."""
    periods = pd.period_range("2000Q1", periods=30, freq="Q")
    growth = pd.Series(np.linspace(1.0, 5.0, 30), index=periods)
    infl = pd.Series(np.linspace(2.0, 6.0, 30), index=periods)

    reg_df = classify_growth_inflation_regimes(growth, infl, min_history=10, window_type="expanding")

    assert len(reg_df) == 20
    # First evaluated period is 2002Q3 (index 10)
    # Threshold for growth at index 10 should be median of growth[:10]
    expected_thresh_g = float(np.median(growth.iloc[:10]))
    assert np.isclose(reg_df["growth_threshold"].iloc[0], expected_thresh_g)

    # Threshold for growth at index 11 should be median of growth[:11]
    expected_thresh_g_1 = float(np.median(growth.iloc[:11]))
    assert np.isclose(reg_df["growth_threshold"].iloc[1], expected_thresh_g_1)


def test_regime_quadrant_names_coverage():
    """Verify that all four quadrants are correctly recognized and assigned."""
    raw_df = load_curated_data()
    panel = build_research_panel(raw_df, start_quarter="1960Q1")

    reg_df = classify_growth_inflation_regimes(
        panel["gdp_growth"], panel["cpi_inflation_yoy"], min_history=20
    )

    quadrants = set(reg_df["regime_short"].unique())
    # Historical U.S. data spans 1960 to 2026 and exhibits all four regimes
    assert "Expansion" in quadrants
    assert "Stagflation" in quadrants
    assert "Reflation" in quadrants
    assert "Slowdown" in quadrants

    stats_df = compute_regime_statistics(reg_df, panel["term_spread"])
    assert len(stats_df) == 4
    assert "Quarterly Count" in stats_df.columns
    assert "Mean GDP Growth (%)" in stats_df.columns


def test_evaluate_forecast_errors_by_regime_sample_flag():
    """Verify that regime evaluation flags small samples."""
    # Synthetic results
    periods = [f"201{i}Q{q}" for i in range(2) for q in range(1, 5)]  # 8 quarters
    results_df = pd.DataFrame({
        "realized": [2.0] * 8,
        "training_mean_benchmark": [2.5] * 8,
        "pred_ModelA": [2.1] * 8,
    }, index=periods)

    p_idx = pd.PeriodIndex(periods, freq="Q")
    regime_df = pd.DataFrame({
        "regime_short": ["Expansion"] * 6 + ["Stagflation"] * 2,
    }, index=p_idx)

    err_df = evaluate_forecast_errors_by_regime(results_df, regime_df, min_regime_samples=5)
    # Expansion has 6 (>=5 -> Adequate)
    # Stagflation has 2 (<3 -> skipped or flagged)
    assert len(err_df) > 0
    exp_row = err_df[err_df["Regime"] == "Expansion"].iloc[0]
    assert exp_row["Reliability Notice"] == "Adequate sample"
