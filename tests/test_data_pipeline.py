"""
Unit tests for macrocycle.data_pipeline
"""

import numpy as np
import pandas as pd
import pytest

from macrocycle.data_pipeline import (
    compute_cpi_inflation_qoq_ann,
    compute_cpi_inflation_yoy,
    compute_gdp_growth,
    compute_ip_growth,
    compute_term_spread,
    build_research_panel,
    create_forecasting_dataset,
    load_curated_data,
    load_fallback_statsmodels,
    get_variable_metadata_table,
)


def test_compute_gdp_growth_exact_formula():
    """Verify GDP growth = 400 * [ln(GDP_t) - ln(GDP_{t-1})]."""
    gdp = pd.Series([100.0, 101.0, 102.5], index=pd.period_range("2020Q1", periods=3, freq="Q"))
    growth = compute_gdp_growth(gdp)
    
    assert pd.isna(growth.iloc[0])
    expected_1 = 400.0 * (np.log(101.0) - np.log(100.0))
    assert np.isclose(growth.iloc[1], expected_1, atol=1e-7)
    expected_2 = 400.0 * (np.log(102.5) - np.log(101.0))
    assert np.isclose(growth.iloc[2], expected_2, atol=1e-7)


def test_compute_gdp_growth_non_positive_raises():
    """Verify that non-positive GDP values raise ValueError."""
    with pytest.raises(ValueError, match="non-positive"):
        compute_gdp_growth(pd.Series([100.0, -5.0]))


def test_compute_cpi_inflation_yoy():
    """Verify YoY CPI inflation = 100 * [CPI_t / CPI_{t-4} - 1]."""
    cpi = pd.Series([100.0, 101.0, 102.0, 103.0, 105.0], index=pd.period_range("2020Q1", periods=5, freq="Q"))
    infl_yoy = compute_cpi_inflation_yoy(cpi)
    
    assert pd.isna(infl_yoy.iloc[0])
    assert pd.isna(infl_yoy.iloc[3])
    expected = 100.0 * (105.0 / 100.0 - 1.0)
    assert np.isclose(infl_yoy.iloc[4], expected, atol=1e-7)


def test_compute_cpi_inflation_qoq_ann():
    """Verify annualized QoQ CPI inflation = 400 * [ln(CPI_t) - ln(CPI_{t-1})]."""
    cpi = pd.Series([100.0, 101.0], index=pd.period_range("2020Q1", periods=2, freq="Q"))
    infl_qoq = compute_cpi_inflation_qoq_ann(cpi)
    expected = 400.0 * (np.log(101.0) - np.log(100.0))
    assert np.isclose(infl_qoq.iloc[1], expected, atol=1e-7)


def test_compute_term_spread():
    """Verify Term Spread = GS10 - TB3MS."""
    gs10 = pd.Series([4.5, 3.8, 5.0])
    tb3ms = pd.Series([4.0, 4.2, 3.5])
    spread = compute_term_spread(gs10, tb3ms)
    np.testing.assert_allclose(spread.values, [0.5, -0.4, 1.5])


def test_curated_data_loading_and_panel_building():
    """Test loading the bundled curated CSV and building the complete research panel."""
    df_raw = load_curated_data()
    assert len(df_raw) > 200
    assert "GDPC1" in df_raw.columns
    assert "GS10" in df_raw.columns

    panel = build_research_panel(df_raw, start_quarter="1955Q1")
    assert len(panel) >= 250
    assert not panel.isna().any().any(), "Panel must not contain any NaNs"
    assert "gdp_growth" in panel.columns
    assert "term_spread" in panel.columns


def test_feature_target_chronological_alignment():
    """Verify strictly no lookahead in feature/target alignment."""
    df_raw = load_curated_data()
    panel = build_research_panel(df_raw, start_quarter="1955Q1")
    X, y, aligned = create_forecasting_dataset(panel, target_variable="gdp_growth", horizon=1)

    assert len(X) == len(y)
    assert len(X) == len(aligned)

    # Every feature period must strictly precede its corresponding target realization period
    for idx in range(len(aligned)):
        feat_p = aligned["feature_period"].iloc[idx]
        targ_p = aligned["target_period"].iloc[idx]
        assert feat_p < targ_p, f"Lookahead violation: feature {feat_p} >= target {targ_p}"
        assert targ_p == feat_p + 1, f"Expected exactly 1 quarter horizon step: {feat_p} to {targ_p}"

        # Realized target matches panel at target period exactly
        assert aligned["target"].iloc[idx] == panel.loc[targ_p, "gdp_growth"]


def test_fallback_statsmodels_loading():
    """Verify fallback dataset loading and structure."""
    fb_df = load_fallback_statsmodels()
    assert len(fb_df) > 100
    assert "GDPC1" in fb_df.columns
    assert "CPIAUCSL" in fb_df.columns
    assert fb_df.index[0] == pd.Period("1959Q1", freq="Q")


def test_variable_metadata_table():
    """Test metadata generation table integrity."""
    df_raw = load_curated_data()
    panel = build_research_panel(df_raw, start_quarter="1960Q1")
    meta = get_variable_metadata_table(panel)
    assert len(meta) == len(panel.columns)
    assert "Variable" in meta.columns
    assert "Transformation" in meta.columns
    assert "First Quarter" in meta.columns
    assert "Last Quarter" in meta.columns
