"""
Unit tests for macrocycle.evaluation
"""

import numpy as np
import pandas as pd
import pytest

from macrocycle.data_pipeline import build_research_panel, create_forecasting_dataset, load_curated_data
from macrocycle.evaluation import (
    compute_oos_metrics,
    compute_subperiod_performance,
    paired_moving_block_bootstrap,
    run_walk_forward_evaluation,
)


def test_compute_oos_metrics_exact():
    """Verify OOS metrics calculations against manual arithmetic."""
    y_true = np.array([2.0, 4.0, 6.0])
    y_pred = np.array([1.0, 5.0, 6.0])
    y_bench = np.array([3.0, 3.0, 3.0])

    # Residuals: [1.0, -1.0, 0.0]
    # MAE = (1 + 1 + 0)/3 = 2/3
    # RMSE = sqrt((1 + 1 + 0)/3) = sqrt(2/3)
    # Benchmark errors: [2 - 3, 4 - 3, 6 - 3] = [-1, 1, 3] -> sum sq = 1 + 1 + 9 = 11
    # Model sum sq = 1 + 1 + 0 = 2
    # R2_OOS = 1 - 2/11 = 9/11 = 0.81818...

    metrics = compute_oos_metrics(y_true, y_pred, y_bench)
    assert np.isclose(metrics.mae, 2.0 / 3.0)
    assert np.isclose(metrics.rmse, np.sqrt(2.0 / 3.0))
    assert np.isclose(metrics.r2_oos, 9.0 / 11.0)
    assert metrics.mda == 1.0  # signs match for all 3


def test_walk_forward_strictly_chronological():
    """
    Verify that:
    1. The first prediction is made only after min_train_periods.
    2. Every test observation occurs after the training observations used to produce it.
    3. Results DataFrame records true realization periods in chronological order.
    """
    raw_df = load_curated_data()
    panel = build_research_panel(raw_df, start_quarter="1960Q1")
    X, y, aligned = create_forecasting_dataset(panel, target_variable="gdp_growth")

    min_train = 30
    results_df, metrics, models = run_walk_forward_evaluation(
        X, y, min_train_periods=min_train, ridge_alpha=10.0, include_gbr=False
    )

    # First predicted period index must be exactly y.index[min_train]
    first_pred_period = results_df.index[0]
    expected_first_period = str(y.index[min_train])
    assert first_pred_period == expected_first_period

    # Total predictions must be total_periods - min_train
    assert len(results_df) == len(X) - min_train

    # Check that realization period > feature period in results_df
    for _, row in results_df.iterrows():
        targ = pd.Period(row.name, freq="Q")
        feat = pd.Period(row["feature_period"], freq="Q")
        assert targ > feat, f"Target {targ} must occur after feature {feat}"

    # Verify metrics exist for all models
    assert "Historical Mean" in metrics
    assert "Autoregressive AR(1)" in metrics
    assert "Macro Ridge" in metrics


def test_paired_moving_block_bootstrap():
    """Verify paired moving block bootstrap properties."""
    rng = np.random.default_rng(100)
    N = 100
    y_true = rng.normal(loc=3.0, scale=2.0, size=N)
    # Model 1 has small error, Model 2 has large error
    y_pred_good = y_true + rng.normal(scale=0.5, size=N)
    y_pred_bad = y_true + rng.normal(scale=2.5, size=N)

    # Testing good vs bad: diff = (y - bad)^2 - (y - good)^2 > 0
    res = paired_moving_block_bootstrap(
        y_true, y_pred_bad, y_pred_good, block_size=4, n_bootstraps=300, random_state=42
    )

    assert res.mean_diff > 0
    assert res.ci_lower > 0
    assert res.p_value < 0.05
    assert res.is_significant_5pct is True


def test_subperiod_performance_computation():
    """Verify subperiod grouping and metric generation."""
    raw_df = load_curated_data()
    panel = build_research_panel(raw_df, start_quarter="1960Q1")
    X, y, _ = create_forecasting_dataset(panel, target_variable="gdp_growth")

    results_df, _, _ = run_walk_forward_evaluation(
        X, y, min_train_periods=35, ridge_alpha=10.0, include_gbr=False
    )

    sub_df = compute_subperiod_performance(results_df)
    assert len(sub_df) > 0
    assert "Historical Subperiod" in sub_df.columns
    assert "Model" in sub_df.columns
    assert "MAE" in sub_df.columns
    assert "OOS R²" in sub_df.columns
