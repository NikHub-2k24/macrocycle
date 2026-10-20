"""
Unit tests for macrocycle.fixed_income
"""

import numpy as np
import pytest

from macrocycle.fixed_income import (
    compute_duration_convexity_price_change,
    evaluate_macro_scenarios,
    generate_shock_grid_analysis,
)


def test_positive_and_negative_yield_shocks():
    """Verify local duration-convexity price change for both positive and negative shocks."""
    D = 8.0
    C = 80.0
    # Shock +100 bps (+0.0100)
    res_pos = compute_duration_convexity_price_change(
        modified_duration=D, convexity=C, yield_shock_bps=100.0, initial_yield_pct=4.0
    )
    # Linear: -8 * 0.01 = -0.08 (-8.0%)
    # Convexity: +0.5 * 80 * (0.01)^2 = 40 * 0.0001 = +0.0040 (+0.40%)
    # Total: -8.0% + 0.40% = -7.60%
    assert np.isclose(res_pos.duration_effect_pct, -8.0)
    assert np.isclose(res_pos.convexity_effect_pct, 0.40)
    assert np.isclose(res_pos.total_price_change_pct, -7.60)
    assert np.isclose(res_pos.new_yield_pct, 5.0)

    # Shock -100 bps (-0.0100)
    res_neg = compute_duration_convexity_price_change(
        modified_duration=D, convexity=C, yield_shock_bps=-100.0, initial_yield_pct=4.0
    )
    # Linear: -8 * (-0.01) = +8.0%
    # Convexity: +0.40%
    # Total: +8.40%
    assert np.isclose(res_neg.duration_effect_pct, 8.0)
    assert np.isclose(res_neg.convexity_effect_pct, 0.40)
    assert np.isclose(res_neg.total_price_change_pct, 8.40)
    assert np.isclose(res_neg.new_yield_pct, 3.0)


def test_convexity_cushion_asymmetry():
    """Due to positive convexity, price gain for -Δy exceeds price loss for +Δy."""
    res_up = compute_duration_convexity_price_change(modified_duration=7.0, convexity=60.0, yield_shock_bps=150.0)
    res_down = compute_duration_convexity_price_change(modified_duration=7.0, convexity=60.0, yield_shock_bps=-150.0)

    gain = res_down.total_price_change_pct
    loss = abs(res_up.total_price_change_pct)
    assert gain > loss, "Positive convexity implies rally gain exceeds selloff loss"


def test_invalid_parameters_raise():
    """Verify negative duration or convexity raises ValueError."""
    with pytest.raises(ValueError, match="duration"):
        compute_duration_convexity_price_change(modified_duration=-5.0, convexity=50.0, yield_shock_bps=50.0)

    with pytest.raises(ValueError, match="Convexity"):
        compute_duration_convexity_price_change(modified_duration=5.0, convexity=-10.0, yield_shock_bps=50.0)


def test_shock_grid_analysis():
    """Verify grid generation across shock range."""
    grid = generate_shock_grid_analysis(modified_duration=6.5, convexity=50.0, min_shock_bps=-100, max_shock_bps=100, step_bps=50)
    assert len(grid) == 5  # -100, -50, 0, 50, 100
    zero_shock = grid[grid["Yield Shock (bps)"] == 0].iloc[0]
    assert zero_shock["Total Price Change (%)"] == 0.0


def test_evaluate_macro_scenarios():
    """Verify evaluation of custom macro scenarios."""
    scenarios = [
        {"name": "Base", "hypothetical_growth": 2.0, "hypothetical_inflation": 2.0, "yield_shock_bps": 0.0},
        {"name": "Cut", "hypothetical_growth": 1.0, "hypothetical_inflation": 1.5, "yield_shock_bps": -50.0},
    ]
    df = evaluate_macro_scenarios(scenarios, modified_duration=7.0, convexity=60.0)
    assert len(df) == 2
    assert "Total Est. Bond Price Change (%)" in df.columns
    assert df.iloc[0]["Total Est. Bond Price Change (%)"] == 0.0
    assert df.iloc[1]["Total Est. Bond Price Change (%)"] > 0.0
