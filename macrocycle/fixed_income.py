"""
Fixed-Income Scenario Engine for MacroCycle Research Lab.

Implements a transparent local duration-convexity interest rate sensitivity model:
ΔP / P ≈ -D * Δy + 0.5 * C * (Δy)^2

Converts basis points to decimal yield shifts correctly and evaluates
both linear duration and quadratic convexity components.
Provides decoupled macro-scenario sensitivity modeling with institutional caveats.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RateSensitivityResult:
    shock_bps: float
    shock_decimal: float
    duration_effect_pct: float
    convexity_effect_pct: float
    total_price_change_pct: float
    modified_duration: float
    convexity: float
    initial_yield_pct: float
    new_yield_pct: float


def compute_duration_convexity_price_change(
    modified_duration: float,
    convexity: float,
    yield_shock_bps: float,
    initial_yield_pct: float = 4.25,
) -> RateSensitivityResult:
    """
    Compute local second-order Taylor series percentage price change:
    ΔP / P = -D * Δy + 0.5 * C * (Δy)^2
    
    Parameters:
    -----------
    modified_duration : float
        Modified duration in years (D >= 0).
    convexity : float
        Bond convexity in years^2 (C >= 0).
    yield_shock_bps : float
        Parallel yield shift in basis points (1 bp = 0.0001 = 0.01%).
    initial_yield_pct : float
        Initial yield to maturity in percent (e.g. 4.25 for 4.25%).
        
    Returns:
    --------
    RateSensitivityResult with duration effect, convexity adjustment, and total change in percent.
    """
    if modified_duration < 0:
        raise ValueError("Modified duration cannot be negative.")
    if convexity < 0:
        raise ValueError("Convexity cannot be negative for standard long bond positions.")

    # Convert basis points to decimal: 100 bps = 0.0100
    delta_y = yield_shock_bps / 10000.0

    # Linear duration contribution: -D * Δy
    linear_comp = -modified_duration * delta_y

    # Second-order convexity adjustment: +0.5 * C * (Δy)^2
    convexity_comp = 0.5 * convexity * (delta_y ** 2)

    total_pct_change = (linear_comp + convexity_comp) * 100.0
    duration_pct = linear_comp * 100.0
    convexity_pct = convexity_comp * 100.0

    new_yield = initial_yield_pct + (yield_shock_bps / 100.0)

    return RateSensitivityResult(
        shock_bps=yield_shock_bps,
        shock_decimal=delta_y,
        duration_effect_pct=duration_pct,
        convexity_effect_pct=convexity_pct,
        total_price_change_pct=total_pct_change,
        modified_duration=modified_duration,
        convexity=convexity,
        initial_yield_pct=initial_yield_pct,
        new_yield_pct=new_yield,
    )


def generate_shock_grid_analysis(
    modified_duration: float,
    convexity: float,
    initial_yield_pct: float = 4.25,
    min_shock_bps: int = -300,
    max_shock_bps: int = 300,
    step_bps: int = 25,
) -> pd.DataFrame:
    """
    Evaluate duration-only vs duration+convexity price change across a range of yield shocks.
    """
    shocks = range(min_shock_bps, max_shock_bps + 1, step_bps)
    records = []

    for s in shocks:
        res = compute_duration_convexity_price_change(
            modified_duration=modified_duration,
            convexity=convexity,
            yield_shock_bps=float(s),
            initial_yield_pct=initial_yield_pct,
        )
        records.append({
            "Yield Shock (bps)": s,
            "Yield Shock (%)": s / 100.0,
            "New Yield (%)": res.new_yield_pct,
            "Duration Only (%)": res.duration_effect_pct,
            "Convexity Adjustment (%)": res.convexity_effect_pct,
            "Total Price Change (%)": res.total_price_change_pct,
        })

    return pd.DataFrame(records)


def evaluate_macro_scenarios(
    scenarios: List[Dict[str, float]],
    modified_duration: float,
    convexity: float,
    initial_yield_pct: float = 4.25,
) -> pd.DataFrame:
    """
    Evaluate fixed-income price impact across multiple user-defined macro narrative scenarios.
    
    Each scenario dictionary must contain:
    - 'name': str
    - 'hypothetical_growth': float (%)
    - 'hypothetical_inflation': float (%)
    - 'yield_shock_bps': float (bps)
    """
    rows = []
    for sc in scenarios:
        res = compute_duration_convexity_price_change(
            modified_duration=modified_duration,
            convexity=convexity,
            yield_shock_bps=sc.get("yield_shock_bps", 0.0),
            initial_yield_pct=initial_yield_pct,
        )
        rows.append({
            "Scenario Name": sc.get("name", "Custom Scenario"),
            "Hypothetical GDP Growth (%)": sc.get("hypothetical_growth", 2.0),
            "Hypothetical Inflation (%)": sc.get("hypothetical_inflation", 2.5),
            "Parallel Yield Shock (bps)": sc.get("yield_shock_bps", 0.0),
            "Resulting Yield (%)": res.new_yield_pct,
            "Duration Impact (%)": res.duration_effect_pct,
            "Convexity Adjustment (%)": res.convexity_effect_pct,
            "Total Est. Bond Price Change (%)": res.total_price_change_pct,
        })

    return pd.DataFrame(rows)


DEFAULT_PRESET_SCENARIOS = [
    {
        "name": "Soft Landing (Trend Growth, Tame Inflation)",
        "hypothetical_growth": 2.2,
        "hypothetical_inflation": 2.1,
        "yield_shock_bps": -25.0,
    },
    {
        "name": "Hawkish Tightening (Overheating & Sticky Inflation)",
        "hypothetical_growth": 3.2,
        "hypothetical_inflation": 4.5,
        "yield_shock_bps": 100.0,
    },
    {
        "name": "Stagflation Squeeze (Growth Slump, Inflation Spike)",
        "hypothetical_growth": 0.5,
        "hypothetical_inflation": 5.2,
        "yield_shock_bps": 75.0,
    },
    {
        "name": "Severe Recession / Flight to Quality",
        "hypothetical_growth": -2.5,
        "hypothetical_inflation": 1.2,
        "yield_shock_bps": -150.0,
    },
]
