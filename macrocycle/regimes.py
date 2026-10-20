"""
Macroeconomic Regime Analysis for MacroCycle Research Lab.

Implements descriptive growth/inflation quadrant classification using strictly trailing
historical benchmarks (expanding or rolling median from past observations only),
regime-conditional statistics, term spread lead-lag analysis, and regime-conditioned
forecast error evaluation with sample size validation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


QUADRANT_LABELS = {
    (True, False): "Expansion (High Growth, Low Inflation)",
    (True, True): "Reflation (High Growth, High Inflation)",
    (False, True): "Stagflation (Low Growth, High Inflation)",
    (False, False): "Slowdown (Low Growth, Low Inflation)",
}

QUADRANT_SHORT_NAMES = {
    (True, False): "Expansion",
    (True, True): "Reflation",
    (False, True): "Stagflation",
    (False, False): "Slowdown",
}

QUADRANT_COLORS = {
    "Expansion": "#10B981",    # Emerald green
    "Reflation": "#F59E0B",    # Amber / warm orange
    "Stagflation": "#EF4444",  # Crimson / alert red
    "Slowdown": "#64748B",     # Slate / cool gray
}


@dataclass
class RegimeSummary:
    name: str
    count: int
    pct_total: float
    mean_growth: float
    std_growth: float
    mean_inflation: float
    std_inflation: float
    mean_spread: float
    avg_duration_quarters: float


def classify_growth_inflation_regimes(
    growth_series: pd.Series,
    inflation_series: pd.Series,
    min_history: int = 20,
    window_type: str = "expanding",
    rolling_window: int = 20,
) -> pd.DataFrame:
    """
    Classify growth and inflation into 4 descriptive regimes using trailing historical
    medians calculated strictly from past observations only (zero lookahead).
    
    Parameters:
    -----------
    growth_series : pd.Series
        Quarterly real GDP growth series.
    inflation_series : pd.Series
        Quarterly inflation series (e.g. CPI YoY or QoQ).
    min_history : int
        Minimum historical quarters required before regime classification starts.
    window_type : str
        'expanding' (default) or 'rolling'.
    rolling_window : int
        Window size if window_type == 'rolling'.
        
    Returns:
    --------
    regime_df : pd.DataFrame
        DataFrame with growth, inflation, trailing thresholds, quadrant name, and short name.
    """
    common_idx = growth_series.dropna().index.intersection(inflation_series.dropna().index)
    g = growth_series.loc[common_idx]
    pi = inflation_series.loc[common_idx]
    n = len(common_idx)

    if n < min_history + 5:
        raise ValueError(f"Insufficient history ({n} quarters) for regime analysis with min_history={min_history}.")

    records = []
    for i in range(min_history, n):
        idx_t = common_idx[i]
        
        # Trailing past window strictly up to quarter i
        if window_type == "rolling":
            start_i = max(0, i - rolling_window)
            hist_g = g.iloc[start_i:i]
            hist_pi = pi.iloc[start_i:i]
        else:  # expanding
            hist_g = g.iloc[:i]
            hist_pi = pi.iloc[:i]

        thresh_g = float(np.median(hist_g))
        thresh_pi = float(np.median(hist_pi))

        current_g = float(g.iloc[i])
        current_pi = float(pi.iloc[i])

        high_growth = current_g >= thresh_g
        high_infl = current_pi >= thresh_pi

        full_label = QUADRANT_LABELS.get((high_growth, high_infl), "Unknown")
        short_label = QUADRANT_SHORT_NAMES.get((high_growth, high_infl), "Unknown")

        records.append({
            "period": idx_t,
            "growth": current_g,
            "inflation": current_pi,
            "growth_threshold": thresh_g,
            "inflation_threshold": thresh_pi,
            "high_growth": high_growth,
            "high_inflation": high_infl,
            "regime": full_label,
            "regime_short": short_label,
        })

    regime_df = pd.DataFrame(records).set_index("period")
    return regime_df


def compute_regime_statistics(
    regime_df: pd.DataFrame,
    spread_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute sample counts, percentage, averages, volatility, and average duration
    for each macro regime.
    """
    df = regime_df.copy()
    if spread_series is not None:
        df["spread"] = spread_series.loc[df.index]

    total_quarters = len(df)
    summaries = []

    # Calculate regime run durations
    df["regime_shift"] = (df["regime_short"] != df["regime_short"].shift(1)).cumsum()
    durations = df.groupby(["regime_short", "regime_shift"]).size()

    for r_name in ["Expansion", "Reflation", "Stagflation", "Slowdown"]:
        sub = df[df["regime_short"] == r_name]
        cnt = len(sub)
        if cnt == 0:
            continue
        pct = (cnt / total_quarters) * 100.0
        r_durations = durations.get(r_name, pd.Series(dtype=float))
        avg_dur = float(r_durations.mean()) if len(r_durations) > 0 else float(cnt)
        mean_spread = float(sub["spread"].mean()) if "spread" in sub.columns else np.nan

        summaries.append({
            "Macro Regime": r_name,
            "Full Definition": [k for k, v in QUADRANT_SHORT_NAMES.items() if v == r_name][0],
            "Quarterly Count": cnt,
            "Frequency (%)": pct,
            "Mean GDP Growth (%)": float(sub["growth"].mean()),
            "GDP Growth Volatility (%)": float(sub["growth"].std()),
            "Mean Inflation (%)": float(sub["inflation"].mean()),
            "Inflation Volatility (%)": float(sub["inflation"].std()),
            "Mean 10Y-3M Spread (pp)": mean_spread,
            "Avg Spell Duration (Qtrs)": avg_dur,
        })

    return pd.DataFrame(summaries)


def evaluate_forecast_errors_by_regime(
    results_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    min_regime_samples: int = 10,
) -> pd.DataFrame:
    """
    Break down model out-of-sample forecast errors across historical regimes.
    Flags regimes with sample size below min_regime_samples.
    """
    # Align by period
    common_idx = results_df.index.intersection(regime_df.index.astype(str))
    if len(common_idx) == 0:
        # Check if index conversion needed
        regime_idx_str = [str(p) for p in regime_df.index]
        common_idx = results_df.index.intersection(regime_idx_str)

    eval_df = results_df.loc[common_idx].copy()
    eval_df["regime_short"] = [regime_df.loc[pd.Period(p, freq="Q"), "regime_short"] for p in common_idx]

    pred_cols = [c for c in eval_df.columns if c.startswith("pred_")]
    model_names = [c.replace("pred_", "") for c in pred_cols]

    records = []
    y_true_all = eval_df["realized"].values

    for r_name in ["Expansion", "Reflation", "Stagflation", "Slowdown"]:
        sub = eval_df[eval_df["regime_short"] == r_name]
        n_obs = len(sub)
        is_reliable = n_obs >= min_regime_samples

        if n_obs < 3:
            continue

        y_true = sub["realized"].values
        y_bench = sub["training_mean_benchmark"].values

        for m_name in model_names:
            y_pred = sub[f"pred_{m_name}"].values
            residuals = y_true - y_pred
            mae = float(np.mean(np.abs(residuals)))
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            bias = float(np.mean(residuals))

            records.append({
                "Regime": r_name,
                "Model": m_name,
                "Observations": n_obs,
                "Reliability Notice": "Adequate sample" if is_reliable else f"Limited sample (<{min_regime_samples} qtrs)",
                "MAE": mae,
                "RMSE": rmse,
                "Mean Bias": bias,
            })

    return pd.DataFrame(records)
