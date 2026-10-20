"""
Institutional Plotly Visualizations for MacroCycle Research Lab.

Follows institutional research aesthetics: clean typography, restrained slate/navy/neutral
color palette, explicit axes, uncluttered legends, and research-focused visual cues.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Institutional Color Palette
PALETTE = {
    "navy_primary": "#0F172A",
    "navy_dark": "#1E293B",
    "blue_accent": "#0284C7",
    "slate_text": "#334155",
    "slate_subtle": "#64748B",
    "grid_line": "#E2E8F0",
    "bg_white": "#FFFFFF",
    "bg_panel": "#F8FAFC",
    "green_success": "#059669",
    "amber_warning": "#D97706",
    "red_danger": "#DC2626",
    "purple_accent": "#7C3AED",
}

MODEL_COLORS = {
    "Historical Mean": "#64748B",       # Neutral Slate
    "Autoregressive AR(1)": "#0284C7",   # Sky Blue
    "Macro Ridge": "#1E3A8A",           # Deep Navy
    "Constrained GBR": "#059669",        # Forest Green
}


def apply_institutional_layout(fig: go.Figure, title: str, subtitle: Optional[str] = None, height: int = 420) -> go.Figure:
    """Apply consistent institutional research styling to Plotly figures."""
    full_title = f"<b>{title}</b>"
    if subtitle:
        full_title += f"<br><span style='font-size:12px; color:{PALETTE['slate_subtle']}; font-weight:normal;'>{subtitle}</span>"

    fig.update_layout(
        title={
            "text": full_title,
            "x": 0.01,
            "xanchor": "left",
            "font": {"family": "Inter, Segoe UI, sans-serif", "size": 15, "color": PALETTE["navy_primary"]},
        },
        template="plotly_white",
        paper_bgcolor=PALETTE["bg_white"],
        plot_bgcolor=PALETTE["bg_white"],
        font={"family": "Inter, Segoe UI, sans-serif", "size": 12, "color": PALETTE["slate_text"]},
        height=height,
        margin={"l": 55, "r": 35, "t": 65, "b": 45},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": 11},
            "bgcolor": "rgba(255,255,255,0.8)",
        },
        xaxis={
            "showgrid": True,
            "gridcolor": PALETTE["grid_line"],
            "gridwidth": 1,
            "linecolor": PALETTE["grid_line"],
            "ticks": "outside",
        },
        yaxis={
            "showgrid": True,
            "gridcolor": PALETTE["grid_line"],
            "gridwidth": 1,
            "linecolor": PALETTE["grid_line"],
            "ticks": "outside",
        },
        hovermode="x unified",
    )
    return fig


def plot_macro_series(panel_df: pd.DataFrame, variables: List[str]) -> go.Figure:
    """Plot multi-panel or overlay of macroeconomic indicators."""
    x_dates = [p.to_timestamp() for p in panel_df.index]
    fig = go.Figure()

    colors = [PALETTE["navy_primary"], PALETTE["blue_accent"], PALETTE["green_success"], PALETTE["amber_warning"], PALETTE["red_danger"]]
    for i, var in enumerate(variables):
        if var in panel_df.columns:
            s = panel_df[var]
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=x_dates,
                    y=s.values,
                    mode="lines",
                    name=var.replace("_", " ").title(),
                    line={"width": 1.75, "color": color},
                )
            )

    fig.add_hline(y=0.0, line_dash="dash", line_color=PALETTE["slate_subtle"], line_width=1)
    apply_institutional_layout(
        fig,
        title="Historical Macroeconomic Panel Indicators",
        subtitle="Quarterly series resampled and transformed from FRED historical data",
    )
    fig.update_yaxes(title_text="Percent / Percentage Points")
    fig.update_xaxes(title_text="Observation Date")
    return fig


def plot_correlation_heatmap(panel_df: pd.DataFrame) -> go.Figure:
    """Correlation matrix heatmap for macroeconomic indicators."""
    corr = panel_df.corr()
    clean_labels = [c.replace("_", " ").title() for c in corr.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=clean_labels,
            y=clean_labels,
            colorscale="RdBu_r",
            zmid=0.0,
            zmin=-1.0,
            zmax=1.0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 11, "family": "Inter, sans-serif"},
            colorbar={"title": {"text": "Pearson r"}},
        )
    )
    apply_institutional_layout(
        fig,
        title="Macro Predictor Correlation Matrix",
        subtitle="Contemporaneous pairwise linear correlations across the quarterly panel",
        height=450,
    )
    fig.update_layout(xaxis={"tickangle": -35})
    return fig


def plot_walk_forward_forecasts(results_df: pd.DataFrame, target_name: str = "Real GDP Growth (QoQ Ann.)") -> go.Figure:
    """
    Plot realized target vs model out-of-sample walk-forward predictions.
    """
    p_indices = pd.PeriodIndex(results_df.index, freq="Q")
    x_dates = [p.to_timestamp() for p in p_indices]

    fig = go.Figure()

    # Ground truth realized line
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=results_df["realized"].values,
            mode="lines",
            name="Realized Target",
            line={"width": 2.2, "color": PALETTE["navy_primary"]},
        )
    )

    # Models
    pred_cols = [c for c in results_df.columns if c.startswith("pred_")]
    for col in pred_cols:
        model_name = col.replace("pred_", "")
        color = MODEL_COLORS.get(model_name, PALETTE["purple_accent"])
        dash_style = "dash" if "Mean" in model_name else "solid"
        fig.add_trace(
            go.Scatter(
                x=x_dates,
                y=results_df[col].values,
                mode="lines",
                name=f"Pred: {model_name}",
                line={"width": 1.6, "color": color, "dash": dash_style},
            )
        )

    fig.add_hline(y=0.0, line_dash="dot", line_color=PALETTE["slate_subtle"], line_width=1)
    apply_institutional_layout(
        fig,
        title=f"Walk-Forward Out-of-Sample Predictions vs Realized {target_name}",
        subtitle="Strict chronological walk-forward validation: models refit at each quarter using only prior data",
        height=450,
    )
    fig.update_yaxes(title_text="Annualized Growth (%)")
    fig.update_xaxes(title_text="Evaluation Realization Quarter")
    return fig


def plot_forecast_errors(results_df: pd.DataFrame) -> go.Figure:
    """Plot out-of-sample forecast errors (Realized - Forecast) over time."""
    p_indices = pd.PeriodIndex(results_df.index, freq="Q")
    x_dates = [p.to_timestamp() for p in p_indices]

    fig = go.Figure()
    err_cols = [c for c in results_df.columns if c.startswith("err_")]

    for col in err_cols:
        model_name = col.replace("err_", "")
        color = MODEL_COLORS.get(model_name, PALETTE["purple_accent"])
        fig.add_trace(
            go.Scatter(
                x=x_dates,
                y=results_df[col].values,
                mode="lines",
                name=f"Error: {model_name}",
                line={"width": 1.4, "color": color},
            )
        )

    fig.add_hline(y=0.0, line_dash="solid", line_color=PALETTE["navy_primary"], line_width=1)
    apply_institutional_layout(
        fig,
        title="Walk-Forward Forecast Errors Over Time (Realized - Predicted)",
        subtitle="Positive values denote under-prediction; negative values denote over-prediction",
        height=380,
    )
    fig.update_yaxes(title_text="Forecast Error (pp)")
    fig.update_xaxes(title_text="Realization Quarter")
    return fig


def plot_cumulative_squared_error_diff(results_df: pd.DataFrame, baseline_model: str = "Historical Mean") -> go.Figure:
    """
    Plot cumulative sum of squared error differences:
    CSSE_diff_t = sum_{i=1}^t [ (y_i - y_hat_base,i)^2 - (y_i - y_hat_model,i)^2 ]
    Rising curve denotes cumulative outperformance relative to baseline.
    """
    p_indices = pd.PeriodIndex(results_df.index, freq="Q")
    x_dates = [p.to_timestamp() for p in p_indices]

    fig = go.Figure()
    base_err_col = f"err_{baseline_model}"
    if base_err_col not in results_df.columns:
        return fig

    e_base_sq = results_df[base_err_col].values ** 2

    err_cols = [c for c in results_df.columns if c.startswith("err_") and c != base_err_col]

    for col in err_cols:
        model_name = col.replace("err_", "")
        e_model_sq = results_df[col].values ** 2
        diff = e_base_sq - e_model_sq
        cumsum_diff = np.cumsum(diff)

        color = MODEL_COLORS.get(model_name, PALETTE["purple_accent"])
        fig.add_trace(
            go.Scatter(
                x=x_dates,
                y=cumsum_diff,
                mode="lines",
                name=f"{model_name} vs {baseline_model}",
                line={"width": 1.8, "color": color},
            )
        )

    fig.add_hline(y=0.0, line_dash="dash", line_color=PALETTE["slate_subtle"], line_width=1)
    apply_institutional_layout(
        fig,
        title=f"Cumulative Sum of Squared Error Differences vs {baseline_model}",
        subtitle="Upward slope indicates periods where the challenger outperformed the baseline",
        height=380,
    )
    fig.update_yaxes(title_text="Cumulative ΔSSE (pp²)")
    fig.update_xaxes(title_text="Realization Quarter")
    return fig


def plot_bootstrap_distribution(
    boot_result,
    challenger_name: str = "Macro Ridge",
    baseline_name: str = "Historical Mean",
) -> go.Figure:
    """Plot paired moving block bootstrap confidence interval and estimate."""
    fig = go.Figure()

    # Point estimate and CI
    fig.add_vline(x=0.0, line_dash="solid", line_color=PALETTE["slate_subtle"], line_width=1.5)
    
    # Error bar representation
    fig.add_trace(
        go.Scatter(
            x=[boot_result.mean_diff],
            y=["MSE Difference"],
            mode="markers",
            name="Point Estimate",
            marker={"size": 12, "color": PALETTE["navy_primary"], "symbol": "diamond"},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [boot_result.ci_upper - boot_result.mean_diff],
                "arrayminus": [boot_result.mean_diff - boot_result.ci_lower],
                "color": PALETTE["blue_accent"],
                "thickness": 2.5,
                "width": 10,
            },
        )
    )

    subtitle = (
        f"Block Size: {boot_result.block_size} quarters | B={boot_result.n_bootstraps} resamples | "
        f"p-val: {boot_result.p_value:.3f} | 95% CI: [{boot_result.ci_lower:.2f}, {boot_result.ci_upper:.2f}]"
    )

    apply_institutional_layout(
        fig,
        title=f"Paired Block Bootstrap: {challenger_name} vs {baseline_name}",
        subtitle=subtitle,
        height=240,
    )
    fig.update_xaxes(title_text="Mean Squared Error Differential (pp²) [Positive = Challenger is Better]")
    fig.update_yaxes(showticklabels=False)
    return fig


def plot_regime_timeline(regime_df: pd.DataFrame) -> go.Figure:
    """Plot timeline of the 4 macroeconomic regimes over history."""
    p_indices = pd.PeriodIndex(regime_df.index, freq="Q")
    x_dates = [p.to_timestamp() for p in p_indices]

    regime_order = ["Expansion", "Reflation", "Stagflation", "Slowdown"]
    regime_map = {name: i for i, name in enumerate(regime_order)}
    y_vals = [regime_map.get(r, 0) for r in regime_df["regime_short"]]

    fig = go.Figure()

    from macrocycle.regimes import QUADRANT_COLORS

    for r_name in regime_order:
        mask = regime_df["regime_short"] == r_name
        sub_dates = [x_dates[i] for i, m in enumerate(mask) if m]
        sub_y = [y_vals[i] for i, m in enumerate(mask) if m]
        fig.add_trace(
            go.Scatter(
                x=sub_dates,
                y=sub_y,
                mode="markers",
                name=r_name,
                marker={"size": 8, "color": QUADRANT_COLORS.get(r_name, "#64748B"), "symbol": "square"},
            )
        )

    apply_institutional_layout(
        fig,
        title="Historical Macroeconomic Regimes Timeline (Growth / Inflation Quadrants)",
        subtitle="Regimes classified using strictly trailing expanding-window medians from prior quarters",
        height=320,
    )
    fig.update_yaxes(
        tickvals=[0, 1, 2, 3],
        ticktext=regime_order,
        showgrid=True,
    )
    fig.update_xaxes(title_text="Quarter")
    return fig


def plot_regime_scatter(regime_df: pd.DataFrame) -> go.Figure:
    """Plot 2D Growth vs Inflation scatter with trailing median quadrants."""
    from macrocycle.regimes import QUADRANT_COLORS

    fig = go.Figure()
    last_row = regime_df.iloc[-1]

    for r_name in ["Expansion", "Reflation", "Stagflation", "Slowdown"]:
        sub = regime_df[regime_df["regime_short"] == r_name]
        fig.add_trace(
            go.Scatter(
                x=sub["growth"],
                y=sub["inflation"],
                mode="markers",
                name=r_name,
                marker={
                    "size": 7,
                    "color": QUADRANT_COLORS.get(r_name, "#64748B"),
                    "opacity": 0.65,
                },
                text=[f"Quarter: {p}<br>Growth: {g:.2f}%<br>Inflation: {i:.2f}%" for p, g, i in zip(sub.index, sub["growth"], sub["inflation"])],
                hoverinfo="text",
            )
        )

    # Highlight latest available quarter
    fig.add_trace(
        go.Scatter(
            x=[last_row["growth"]],
            y=[last_row["inflation"]],
            mode="markers+text",
            name=f"Latest ({regime_df.index[-1]})",
            text=[f"  Latest: {regime_df.index[-1]}"],
            textposition="top right",
            marker={"size": 14, "color": PALETTE["navy_primary"], "symbol": "star"},
        )
    )

    # Add trailing median thresholds of latest observation as quadrant dividers
    thresh_g = last_row["growth_threshold"]
    thresh_pi = last_row["inflation_threshold"]
    fig.add_vline(x=thresh_g, line_dash="dash", line_color=PALETTE["slate_subtle"], line_width=1.2)
    fig.add_hline(y=thresh_pi, line_dash="dash", line_color=PALETTE["slate_subtle"], line_width=1.2)

    apply_institutional_layout(
        fig,
        title="Macroeconomic Quadrant Map: Real GDP Growth vs CPI Inflation",
        subtitle=f"Dashed lines indicate prevailing trailing medians (Growth: {thresh_g:.2f}%, Inflation: {thresh_pi:.2f}%)",
        height=450,
    )
    fig.update_xaxes(title_text="Real GDP Growth (QoQ Ann., %)")
    fig.update_yaxes(title_text="CPI Inflation (YoY, %)")
    return fig


def plot_term_spread_vs_subsequent_gdp(panel_df: pd.DataFrame, lag_quarters: int = 4) -> go.Figure:
    """
    Plot Treasury 10Y-3M Term Spread alongside subsequent Real GDP Growth lagged by k quarters.
    """
    spread = panel_df["term_spread"]
    gdp_future = panel_df["gdp_growth"].shift(-lag_quarters)
    valid_idx = spread.dropna().index.intersection(gdp_future.dropna().index)

    x_dates = [p.to_timestamp() for p in valid_idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=spread.loc[valid_idx].values,
            mode="lines",
            name="Term Spread (10Y - 3M Yield, pp)",
            line={"width": 1.8, "color": PALETTE["blue_accent"]},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_dates,
            y=gdp_future.loc[valid_idx].values,
            mode="lines",
            name=f"Subsequent Real GDP Growth (t+{lag_quarters} Qtrs, %)",
            line={"width": 1.6, "color": PALETTE["navy_primary"], "dash": "dot"},
        )
    )

    fig.add_hline(y=0.0, line_dash="solid", line_color=PALETTE["red_danger"], line_width=1.2)
    apply_institutional_layout(
        fig,
        title=f"Yield-Curve Term Spread vs Realized Subsequent GDP Growth (Lag = {lag_quarters} Quarters)",
        subtitle="Empirical inversion indicator: term spread dipping below zero historically precedes cyclical growth decelerations",
        height=400,
    )
    fig.update_yaxes(title_text="Percent / Spread (pp)")
    fig.update_xaxes(title_text="Observation Quarter t")
    return fig


def plot_rate_sensitivity_curve(grid_df: pd.DataFrame, user_shock_bps: float) -> go.Figure:
    """
    Plot fixed-income local duration-convexity price sensitivity curve.
    Shows linear duration tangent line vs duration + convexity curved response.
    """
    fig = go.Figure()

    # Linear duration-only line
    fig.add_trace(
        go.Scatter(
            x=grid_df["Yield Shock (bps)"],
            y=grid_df["Duration Only (%)"],
            mode="lines",
            name="Linear Duration Only (-D · Δy)",
            line={"width": 1.6, "color": PALETTE["slate_subtle"], "dash": "dash"},
        )
    )

    # Nonlinear duration + convexity curve
    fig.add_trace(
        go.Scatter(
            x=grid_df["Yield Shock (bps)"],
            y=grid_df["Total Price Change (%)"],
            mode="lines",
            name="Duration + Convexity (-D · Δy + 0.5 · C · Δy²)",
            line={"width": 2.4, "color": PALETTE["navy_primary"]},
        )
    )

    # Highlight user shock point
    user_row = grid_df.iloc[(grid_df["Yield Shock (bps)"] - user_shock_bps).abs().argsort()[:1]].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=[user_shock_bps],
            y=[user_row["Total Price Change (%)"]],
            mode="markers+text",
            name=f"Selected Shock ({user_shock_bps:+.0f} bps)",
            text=[f"  {user_row['Total Price Change (%)']:+.2f}%"],
            textposition="top center",
            marker={"size": 12, "color": PALETTE["red_danger"], "symbol": "circle"},
        )
    )

    fig.add_vline(x=0.0, line_dash="solid", line_color=PALETTE["grid_line"], line_width=1)
    fig.add_hline(y=0.0, line_dash="solid", line_color=PALETTE["grid_line"], line_width=1)

    apply_institutional_layout(
        fig,
        title="Local Fixed-Income Interest Rate Price Sensitivity Curve",
        subtitle="Second-order Taylor series approximation demonstrating positive convexity cushion",
        height=420,
    )
    fig.update_xaxes(title_text="Parallel Yield Shock Δy (Basis Points)")
    fig.update_yaxes(title_text="Estimated Price Change ΔP / P (%)")
    return fig


def plot_scenario_comparison_bar(scenario_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing price impacts across macro narrative scenarios."""
    fig = go.Figure()

    colors = [PALETTE["green_success"] if p >= 0 else PALETTE["red_danger"] for p in scenario_df["Total Est. Bond Price Change (%)"]]

    fig.add_trace(
        go.Bar(
            x=scenario_df["Scenario Name"],
            y=scenario_df["Total Est. Bond Price Change (%)"],
            marker_color=colors,
            text=[f"{v:+.2f}%" for v in scenario_df["Total Est. Bond Price Change (%)"]],
            textposition="auto",
        )
    )

    fig.add_hline(y=0.0, line_dash="solid", line_color=PALETTE["slate_subtle"], line_width=1)
    apply_institutional_layout(
        fig,
        title="Hypothetical Bond Price Impact by Macro Scenario",
        subtitle="Independent scenario-based sensitivity calculation (not an automatic model forecast)",
        height=380,
    )
    fig.update_xaxes(tickangle=-15)
    fig.update_yaxes(title_text="Estimated Bond Price Change (%)")
    return fig
