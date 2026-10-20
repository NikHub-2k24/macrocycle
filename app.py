"""
MacroCycle: Growth, Inflation and Yield-Curve Research Lab
Main Streamlit Application.
"""

import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# Add current workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from macrocycle.data_pipeline import (
    VARIABLE_SPECS,
    build_research_panel,
    create_forecasting_dataset,
    get_variable_metadata_table,
    load_curated_data,
    load_fallback_statsmodels,
    load_manifest,
)
from macrocycle.evaluation import (
    compute_subperiod_performance,
    paired_moving_block_bootstrap,
    run_walk_forward_evaluation,
)
from macrocycle.fixed_income import (
    DEFAULT_PRESET_SCENARIOS,
    compute_duration_convexity_price_change,
    evaluate_macro_scenarios,
    generate_shock_grid_analysis,
)
from macrocycle.plotting import (
    PALETTE,
    plot_bootstrap_distribution,
    plot_correlation_heatmap,
    plot_cumulative_squared_error_diff,
    plot_forecast_errors,
    plot_macro_series,
    plot_rate_sensitivity_curve,
    plot_regime_scatter,
    plot_regime_timeline,
    plot_scenario_comparison_bar,
    plot_term_spread_vs_subsequent_gdp,
    plot_walk_forward_forecasts,
)
from macrocycle.regimes import (
    classify_growth_inflation_regimes,
    compute_regime_statistics,
    evaluate_forecast_errors_by_regime,
)

# Page configuration
st.set_page_config(
    page_title="MacroCycle: Macro & Yield-Curve Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Institutional styling CSS
st.markdown(
    """
    <style>
    /* Institutional typography and clean layout */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1300px;
    }
    h1, h2, h3, h4 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0F172A;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #CBD5E1;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 500;
        color: #475569;
        font-size: 0.95rem;
    }
    .stTabs [aria-selected="true"] {
        color: #0F172A !important;
        border-bottom-color: #0F172A !important;
        font-weight: 600 !important;
    }
    .research-callout {
        background-color: #F8FAFC;
        border-left: 4px solid #0F172A;
        padding: 14px 18px;
        border-radius: 4px;
        margin-bottom: 20px;
        font-size: 0.92rem;
        line-height: 1.5;
        color: #334155;
    }
    .warning-callout {
        background-color: #FFFBEB;
        border-left: 4px solid #D97706;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 16px;
        font-size: 0.88rem;
        color: #92400E;
    }
    .metric-badge-container {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-bottom: 15px;
    }
    .metric-badge {
        background-color: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 10px 14px;
        min-width: 150px;
    }
    .metric-badge-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #64748B;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-badge-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_and_prep_data(use_fallback: bool = False) -> Tuple[pd.DataFrame, Dict]:
    manifest = load_manifest()
    if use_fallback:
        raw_df = load_fallback_statsmodels()
        panel = build_research_panel(raw_df, start_quarter="1960Q1")
    else:
        raw_df = load_curated_data()
        panel = build_research_panel(raw_df, start_quarter="1954Q3")
    return panel, manifest


@st.cache_data
def get_cached_walk_forward_results(
    panel_df: pd.DataFrame,
    target_var: str,
    min_train: int,
    ridge_alpha: float,
    include_gbr: bool,
    random_state: int,
):
    X, y, aligned = create_forecasting_dataset(panel_df, target_variable=target_var, horizon=1)
    results_df, metrics, models = run_walk_forward_evaluation(
        X,
        y,
        min_train_periods=min_train,
        ridge_alpha=ridge_alpha,
        include_gbr=include_gbr,
        random_state=random_state,
    )
    return results_df, metrics, models, aligned


# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("### **MacroCycle Controls**")
st.sidebar.markdown(
    "<span style='font-size:12px; color:#64748B;'>Empirical Macroeconomic Research Panel</span>",
    unsafe_allow_html=True,
)

data_choice = st.sidebar.radio(
    "Data Source",
    options=["Historical FRED Curated (1954–2026)", "Statsmodels Macrodata Fallback (1959–2009)"],
    index=0,
    help="Select the bundled offline dataset for analysis.",
)
use_fallback = "Statsmodels" in data_choice

panel, manifest = load_and_prep_data(use_fallback=use_fallback)

st.sidebar.markdown("---")
st.sidebar.markdown("#### **Forecast Configuration**")

target_var = st.sidebar.selectbox(
    "Forecast Target (t+1)",
    options=["gdp_growth", "cpi_inflation_qoq_ann"],
    format_func=lambda x: "Real GDP Growth (QoQ Ann.)" if x == "gdp_growth" else "CPI Inflation (QoQ Ann.)",
    index=0,
    help="Target macroeconomic indicator to predict one quarter ahead using features at t.",
)

max_train_allowed = min(80, len(panel) - 30)
min_train_periods = st.sidebar.slider(
    "Initial Training Window (Quarters)",
    min_value=25,
    max_value=max_train_allowed,
    value=40,
    step=5,
    help="Number of quarters in the initial estimation window before walk-forward testing begins.",
)

ridge_alpha = st.sidebar.slider(
    "Ridge Regularization (α)",
    min_value=0.1,
    max_value=100.0,
    value=10.0,
    step=1.0,
    help="L2 regularization penalty for the multivariate macro Ridge model.",
)

include_gbr = st.sidebar.checkbox(
    "Include Constrained GBR Challenger",
    value=True,
    help="Include constrained Gradient Boosting Regressor with conservative tree depth.",
)

boot_resamples = st.sidebar.selectbox(
    "Bootstrap Resamples (B)",
    options=[250, 500, 1000],
    index=1,
    help="Number of moving block bootstrap iterations for loss differential hypothesis testing.",
)

RANDOM_SEED = 42

st.sidebar.markdown("---")
st.sidebar.markdown("#### **Sample Summary**")
st.sidebar.markdown(f"**Total Quarters:** {len(panel)}")
st.sidebar.markdown(f"**Sample Period:** {panel.index.min()} to {panel.index.max()}")
st.sidebar.markdown(f"**OOS Test Quarters:** {len(panel) - 1 - min_train_periods}")
st.sidebar.markdown(f"**Deterministic Seed:** `{RANDOM_SEED}`")

# --- HEADER SECTION ---
st.title("MacroCycle: Growth, Inflation and Yield-Curve Lab")
st.markdown(
    "**An Empirical Research Platform for Out-of-Sample Macroeconomic Forecasting, "
    "Descriptive Regime Modeling, and Decoupled Fixed-Income Sensitivity.**"
)

# Institutional Data Quality Notice
st.markdown(
    """
    <div class="warning-callout">
        <b>Data Quality & Retrospective Backtest Notice:</b> This application analyzes historical series 
        from the Federal Reserve Bank of St. Louis (FRED). The dataset contains the <i>latest revised historical values</i> 
        rather than point-in-time real-time vintages. While strict chronological walk-forward alignment prevents temporal lookahead, 
        revised data inherently incorporates historical benchmark and seasonal revisions not available to contemporary forecasters. 
        All models are evaluated on genuine out-of-sample metrics. This tool is designed for macroeconomic research; it is not a live 
        nowcasting service, trading strategy, or investment advice. See <b>Research Notes</b> for ALFRED vintage documentation.
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs
tab_overview, tab_data, tab_forecast, tab_regimes, tab_rates, tab_notes = st.tabs([
    "1. Research Overview",
    "2. Data Explorer",
    "3. Forecast Evaluation",
    "4. Macro Regimes",
    "5. Rates Scenario Lab",
    "6. Research Notes",
])

# ==============================================================================
# TAB 1: RESEARCH OVERVIEW
# ==============================================================================
with tab_overview:
    st.subheader("Core Research Problem")
    st.markdown(
        """
        <div class="research-callout">
            <b>Central Question:</b> Can multivariate macroeconomic predictors (headline inflation, labor market slack, 
            industrial output, the federal funds rate, and the 10-year minus 3-month Treasury term spread) generate 
            genuine out-of-sample forecasting gains over simple historical-mean and univariate autoregressive baselines 
            for U.S. real GDP growth?
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown("#### **Hypotheses & Falsification**")
        st.markdown(
            """
            - **Null Hypothesis ($H_0$):** Multivariate macroeconomic models do not outperform the prevailing expanding 
              historical mean in out-of-sample real GDP growth forecasting:
              $$\\mathbb{E}[L(y_{t+1}, \\hat{y}_{t+1}^{\\text{Model}})] \\ge \\mathbb{E}[L(y_{t+1}, \\hat{y}_{t+1}^{\\text{Hist Mean}})]$$
            - **Alternative Hypothesis ($H_1$):** Lagged macroeconomic predictors (especially the yield-curve term spread and monetary stance) 
              provide statistically detectable predictive power:
              $$\\mathbb{E}[L(y_{t+1}, \\hat{y}_{t+1}^{\\text{Model}})] < \\mathbb{E}[L(y_{t+1}, \\hat{y}_{t+1}^{\\text{Hist Mean}})]$$
            - **Falsification Standard:** $H_1$ is falsified if the 95% paired moving block bootstrap confidence interval for the 
              mean squared error differential spans zero or is strictly negative ($p \\ge 0.05$).
            """
        )

    with col_h2:
        st.markdown("#### **Methodological Framework**")
        st.markdown(
            """
            1. **Expanding-Window Walk-Forward Protocol:** Zero future lookahead. Models are refit at each quarter using 
               strictly prior observations. Feature scaling is fit only on the training slice.
            2. **Honest Baseline Competition:** Every challenger model is evaluated against the expanding historical mean 
               and an AR(1) baseline. If complex models lose to the mean, that outcome is reported candidly.
            3. **Descriptive Macro Regimes:** Growth and inflation quadrants are classified using strictly trailing 
               historical medians calculated from past observations only.
            4. **Decoupled Rates Sensitivity:** An independent local duration-convexity calculator models interest rate 
               sensitivity without asserting an unverified causal link from GDP forecasts to bond returns.
            """
        )

    st.markdown("---")
    st.markdown("#### **Research Panel Scope**")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-badge">
                <div class="metric-badge-label">Sample Horizon</div>
                <div class="metric-badge-value">{panel.index.min()} – {panel.index.max()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="metric-badge">
                <div class="metric-badge-label">Available Quarters</div>
                <div class="metric-badge-value">{len(panel)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            f"""
            <div class="metric-badge">
                <div class="metric-badge-label">Initial Training Quarters</div>
                <div class="metric-badge-value">{min_train_periods}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col4:
        st.markdown(
            f"""
            <div class="metric-badge">
                <div class="metric-badge-label">Out-of-Sample Quarters</div>
                <div class="metric-badge-value">{len(panel) - 1 - min_train_periods}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==============================================================================
# TAB 2: DATA EXPLORER
# ==============================================================================
with tab_data:
    st.subheader("Quarterly Research Panel & Variable Metadata")
    st.markdown(
        "Review data coverage, definitions, units, transformations, and empirical correlations "
        "across the standardized macroeconomic panel."
    )

    meta_df = get_variable_metadata_table(panel)
    st.dataframe(
        meta_df.style.format({
            "Mean": "{:.2f}",
            "Std Dev": "{:.2f}",
            "Min": "{:.2f}",
            "Max": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    col_plot, col_corr = st.columns([3, 2])

    with col_plot:
        st.markdown("#### **Historical Time Series Explorer**")
        selected_vars = st.multiselect(
            "Select series to plot:",
            options=list(panel.columns),
            default=["gdp_growth", "cpi_inflation_yoy", "term_spread"],
            format_func=lambda x: VARIABLE_SPECS.get(x, None).name if x in VARIABLE_SPECS else x,
        )
        if selected_vars:
            fig_macro = plot_macro_series(panel, selected_vars)
            st.plotly_chart(fig_macro, use_container_width=True)
        else:
            st.info("Select at least one variable to visualize.")

    with col_corr:
        st.markdown("#### **Macro Predictor Correlation Matrix**")
        fig_corr = plot_correlation_heatmap(panel)
        st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("Inspect Raw vs Transformed Data Samples"):
        st.markdown("##### Transformed Quarterly Panel (First 8 Quarters)")
        st.dataframe(panel.head(8), use_container_width=True)
        st.markdown("##### Transformed Quarterly Panel (Latest 8 Quarters)")
        st.dataframe(panel.tail(8), use_container_width=True)

# ==============================================================================
# TAB 3: FORECAST EVALUATION
# ==============================================================================
with tab_forecast:
    target_display = "Real GDP Growth" if target_var == "gdp_growth" else "CPI Inflation"
    st.subheader(f"Expanding-Window Walk-Forward Evaluation: {target_display}")
    st.markdown(
        f"Strict chronological out-of-sample evaluation across **{len(panel) - 1 - min_train_periods} quarters**. "
        "At each forecast step, models are trained only on prior data and tested strictly out-of-sample."
    )

    # Compute walk-forward results
    results_df, metrics_by_model, models, aligned = get_cached_walk_forward_results(
        panel,
        target_var=target_var,
        min_train=min_train_periods,
        ridge_alpha=ridge_alpha,
        include_gbr=include_gbr,
        random_state=RANDOM_SEED,
    )

    # Summary table
    metrics_records = []
    for m_name, m in metrics_by_model.items():
        metrics_records.append({
            "Model Name": m_name,
            "MAE": m.mae,
            "RMSE": m.rmse,
            "OOS R² vs Mean": m.r2_oos,
            "Directional Acc (%)": m.mda * 100.0,
            "Mean Bias": m.mean_bias,
            "Observations": int(m.observations),
        })
    metrics_summary_df = pd.DataFrame(metrics_records)

    st.markdown("#### **Out-of-Sample Performance Comparison**")
    st.dataframe(
        metrics_summary_df.style.format({
            "MAE": "{:.3f}",
            "RMSE": "{:.3f}",
            "OOS R² vs Mean": "{:.4f}",
            "Directional Acc (%)": "{:.1f}%",
            "Mean Bias": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Honest institutional findings interpretation callout
    best_rmse_model = min(metrics_by_model.items(), key=lambda x: x[1].rmse)
    ridge_r2 = metrics_by_model.get("Macro Ridge", None)
    ridge_r2_val = ridge_r2.r2_oos if ridge_r2 else 0.0

    if ridge_r2_val <= 0.0:
        st.markdown(
            f"""
            <div class="research-callout">
                <b>Empirical Finding:</b> The multivariate Macro Ridge model yields an out-of-sample $R^2$ of 
                <b>{ridge_r2_val:.4f}</b> relative to the training-mean benchmark. 
                Consistent with macroeconomic forecasting literature (e.g. Stock & Watson 2003, Faust & Wright 2013), 
                incorporating multiple macro predictors does not automatically beat simple historical-mean and 
                autoregressive baselines for real output growth. Structural breaks, parameter instability, and 
                high noise-to-signal ratios limit the out-of-sample gains of complex specifications.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="research-callout">
                <b>Empirical Finding:</b> The multivariate Macro Ridge model achieves a positive out-of-sample $R^2$ of 
                <b>{ridge_r2_val:.4f}</b> relative to the training-mean benchmark. Inspect the block bootstrap test below 
                to determine whether this improvement is statistically significant.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Visualizations
    fig_forecast = plot_walk_forward_forecasts(results_df, target_name=target_display)
    st.plotly_chart(fig_forecast, use_container_width=True)

    col_err, col_csse = st.columns(2)
    with col_err:
        fig_err = plot_forecast_errors(results_df)
        st.plotly_chart(fig_err, use_container_width=True)
    with col_csse:
        fig_csse = plot_cumulative_squared_error_diff(results_df, baseline_model="Historical Mean")
        st.plotly_chart(fig_csse, use_container_width=True)

    st.markdown("---")
    st.markdown("#### **Historical Subperiod Performance Breakdown**")
    subperiod_df = compute_subperiod_performance(results_df)
    st.dataframe(
        subperiod_df.style.format({
            "MAE": "{:.3f}",
            "RMSE": "{:.3f}",
            "OOS R²": "{:.4f}",
            "Directional Acc (%)": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("#### **Hypothesis Testing: Paired Moving Block Bootstrap**")
    st.markdown(
        "To account for potential autocorrelation and temporal clustering in macroeconomic forecast errors, "
        "we apply a paired **Moving Block Bootstrap (MBB)** to the loss differential series "
        "$d_t = (y_t - \\hat{y}_{t}^{\\text{baseline}})^2 - (y_t - \\hat{y}_{t}^{\\text{model}})^2$."
    )

    col_b1, col_b2, col_b3 = st.columns([2, 2, 2])
    with col_b1:
        test_model = st.selectbox(
            "Challenger Model",
            options=[m for m in metrics_by_model.keys() if m != "Historical Mean"],
            index=1 if "Macro Ridge" in metrics_by_model else 0,
        )
    with col_b2:
        baseline_model = st.selectbox(
            "Baseline Benchmark",
            options=["Historical Mean", "Autoregressive AR(1)"],
            index=0,
        )
    with col_b3:
        block_size = st.slider("Bootstrap Block Size (Quarters)", min_value=2, max_value=8, value=4)

    y_true_vals = results_df["realized"].values
    y_base_vals = results_df[f"pred_{baseline_model}"].values
    y_model_vals = results_df[f"pred_{test_model}"].values

    boot_res = paired_moving_block_bootstrap(
        y_true_vals,
        y_base_vals,
        y_model_vals,
        block_size=block_size,
        n_bootstraps=boot_resamples,
        random_state=RANDOM_SEED,
    )

    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        st.metric("Mean MSE Diff (ΔMSE)", f"{boot_res.mean_diff:+.3f} pp²", help="Positive indicates Challenger has lower error")
    with b_col2:
        st.metric("Bootstrap Std Error", f"{boot_res.std_err:.3f}")
    with b_col3:
        st.metric("95% CI Lower / Upper", f"[{boot_res.ci_lower:.2f}, {boot_res.ci_upper:.2f}]")
    with b_col4:
        st.metric("Two-Sided p-value", f"{boot_res.p_value:.3f}", delta="Significant" if boot_res.is_significant_5pct else "Not Significant", delta_color="normal" if boot_res.is_significant_5pct else "off")

    st.markdown(f"**Interpretation:** {boot_res.interpretation}")

    fig_boot = plot_bootstrap_distribution(boot_res, challenger_name=test_model, baseline_name=baseline_model)
    st.plotly_chart(fig_boot, use_container_width=True)

    st.markdown("---")
    st.markdown("#### **Export Computed Results**")
    csv_results = results_df.to_csv().encode("utf-8")
    csv_summary = metrics_summary_df.to_csv(index=False).encode("utf-8")

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="Download Walk-Forward Predictions CSV",
            data=csv_results,
            file_name=f"macrocycle_walk_forward_{target_var}_results.csv",
            mime="text/csv",
        )
    with dl_col2:
        st.download_button(
            label="Download Model Performance Summary CSV",
            data=csv_summary,
            file_name=f"macrocycle_model_metrics_{target_var}.csv",
            mime="text/csv",
        )

# ==============================================================================
# TAB 4: MACRO REGIMES
# ==============================================================================
with tab_regimes:
    st.subheader("Descriptive Growth / Inflation Regimes & Yield-Curve Dynamics")
    st.markdown(
        "Macroeconomic regimes are identified using **strictly trailing historical medians** calculated "
        "only from observations prior to each quarter. This produces four descriptive quadrants without "
        "relying on future sample statistics or circular lookahead."
    )

    regime_df = classify_growth_inflation_regimes(
        panel["gdp_growth"],
        panel["cpi_inflation_yoy"],
        min_history=20,
        window_type="expanding",
    )

    last_regime = regime_df.iloc[-1]
    last_quarter_name = str(regime_df.index[-1])

    reg_badge_col1, reg_badge_col2, reg_badge_col3, reg_badge_col4 = st.columns(4)
    with reg_badge_col1:
        st.markdown(
            f"""
            <div class="metric-badge">
                <div class="metric-badge-label">Latest Classified Quarter</div>
                <div class="metric-badge-value">{last_quarter_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with reg_badge_col2:
        st.markdown(
            f"""
            <div class="metric-badge">
                <div class="metric-badge-label">Prevailing Regime</div>
                <div class="metric-badge-value" style="color:{PALETTE['navy_primary']};">{last_regime['regime_short']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with reg_badge_col3:
        st.markdown(
            f"""
            <div class="metric-badge">
                <div class="metric-badge-label">Trailing Growth Median</div>
                <div class="metric-badge-value">{last_regime['growth_threshold']:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with reg_badge_col4:
        st.markdown(
            f"""
            <div class="metric-badge">
                <div class="metric-badge-label">Trailing Inflation Median</div>
                <div class="metric-badge-value">{last_regime['inflation_threshold']:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col_scat, col_time = st.columns([3, 2])
    with col_scat:
        fig_reg_scat = plot_regime_scatter(regime_df)
        st.plotly_chart(fig_reg_scat, use_container_width=True)
    with col_time:
        fig_reg_time = plot_regime_timeline(regime_df)
        st.plotly_chart(fig_reg_time, use_container_width=True)

    st.markdown("---")
    st.markdown("#### **Regime Historical Summary Statistics**")
    regime_stats_df = compute_regime_statistics(regime_df, panel["term_spread"])
    st.dataframe(
        regime_stats_df.style.format({
            "Frequency (%)": "{:.1f}%",
            "Mean GDP Growth (%)": "{:.2f}%",
            "GDP Growth Volatility (%)": "{:.2f}%",
            "Mean Inflation (%)": "{:.2f}%",
            "Inflation Volatility (%)": "{:.2f}%",
            "Mean 10Y-3M Spread (pp)": "{:.2f}",
            "Avg Spell Duration (Qtrs)": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("#### **Yield-Curve Term Spread and Subsequent Real Growth**")
    lead_lag = st.slider("Term Spread Leading Horizon (Quarters Ahead)", min_value=1, max_value=8, value=4)
    fig_spread = plot_term_spread_vs_subsequent_gdp(panel, lag_quarters=lead_lag)
    st.plotly_chart(fig_spread, use_container_width=True)

    st.markdown("---")
    st.markdown("#### **Regime-Conditioned Model Forecast Errors**")
    st.markdown(
        "Examining whether forecast errors systematically differ across growth/inflation regimes. "
        "Regimes with fewer than 10 quarters are flagged to guard against unstable small-sample inference."
    )
    reg_errors_df = evaluate_forecast_errors_by_regime(results_df, regime_df, min_regime_samples=10)
    st.dataframe(
        reg_errors_df.style.format({
            "MAE": "{:.3f}",
            "RMSE": "{:.3f}",
            "Mean Bias": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class="warning-callout">
            <b>Descriptive Caveat:</b> These regimes are descriptive classifications based on median splits of growth and inflation. 
            They are not official NBER recession indicators, structural Markov regime-switching states, or guaranteed turning points. 
            Negative GDP growth in a quarter does not automatically establish an official recession.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# TAB 5: RATES SCENARIO LAB
# ==============================================================================
with tab_rates:
    st.subheader("Fixed-Income Interest-Rate Sensitivity Calculator")
    st.markdown(
        "A local second-order Taylor series duration-convexity model evaluating bond price sensitivity: "
        "$$\\frac{\\Delta P}{P} \\approx -D \\cdot \\Delta y + \\frac{1}{2} C \\cdot (\\Delta y)^2$$"
    )

    st.markdown(
        """
        <div class="research-callout">
            <b>Model Scope & Independence Notice:</b> This is an independent local approximation. 
            It assumes an instantaneous parallel curve shift and does not model curve reshaping, credit spreads, 
            or roll-down return. Fixed-income returns are <b>not an automatic output of the GDP model</b>; 
            scenario inputs are user-configured to explore interest-rate sensitivity under alternative hypothetical macro environments.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_inp1, col_inp2, col_inp3, col_inp4 = st.columns(4)
    with col_inp1:
        mod_dur = st.number_input("Modified Duration (D, yrs)", min_value=0.5, max_value=25.0, value=7.0, step=0.5)
    with col_inp2:
        convexity = st.number_input("Convexity (C, yrs²)", min_value=0.0, max_value=400.0, value=65.0, step=5.0)
    with col_inp3:
        init_yield = st.number_input("Initial Yield (y₀, %)", min_value=0.1, max_value=15.0, value=4.25, step=0.25)
    with col_inp4:
        shock_bps = st.number_input("Parallel Yield Shock (Δy, bps)", min_value=-500, max_value=500, value=75, step=25)

    single_calc = compute_duration_convexity_price_change(
        modified_duration=mod_dur,
        convexity=convexity,
        yield_shock_bps=float(shock_bps),
        initial_yield_pct=init_yield,
    )

    st.markdown("#### **Local Sensitivity Breakdown**")
    res_c1, res_c2, res_c3, res_c4 = st.columns(4)
    with res_c1:
        st.metric("Linear Duration Impact", f"{single_calc.duration_effect_pct:+.2f}%", help="-D * Δy")
    with res_c2:
        st.metric("Convexity Cushion", f"{single_calc.convexity_effect_pct:+.2f}%", help="+0.5 * C * (Δy)²")
    with res_c3:
        st.metric("Total Est. Price Change", f"{single_calc.total_price_change_pct:+.2f}%")
    with res_c4:
        st.metric("Resulting Yield", f"{single_calc.new_yield_pct:.2f}%")

    grid_df = generate_shock_grid_analysis(
        modified_duration=mod_dur,
        convexity=convexity,
        initial_yield_pct=init_yield,
        min_shock_bps=-300,
        max_shock_bps=300,
        step_bps=25,
    )
    fig_rate_curve = plot_rate_sensitivity_curve(grid_df, user_shock_bps=float(shock_bps))
    st.plotly_chart(fig_rate_curve, use_container_width=True)

    st.markdown("---")
    st.markdown("#### **Side-by-Side Macro Narrative Scenarios**")
    st.markdown(
        "Evaluate hypothetical fixed-income portfolio impact across macro scenarios. "
        "Macro assumptions and yield shocks remain distinct from statistical GDP forecast outputs."
    )

    scenario_df = evaluate_macro_scenarios(
        DEFAULT_PRESET_SCENARIOS,
        modified_duration=mod_dur,
        convexity=convexity,
        initial_yield_pct=init_yield,
    )

    col_s_table, col_s_bar = st.columns([3, 2])
    with col_s_table:
        st.dataframe(
            scenario_df.style.format({
                "Hypothetical GDP Growth (%)": "{:.1f}%",
                "Hypothetical Inflation (%)": "{:.1f}%",
                "Parallel Yield Shock (bps)": "{:+.0f} bps",
                "Resulting Yield (%)": "{:.2f}%",
                "Duration Impact (%)": "{:+.2f}%",
                "Convexity Adjustment (%)": "{:+.2f}%",
                "Total Est. Bond Price Change (%)": "{:+.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )
    with col_s_bar:
        fig_scen_bar = plot_scenario_comparison_bar(scenario_df)
        st.plotly_chart(fig_scen_bar, use_container_width=True)

# ==============================================================================
# TAB 6: RESEARCH NOTES
# ==============================================================================
with tab_notes:
    st.subheader("Methodology, Provenance & Reproducibility")

    st.markdown(
        """
        ### 1. Data Provenance & Transformations
        All historical macroeconomic series are bundled locally in `data/` and loaded offline.
        
        | Series ID | Concept | Source Agency | Frequency | Transformation Formula |
        |---|---|---|---|---|
        | `GDPC1` | Real Gross Domestic Product | Bureau of Economic Analysis | Quarterly | $400 \\times [\\ln(\\text{GDP}_t) - \\ln(\\text{GDP}_{t-1})]$ |
        | `CPIAUCSL` | Consumer Price Index (Urban) | Bureau of Labor Statistics | Monthly $\\to$ Quarterly Mean | $100 \\times [\\text{CPI}_t / \\text{CPI}_{t-4} - 1]$ and $400 \\times \\Delta \\ln(\\text{CPI})$ |
        | `UNRATE` | Civilian Unemployment Rate | Bureau of Labor Statistics | Monthly $\\to$ Quarterly Mean | Level (\\%) and $\\Delta \\text{UNRATE} = \\text{UNRATE}_t - \\text{UNRATE}_{t-1}$ |
        | `INDPRO` | Industrial Production Index | Federal Reserve Board | Monthly $\\to$ Quarterly Mean | $400 \\times [\\ln(\\text{IP}_t) - \\ln(\\text{IP}_{t-1})]$ |
        | `FEDFUNDS` | Effective Federal Funds Rate | Federal Reserve Board | Monthly $\\to$ Quarterly Mean | Level (\\% per annum) |
        | `GS10` | 10-Year Treasury Yield | Federal Reserve Board | Monthly $\\to$ Quarterly Mean | Level (\\% per annum) |
        | `TB3MS` | 3-Month Treasury Bill Rate | Federal Reserve Board | Monthly $\\to$ Quarterly Mean | Level (\\% per annum) |
        | `Spread` | Term Spread | Derived | Quarterly | $\\text{GS10}_t - \\text{TB3MS}_t$ (percentage points) |

        ---
        ### 2. Revision Bias & The ALFRED Architecture
        - **Retrospective Backtest Limitation:** This study relies on revised historical data released by BEA and BLS. 
          Real-time macroeconomic forecasters at quarter $t$ only possessed preliminary (Advance) prints. 
          Subsequent annual revisions, benchmark comprehensive updates, and seasonal factor changes alter historical values.
        - **Mitigation:** A strict 1-quarter feature lag is applied so that features at quarter $t$ only predict growth at $t+1$. 
          However, lags do not eliminate data revision bias.
        - **ALFRED Reference:** For point-in-time real-time vintages, consult the 
          [ALFRED (ArchivaL Federal Reserve Economic Data) Documentation](https://alfred.stlouisfed.org/help).

        ---
        ### 3. Walk-Forward Expanding Window Equations
        At step $k$ for target quarter $t = k+1$:
        - **Historical Mean Benchmark:**
          $$\\hat{y}_{k+1}^{\\text{Mean}} = \\frac{1}{k} \\sum_{i=1}^k y_i$$
        - **Autoregressive AR(1) Model:**
          $$\\hat{y}_{k+1}^{\\text{AR(1)}} = \\hat{\\beta}_0^{(k)} + \\hat{\\beta}_1^{(k)} y_k$$
        - **Multivariate Macro Ridge:**
          $$\\hat{\\beta}^{(k)} = \\arg\\min_{\\beta} \\| y_{1:k} - X_{1:k} \\beta \\|_2^2 + \\alpha \\| \\beta \\|_2^2$$
          where $X$ is standardized using mean and variance computed strictly on $X_{1:k}$.
        - **Out-of-Sample $R^2$:**
          $$R^2_{\\text{OOS}} = 1 - \\frac{\\sum_{t} (y_t - \\hat{y}_t)^2}{\\sum_t (y_t - \\bar{y}_{\\text{train}, t})^2}$$
        
        ---
        ### 4. Hypothesis Testing & Falsification Standard
        Macroeconomic forecast errors typically exhibit temporal persistence. We conduct a **Moving Block Bootstrap (MBB)** 
        on the squared error differential:
        $$d_t = (y_t - \\hat{y}_{t}^{\\text{baseline}})^2 - (y_t - \\hat{y}_{t}^{\\text{challenger}})^2$$
        - Resampling blocks of contiguous quarters (default $b=4$ quarters) preserves the empirical autocorrelation structure.
        - If the 95\\% confidence interval spans zero ($p \\ge 0.05$), the null hypothesis of equal forecast accuracy cannot be rejected.

        ---
        ### 5. Fixed-Income Sensitivity Equations
        For a bond with modified duration $D$ and convexity $C$, subject to a parallel yield shock $\\Delta y = \\Delta y_{\\text{bps}} / 10{,}000$:
        $$\\frac{\\Delta P}{P} = \\underbrace{-D \\cdot \\Delta y}_{\\text{Linear Duration}} + \\underbrace{\\frac{1}{2} C \\cdot (\\Delta y)^2}_{\\text{Convexity Adjustment}}$$
        
        ---
        ### 6. Automated Verification & Reproducibility
        Run the complete test suite locally:
        ```bash
        python -m pytest
        ```
        Deterministic random seed: `42`.
        """
    )

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748B; font-size:12px; padding:10px;'>"
    "MacroCycle Research Lab &bull; Quantitative Macroeconomic & Fixed-Income Research System &bull; Offline Reproducible Distribution"
    "</div>",
    unsafe_allow_html=True,
)
