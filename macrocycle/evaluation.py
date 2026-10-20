"""
Walk-Forward Out-of-Sample Evaluation Engine for MacroCycle Research Lab.

Implements expanding-window walk-forward validation with zero lookahead,
genuine out-of-sample performance metrics (MAE, RMSE, OOS R-squared relative
to training mean), subperiod analysis, and moving block bootstrap hypothesis testing.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from macrocycle.models import (
    AutoregressiveBaselineModel,
    BaseMacroModel,
    ConstrainedGBRModel,
    HistoricalMeanModel,
    RidgeMacroModel,
)


@dataclass
class OOSMetrics:
    mae: float
    rmse: float
    r2_oos: float
    mda: float
    mean_bias: float
    observations: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "MAE": self.mae,
            "RMSE": self.rmse,
            "OOS R²": self.r2_oos,
            "Directional Acc (%)": self.mda * 100.0,
            "Mean Bias": self.mean_bias,
            "Observations": int(self.observations),
        }


@dataclass
class BootstrapResult:
    mean_diff: float  # MSE(Baseline) - MSE(Model). Positive means Model is better.
    std_err: float
    ci_lower: float
    ci_upper: float
    p_value: float
    block_size: int
    n_bootstraps: int
    is_significant_5pct: bool
    interpretation: str


def compute_oos_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_means: np.ndarray,
) -> OOSMetrics:
    """
    Compute genuine out-of-sample metrics.
    OOS R-squared is computed relative to the historical training mean benchmark:
    R2_OOS = 1 - sum((y - y_hat)^2) / sum((y - y_bar_train)^2)
    """
    if len(y_true) == 0:
        raise ValueError("Empty ground truth array in compute_oos_metrics.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have matching length.")

    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mean_bias = float(np.mean(y_pred - y_true))

    # Directional accuracy (percentage of identical signs)
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    mda = float(np.mean(sign_true == sign_pred))

    # OOS R-squared vs training mean
    benchmark_errors = y_true - y_train_means
    sse_model = float(np.sum(residuals ** 2))
    sse_bench = float(np.sum(benchmark_errors ** 2))
    if sse_bench == 0:
        r2_oos = 0.0
    else:
        r2_oos = float(1.0 - (sse_model / sse_bench))

    return OOSMetrics(
        mae=mae,
        rmse=rmse,
        r2_oos=r2_oos,
        mda=mda,
        mean_bias=mean_bias,
        observations=len(y_true),
    )


def run_walk_forward_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    min_train_periods: int = 40,
    ridge_alpha: float = 10.0,
    gbr_estimators: int = 35,
    random_state: int = 42,
    include_gbr: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, OOSMetrics], Dict[str, BaseMacroModel]]:
    """
    Strict expanding-window walk-forward evaluation.
    
    At step k (from min_train_periods to N-1):
      - Train models strictly on X.iloc[:k], y.iloc[:k]
      - Predict step k for target quarter y.index[k] using feature X.iloc[k]
      - Record out-of-sample prediction and training mean benchmark
      
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame indexed by realization quarter containing realized y,
        benchmark training mean, and predictions from all models.
    metrics_by_model : Dict[str, OOSMetrics]
        OOS metrics dictionary keyed by model name.
    last_fitted_models : Dict[str, BaseMacroModel]
        Models fitted on the full training window for inspection.
    """
    n_samples = len(X)
    if n_samples <= min_train_periods:
        raise ValueError(
            f"Total available periods ({n_samples}) must exceed minimum training window ({min_train_periods})."
        )

    # Initialize model constructors
    models = {
        "Historical Mean": HistoricalMeanModel(),
        "Autoregressive AR(1)": AutoregressiveBaselineModel(target_col_idx=0),
        "Macro Ridge": RidgeMacroModel(alpha=ridge_alpha),
    }
    if include_gbr:
        models["Constrained GBR"] = ConstrainedGBRModel(
            n_estimators=gbr_estimators,
            max_depth=2,
            min_samples_leaf=5,
            learning_rate=0.05,
            random_state=random_state,
        )

    records = []
    feature_names = list(X.columns)

    X_mat = X.values
    y_vec = y.values

    # Step strictly chronologically
    for k in range(min_train_periods, n_samples):
        # Strictly past training window
        X_train = X_mat[:k]
        y_train = y_vec[:k]

        # Feature at quarter k predicting target realized at quarter k (which is y.index[k])
        X_eval = X_mat[k : k + 1]
        realized_target = y_vec[k]
        realized_period = y.index[k]
        feature_period = X.index[k]

        train_mean = float(np.mean(y_train))

        row_record = {
            "target_period": str(realized_period),
            "feature_period": str(feature_period),
            "realized": realized_target,
            "training_mean_benchmark": train_mean,
        }

        for model_name, model in models.items():
            model.fit(X_train, y_train, feature_names=feature_names)
            pred = float(model.predict(X_eval)[0])
            row_record[f"pred_{model_name}"] = pred
            row_record[f"err_{model_name}"] = realized_target - pred

        records.append(row_record)

    results_df = pd.DataFrame(records).set_index("target_period")

    # Compute overall metrics
    metrics_by_model = {}
    y_true = results_df["realized"].values
    y_bench = results_df["training_mean_benchmark"].values

    for model_name in models.keys():
        pred_col = f"pred_{model_name}"
        y_pred = results_df[pred_col].values
        metrics_by_model[model_name] = compute_oos_metrics(y_true, y_pred, y_bench)

    return results_df, metrics_by_model, models


def compute_subperiod_performance(
    results_df: pd.DataFrame,
    subperiods: Optional[Dict[str, Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Evaluate model out-of-sample performance across historical macroeconomic regimes/eras.
    """
    if subperiods is None:
        subperiods = {
            "Pre-1984 (Volatile Inflation)": ("1950Q1", "1983Q4"),
            "1984-2007 (Great Moderation)": ("1984Q1", "2007Q4"),
            "2008-2019 (Post-GFC / Low Rates)": ("2008Q1", "2019Q4"),
            "2020+ (Pandemic & Aftermath)": ("2020Q1", "2030Q4"),
        }

    # Identify models
    pred_cols = [c for c in results_df.columns if c.startswith("pred_")]
    model_names = [c.replace("pred_", "") for c in pred_cols]

    rows = []
    # Convert index to PeriodIndex for chronological range slicing
    p_index = pd.PeriodIndex(results_df.index, freq="Q")

    for era_name, (start_str, end_str) in subperiods.items():
        start_p = pd.Period(start_str, freq="Q")
        end_p = pd.Period(end_str, freq="Q")
        mask = (p_index >= start_p) & (p_index <= end_p)
        era_df = results_df[mask]

        if len(era_df) < 4:
            continue

        y_true = era_df["realized"].values
        y_bench = era_df["training_mean_benchmark"].values

        for m_name in model_names:
            y_pred = era_df[f"pred_{m_name}"].values
            m = compute_oos_metrics(y_true, y_pred, y_bench)
            rows.append({
                "Historical Subperiod": era_name,
                "Model": m_name,
                "Observations": m.observations,
                "MAE": m.mae,
                "RMSE": m.rmse,
                "OOS R²": m.r2_oos,
                "Directional Acc (%)": m.mda * 100.0,
            })

    return pd.DataFrame(rows)


def paired_moving_block_bootstrap(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_model: np.ndarray,
    block_size: int = 4,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> BootstrapResult:
    """
    Moving Block Bootstrap test for forecast accuracy differential.
    
    Loss differential d_t = (y_t - y_hat_base,t)^2 - (y_t - y_hat_model,t)^2.
    A positive mean differential implies the model has lower MSE than the baseline.
    The block bootstrap preserves temporal correlation in forecast errors.
    """
    N = len(y_true)
    if N < block_size * 2:
        raise ValueError(f"Sample size {N} too small for block size {block_size}.")

    loss_base = (y_true - y_pred_baseline) ** 2
    loss_model = (y_true - y_pred_model) ** 2
    d = loss_base - loss_model  # positive => model outperforms baseline
    mean_d = float(np.mean(d))

    # Construct overlapping blocks
    num_blocks_available = N - block_size + 1
    blocks = [d[i : i + block_size] for i in range(num_blocks_available)]

    rng = np.random.default_rng(random_state)
    k_needed = int(np.ceil(N / block_size))

    boot_means = np.zeros(n_bootstraps, dtype=float)
    for b in range(n_bootstraps):
        sampled_block_indices = rng.integers(0, num_blocks_available, size=k_needed)
        boot_sample = np.concatenate([blocks[idx] for idx in sampled_block_indices])[:N]
        boot_means[b] = np.mean(boot_sample)

    std_err = float(np.std(boot_means, ddof=1))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # Two-sided empirical p-value for H0: mean(d) = 0
    p_upper = np.mean(boot_means <= 0)
    p_lower = np.mean(boot_means >= 0)
    p_val = float(2.0 * min(p_upper, p_lower))
    p_val = min(1.0, max(0.0, p_val))

    is_sig = bool(ci_lower > 0 and p_val < 0.05)

    if ci_lower > 0:
        interp = "Statistically significant outperformance over baseline at 95% confidence."
    elif ci_upper < 0:
        interp = "Statistically significant underperformance relative to baseline."
    else:
        interp = "No statistically significant difference in forecast accuracy (CI spans zero)."

    return BootstrapResult(
        mean_diff=mean_d,
        std_err=std_err,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_val,
        block_size=block_size,
        n_bootstraps=n_bootstraps,
        is_significant_5pct=is_sig,
        interpretation=interp,
    )
