"""
Data Pipeline for MacroCycle: Growth, Inflation and Yield-Curve Research Lab.

Handles data loading from bundled historical datasets, cleaning, quarterly resampling,
macroeconomic variable transformations, and strict chronological feature/target alignment.
"""

from dataclasses import dataclass
import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Default directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CURATED_CSV_PATH = os.path.join(DATA_DIR, "curated_macro_quarterly.csv")
FALLBACK_CSV_PATH = os.path.join(DATA_DIR, "statsmodels_macrodata_fallback.csv")
MANIFEST_PATH = os.path.join(DATA_DIR, "data_manifest.json")


@dataclass(frozen=True)
class VariableSpec:
    symbol: str
    name: str
    units: str
    frequency: str
    transformation_formula: str
    description: str


VARIABLE_SPECS: Dict[str, VariableSpec] = {
    "gdp_growth": VariableSpec(
        symbol="GDP_growth_t",
        name="Real GDP Growth (QoQ Ann.)",
        units="Percent (Annualized)",
        frequency="Quarterly",
        transformation_formula="400 * [ln(GDP_t) - ln(GDP_{t-1})]",
        description="Quarter-on-quarter annualized percentage growth rate in real gross domestic product.",
    ),
    "cpi_inflation_yoy": VariableSpec(
        symbol="CPI_yoy_t",
        name="CPI Inflation (YoY)",
        units="Percent",
        frequency="Quarterly (resampled from monthly)",
        transformation_formula="100 * [CPI_t / CPI_{t-4} - 1]",
        description="Year-over-year percentage change in headline Consumer Price Index for All Urban Consumers.",
    ),
    "cpi_inflation_qoq_ann": VariableSpec(
        symbol="CPI_qoq_t",
        name="CPI Inflation (QoQ Ann.)",
        units="Percent (Annualized)",
        frequency="Quarterly (resampled from monthly)",
        transformation_formula="400 * [ln(CPI_t) - ln(CPI_{t-1})]",
        description="Quarter-on-quarter annualized percentage change in headline Consumer Price Index.",
    ),
    "unemployment_rate": VariableSpec(
        symbol="UNRATE_t",
        name="Unemployment Rate",
        units="Percent",
        frequency="Quarterly (resampled from monthly mean)",
        transformation_formula="Level (UNRATE_t)",
        description="Quarterly arithmetic average of the monthly civilian unemployment rate.",
    ),
    "unemployment_change": VariableSpec(
        symbol="ΔUNRATE_t",
        name="Unemployment Rate Change (QoQ)",
        units="Percentage Points",
        frequency="Quarterly",
        transformation_formula="UNRATE_t - UNRATE_{t-1}",
        description="Quarterly change in the civilian unemployment rate.",
    ),
    "industrial_production_growth": VariableSpec(
        symbol="IP_growth_t",
        name="Industrial Production Growth (QoQ Ann.)",
        units="Percent (Annualized)",
        frequency="Quarterly (resampled from monthly mean)",
        transformation_formula="400 * [ln(INDPRO_t) - ln(INDPRO_{t-1})]",
        description="Quarter-on-quarter annualized percentage growth in the total industrial production index.",
    ),
    "policy_rate": VariableSpec(
        symbol="FEDFUNDS_t",
        name="Federal Funds Effective Rate",
        units="Percent per Annum",
        frequency="Quarterly (resampled from monthly mean)",
        transformation_formula="Level (FEDFUNDS_t)",
        description="Quarterly average of effective overnight federal funds rate.",
    ),
    "term_spread": VariableSpec(
        symbol="Spread_t",
        name="Yield-Curve Term Spread (10Y - 3M)",
        units="Percentage Points",
        frequency="Quarterly (resampled from monthly mean)",
        transformation_formula="GS10_t - TB3MS_t",
        description="Spread between 10-year Treasury constant-maturity yield and 3-month Treasury bill rate.",
    ),
}


def load_manifest() -> Dict:
    """Load data manifest if available, or return default metadata structure."""
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "dataset_name": "MacroCycle Research Historical Panel",
        "primary_source": "FRED / St. Louis Fed",
        "revision_status": "Retrospective revised series (subject to data revision bias)",
    }


def compute_gdp_growth(gdp_series: pd.Series) -> pd.Series:
    """
    Calculate annualized quarter-on-quarter real GDP growth:
    GDP_growth_t = 400 * [log(GDP_t) - log(GDP_{t-1})]
    """
    if (gdp_series <= 0).any():
        raise ValueError("GDP series contains non-positive values; log transform invalid.")
    growth = 400.0 * (np.log(gdp_series) - np.log(gdp_series.shift(1)))
    growth.name = "gdp_growth"
    return growth


def compute_cpi_inflation_yoy(cpi_series: pd.Series) -> pd.Series:
    """
    Calculate year-over-year CPI inflation:
    CPI_inflation_yoy = 100 * [CPI_t / CPI_{t-4} - 1]
    """
    if (cpi_series <= 0).any():
        raise ValueError("CPI series contains non-positive values.")
    infl = 100.0 * (cpi_series / cpi_series.shift(4) - 1.0)
    infl.name = "cpi_inflation_yoy"
    return infl


def compute_cpi_inflation_qoq_ann(cpi_series: pd.Series) -> pd.Series:
    """
    Calculate annualized quarter-on-quarter CPI inflation:
    CPI_inflation_qoq_ann = 400 * [log(CPI_t) - log(CPI_{t-1})]
    """
    if (cpi_series <= 0).any():
        raise ValueError("CPI series contains non-positive values.")
    infl = 400.0 * (np.log(cpi_series) - np.log(cpi_series.shift(1)))
    infl.name = "cpi_inflation_qoq_ann"
    return infl


def compute_ip_growth(ip_series: pd.Series) -> pd.Series:
    """
    Calculate annualized quarter-on-quarter industrial production growth:
    IP_growth_t = 400 * [log(INDPRO_t) - log(INDPRO_{t-1})]
    """
    if (ip_series <= 0).any():
        raise ValueError("Industrial production series contains non-positive values.")
    ip_growth = 400.0 * (np.log(ip_series) - np.log(ip_series.shift(1)))
    ip_growth.name = "industrial_production_growth"
    return ip_growth


def compute_term_spread(gs10: pd.Series, tb3ms: pd.Series) -> pd.Series:
    """
    Calculate 10-year minus 3-month Treasury yield curve term spread.
    Spread_t = GS10_t - TB3MS_t
    """
    spread = gs10 - tb3ms
    spread.name = "term_spread"
    return spread


def load_curated_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load curated quarterly macro series from CSV and set clean PeriodIndex('Q').
    """
    path = csv_path or CURATED_CSV_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Curated dataset not found at {path}. Run fetch script or check data dir.")
    
    df = pd.read_csv(path, index_col=0)
    df.index = pd.PeriodIndex(df.index.astype(str), freq="Q")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_fallback_statsmodels(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load statsmodels macrodata fallback dataset (1959Q1-2009Q3).
    """
    path = csv_path or FALLBACK_CSV_PATH
    if not os.path.exists(path):
        import statsmodels.api as sm
        mdata = sm.datasets.macrodata.load_pandas().data
        periods = [pd.Period(f"{int(r.year)}Q{int(r.quarter)}", freq="Q") for _, r in mdata.iterrows()]
        df = pd.DataFrame(index=pd.PeriodIndex(periods, name="period"))
        df["GDPC1"] = mdata["realgdp"].values
        df["CPIAUCSL"] = mdata["cpi"].values
        df["UNRATE"] = mdata["unemp"].values
        df["TB3MS"] = mdata["tbilrate"].values
        # Approximations for missing series in fallback:
        # FEDFUNDS closely tracked TB3MS; GS10 can be approximated from tbilrate + historical mean spread (1.5%)
        df["FEDFUNDS"] = mdata["tbilrate"].values
        df["GS10"] = mdata["tbilrate"].values + 1.5
        df["INDPRO"] = mdata["realgdp"].values / mdata["realgdp"].values[0] * 100.0
        return df

    df = pd.read_csv(path, index_col=0)
    df.index = pd.PeriodIndex(df.index.astype(str), freq="Q")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "FEDFUNDS" not in df.columns and "TB3MS" in df.columns:
        df["FEDFUNDS"] = df["TB3MS"]
    if "GS10" not in df.columns and "TB3MS" in df.columns:
        df["GS10"] = df["TB3MS"] + 1.5
    if "INDPRO" not in df.columns and "GDPC1" in df.columns:
        df["INDPRO"] = df["GDPC1"] / df["GDPC1"].iloc[0] * 100.0
    return df


def build_research_panel(
    raw_df: pd.DataFrame,
    start_quarter: Optional[str] = "1954Q3",
    end_quarter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Construct fully transformed quarterly research panel with macro predictors.
    Drops any initial rows with missing values resulting from multi-period lags.
    """
    df = raw_df.copy()
    
    # Required raw columns check
    required_cols = ["GDPC1", "CPIAUCSL", "UNRATE", "FEDFUNDS", "INDPRO", "GS10", "TB3MS"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required raw series columns: {missing_cols}")

    panel = pd.DataFrame(index=df.index)

    # 1. Real GDP growth (QoQ annualized)
    panel["gdp_growth"] = compute_gdp_growth(df["GDPC1"])

    # 2. CPI Inflation YoY and QoQ annualized
    panel["cpi_inflation_yoy"] = compute_cpi_inflation_yoy(df["CPIAUCSL"])
    panel["cpi_inflation_qoq_ann"] = compute_cpi_inflation_qoq_ann(df["CPIAUCSL"])

    # 3. Unemployment rate and change
    panel["unemployment_rate"] = df["UNRATE"]
    panel["unemployment_change"] = df["UNRATE"] - df["UNRATE"].shift(1)

    # 4. Industrial production growth (QoQ annualized)
    panel["industrial_production_growth"] = compute_ip_growth(df["INDPRO"])

    # 5. Effective policy rate
    panel["policy_rate"] = df["FEDFUNDS"]

    # 6. Yield curve term spread (10Y - 3M)
    panel["term_spread"] = compute_term_spread(df["GS10"], df["TB3MS"])

    # Optional slice by user date boundaries
    if start_quarter:
        start_p = pd.Period(start_quarter, freq="Q")
        panel = panel[panel.index >= start_p]
    if end_quarter:
        end_p = pd.Period(end_quarter, freq="Q")
        panel = panel[panel.index <= end_p]

    # Drop any remaining non-finite / NaN values (e.g. from shift operations)
    panel = panel.dropna()
    if len(panel) < 20:
        raise ValueError(f"Insufficient panel length ({len(panel)} quarters) after transformation and filtering.")

    return panel


def create_forecasting_dataset(
    panel: pd.DataFrame,
    target_variable: str = "gdp_growth",
    predictor_variables: Optional[List[str]] = None,
    horizon: int = 1,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Construct aligned feature matrix X and target vector y for h-quarter-ahead forecasting.
    
    Parameters:
    -----------
    panel : pd.DataFrame
        Cleaned quarterly research panel.
    target_variable : str
        Target variable name (e.g. 'gdp_growth' or 'cpi_inflation_qoq_ann').
    predictor_variables : list of str, optional
        Subset of panel columns to include as features. Default is all standard predictors.
    horizon : int
        Forecast horizon in quarters (default h=1).
        At time t, features X_t predict y_{t+h}.
        
    Returns:
    --------
    X : pd.DataFrame
        Features aligned at observation quarter t.
    y : pd.Series
        Realized target at future quarter t+h, indexed at future quarter t+h.
    aligned_df : pd.DataFrame
        Combined aligned dataframe containing features, target, target_date, and feature_date.
    """
    if target_variable not in panel.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in panel columns: {list(panel.columns)}")

    default_predictors = [
        "gdp_growth",
        "cpi_inflation_yoy",
        "unemployment_rate",
        "unemployment_change",
        "industrial_production_growth",
        "policy_rate",
        "term_spread",
    ]
    predictors = predictor_variables if predictor_variables is not None else default_predictors
    for p in predictors:
        if p not in panel.columns:
            raise ValueError(f"Predictor variable '{p}' not found in panel.")

    # Target y is shifted back by horizon periods so that X at row t corresponds to y at t+h
    # Feature row at period t: data available up through quarter t
    # Target value realized at quarter t + horizon
    target_series = panel[target_variable].shift(-horizon)

    aligned = panel[predictors].copy()
    aligned["target"] = target_series
    aligned["target_period"] = panel.index.shift(horizon)
    aligned["feature_period"] = panel.index

    # Drop the last horizon rows where target is NaN (since future quarter hasn't occurred yet)
    valid_mask = aligned["target"].notna()
    aligned = aligned[valid_mask]

    X = aligned[predictors]
    # y is indexed by the realization quarter (t + horizon) for clean reporting
    y = aligned["target"]
    y.index = aligned["target_period"]

    return X, y, aligned


def get_variable_metadata_table(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Generate an institutional documentation table showing variable, symbol, units,
    transformation, frequency, first valid date, last valid date, and valid observations.
    """
    records = []
    for col in panel.columns:
        spec = VARIABLE_SPECS.get(
            col,
            VariableSpec(
                symbol=col,
                name=col.replace("_", " ").title(),
                units="Standard",
                frequency="Quarterly",
                transformation_formula="Direct level",
                description="Macroeconomic indicator",
            ),
        )
        s = panel[col].dropna()
        records.append({
            "Variable": spec.name,
            "Symbol": spec.symbol,
            "Units": spec.units,
            "Frequency": spec.frequency,
            "Transformation": spec.transformation_formula,
            "First Quarter": str(s.index.min()),
            "Last Quarter": str(s.index.max()),
            "Observations": int(len(s)),
            "Mean": float(s.mean()),
            "Std Dev": float(s.std()),
            "Min": float(s.min()),
            "Max": float(s.max()),
        })
    return pd.DataFrame(records)
