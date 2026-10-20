"""
Fetch historical macroeconomic data from FRED and fallback sources,
clean, resample to quarterly frequency, construct research panels,
and generate the data manifest.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

SERIES_META = {
    "GDPC1": {
        "title": "Real Gross Domestic Product",
        "units": "Billions of Chained 2017 Dollars, Seasonally Adjusted Annual Rate",
        "frequency": "Quarterly",
        "source": "U.S. Bureau of Economic Analysis / FRED",
        "url": "https://fred.stlouisfed.org/series/GDPC1",
        "resample_rule": "quarterly_native",
        "description": "Inflation-adjusted measure of the value of all goods and services produced by the U.S. economy.",
    },
    "CPIAUCSL": {
        "title": "Consumer Price Index for All Urban Consumers: All Items",
        "units": "Index 1982-1984=100, Seasonally Adjusted",
        "frequency": "Monthly",
        "source": "U.S. Bureau of Labor Statistics / FRED",
        "url": "https://fred.stlouisfed.org/series/CPIAUCSL",
        "resample_rule": "quarterly_mean",
        "description": "Monthly measure of the average change over time in prices paid by urban consumers for market basket.",
    },
    "UNRATE": {
        "title": "Unemployment Rate",
        "units": "Percent, Seasonally Adjusted",
        "frequency": "Monthly",
        "source": "U.S. Bureau of Labor Statistics / FRED",
        "url": "https://fred.stlouisfed.org/series/UNRATE",
        "resample_rule": "quarterly_mean",
        "description": "Percentage of the civilian labor force that is unemployed.",
    },
    "FEDFUNDS": {
        "title": "Federal Funds Effective Rate",
        "units": "Percent, Monthly Average",
        "frequency": "Monthly",
        "source": "Board of Governors of the Federal Reserve System / FRED",
        "url": "https://fred.stlouisfed.org/series/FEDFUNDS",
        "resample_rule": "quarterly_mean",
        "description": "Volume-weighted median of overnight federal funds transactions.",
    },
    "INDPRO": {
        "title": "Industrial Production: Total Index",
        "units": "Index 2017=100, Seasonally Adjusted",
        "frequency": "Monthly",
        "source": "Board of Governors of the Federal Reserve System / FRED",
        "url": "https://fred.stlouisfed.org/series/INDPRO",
        "resample_rule": "quarterly_mean",
        "description": "Measures the real output of manufacturing, mining, and electric and gas utilities.",
    },
    "GS10": {
        "title": "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity",
        "units": "Percent per Annum, Monthly Average",
        "frequency": "Monthly",
        "source": "Board of Governors of the Federal Reserve System / FRED",
        "url": "https://fred.stlouisfed.org/series/GS10",
        "resample_rule": "quarterly_mean",
        "description": "Constant maturity 10-year Treasury yield, monthly average.",
    },
    "TB3MS": {
        "title": "3-Month Treasury Bill Secondary Market Rate",
        "units": "Percent per Annum, Monthly Average",
        "frequency": "Monthly",
        "source": "Board of Governors of the Federal Reserve System / FRED",
        "url": "https://fred.stlouisfed.org/series/TB3MS",
        "resample_rule": "quarterly_mean",
        "description": "Secondary market rate for 3-month Treasury bills, discount basis, monthly average.",
    },
}


def download_fred_series(series_id: str, retries: int = 3, timeout: int = 30) -> pd.DataFrame:
    """Download single FRED series via public fredgraph CSV endpoint with retries."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MacroCycleResearch/1.0"}
    raw_path = os.path.join(RAW_DIR, f"{series_id}.csv")

    for attempt in range(1, retries + 1):
        try:
            print(f"Fetching {series_id} (attempt {attempt}/{retries})...")
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200 and len(resp.text) > 50:
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                print(f"  Successfully wrote {raw_path} ({len(resp.text)} bytes)")
                df = pd.read_csv(raw_path, na_values=[".", "NA", "null", ""])
                date_col = df.columns[0]
                val_col = df.columns[1]
                df[date_col] = pd.to_datetime(df[date_col])
                df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
                df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
                return df
            else:
                print(f"  HTTP status {resp.status_code}, retrying...")
        except Exception as exc:
            print(f"  Request error: {exc}")
            if attempt < retries:
                time.sleep(2)

    if os.path.exists(raw_path):
        print(f"  Falling back to previously cached {raw_path}")
        df = pd.read_csv(raw_path, na_values=[".", "NA", "null", ""])
        date_col = df.columns[0]
        val_col = df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col])
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        return df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    raise RuntimeError(f"Could not download or load series {series_id}")


def build_fallback_statsmodels() -> pd.DataFrame:
    """Extract and structure the statsmodels macrodata fallback dataset."""
    import statsmodels.api as sm
    mdata = sm.datasets.macrodata.load_pandas().data
    # Construct quarterly period index from year and quarter
    # Note: statsmodels macrodata runs from 1959 Q1 to 2009 Q3
    periods = []
    for _, row in mdata.iterrows():
        y = int(row["year"])
        q = int(row["quarter"])
        periods.append(pd.Period(f"{y}Q{q}", freq="Q"))
    
    mdf = pd.DataFrame(index=pd.PeriodIndex(periods, name="period"))
    mdf["GDPC1"] = mdata["realgdp"].values
    mdf["CPIAUCSL"] = mdata["cpi"].values
    mdf["UNRATE"] = mdata["unemp"].values
    mdf["TB3MS"] = mdata["tbilrate"].values
    # In macrodata, FEDFUNDS and GS10 are not present; approximate term spread or flag limitation
    # We save this as a separate standalone fallback CSV
    fallback_path = os.path.join(DATA_DIR, "statsmodels_macrodata_fallback.csv")
    mdf.to_csv(fallback_path)
    print(f"Saved statsmodels fallback to {fallback_path} (shape: {mdf.shape})")
    return mdf


def run():
    os.makedirs(RAW_DIR, exist_ok=True)
    raw_dfs = {}
    for sid in SERIES_META.keys():
        raw_dfs[sid] = download_fred_series(sid)

    # Build Quarterly Resampled Panel
    # All monthly series resampled to quarterly mean to represent average macroeconomic condition across quarter
    quarterly_series = {}

    for sid, df in raw_dfs.items():
        val_col = [c for c in df.columns if c != "observation_date" and c != df.columns[0]][0]
        s = df.set_index(df.columns[0])[val_col].dropna()
        s.index = pd.to_datetime(s.index)
        
        if sid == "GDPC1":
            # GDPC1 is already quarterly, indexed at start of quarter (e.g. 1947-01-01 -> 1947Q1)
            # Convert to PeriodIndex('Q')
            q_s = s.to_period("Q")
        else:
            # Resample monthly to quarterly mean
            q_s = s.resample("QE").mean().to_period("Q")
        
        quarterly_series[sid] = q_s

    panel_df = pd.DataFrame(quarterly_series)
    # Sort index
    panel_df = panel_df.sort_index()

    # Drop quarters before 1954 Q3 where FEDFUNDS begins (or 1953 Q2 for GS10)
    # We want complete macroeconomic predictors across the shared historical panel
    # Let's inspect where all predictors are non-null
    complete_start = panel_df.dropna().index.min()
    print(f"Complete predictor series available from: {complete_start}")

    curated_path = os.path.join(DATA_DIR, "curated_macro_quarterly.csv")
    panel_df.to_csv(curated_path)
    print(f"Saved curated quarterly panel to {curated_path} (shape: {panel_df.shape})")

    # Build fallback statsmodels data
    build_fallback_statsmodels()

    # Build data manifest
    manifest = {
        "dataset_name": "MacroCycle Research Historical Panel",
        "retrieval_date_utc": datetime.now(timezone.utc).isoformat(),
        "primary_source": "Federal Reserve Bank of St. Louis (FRED) Public Data",
        "fallback_source": "Statsmodels Macrodata (1959Q1-2009Q3)",
        "frequency": "Quarterly (resampled from monthly using quarterly arithmetic mean; GDP native quarterly)",
        "revision_status": "Revised historical series (subject to data revision bias; not real-time vintages)",
        "reference_vintage_documentation": "https://alfred.stlouisfed.org/help",
        "sample_range": {
            "first_quarter": str(panel_df.dropna().index.min()),
            "last_quarter": str(panel_df.dropna().index.max()),
            "total_quarters": int(len(panel_df.dropna())),
        },
        "series": {
            sid: {
                **meta,
                "first_date": str(raw_dfs[sid].iloc[0, 0].date()),
                "last_date": str(raw_dfs[sid].iloc[-1, 0].date()),
                "raw_observations": int(len(raw_dfs[sid])),
            }
            for sid, meta in SERIES_META.items()
        },
        "transformations": {
            "gdp_growth_qoq_ann": "400 * [log(GDP_t) - log(GDP_{t-1})] (Annualized Quarter-on-Quarter Growth)",
            "cpi_inflation_yoy": "100 * [CPI_t / CPI_{t-4} - 1] (Year-over-Year CPI Inflation)",
            "cpi_inflation_qoq_ann": "400 * [log(CPI_t) - log(CPI_{t-1})] (Annualized QoQ Inflation)",
            "unemployment_rate": "Level in percent (UNRATE)",
            "unemployment_change_qoq": "UNRATE_t - UNRATE_{t-1} (Quarterly Change in Percentage Points)",
            "industrial_production_growth_qoq_ann": "400 * [log(INDPRO_t) - log(INDPRO_{t-1})] (Annualized QoQ IP Growth)",
            "policy_rate": "Level in percent (FEDFUNDS)",
            "term_spread_10y_3m": "GS10 - TB3MS (10-Year Treasury Yield minus 3-Month T-Bill Rate, Percentage Points)",
        },
        "missing_data_treatment": "No forward filling across quarterly macro series; strict complete-case alignment for walk-forward evaluation",
    }

    manifest_path = os.path.join(DATA_DIR, "data_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    run()
