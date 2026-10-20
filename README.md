# MacroCycle: Growth, Inflation and Yield-Curve Research Lab

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/streamlit-1.40%2B-red.svg)](https://streamlit.io/)
[![Testing Status](https://img.shields.io/badge/pytest-27%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-slate.svg)]()

**MacroCycle** is an empirical quantitative-macroeconomic research application and fixed-income scenario lab built using Python, Streamlit, Plotly, pandas, and scikit-learn. It is designed to evaluate out-of-sample predictability of macroeconomic output and inflation, identify historical growth/inflation regimes using strictly non-anticipating benchmarks, and provide a transparent, mathematically decoupled local duration-convexity interest-rate sensitivity calculator.

---

## 1. Core Research Question & Central Hypotheses

### Central Research Question
> *Does incorporating multivariate macroeconomic predictors (consumer inflation, labor market slack, industrial output, the effective federal funds rate, and the 10-year minus 3-month Treasury term spread) yield genuine, statistically significant out-of-sample forecasting gains over simple historical-mean and univariate autoregressive baselines for U.S. real GDP growth?*

### Formal Hypotheses
Let $y_{t+1}$ denote realized annualized quarter-on-quarter real GDP growth at quarter $t+1$, and let $\hat{y}_{t+1}^{\text{Model}}$ denote the out-of-sample forecast generated using information available through quarter $t$. Under squared error loss $L(y, \hat{y}) = (y - \hat{y})^2$:

- **Null Hypothesis ($H_0$):** Multivariate macroeconomic models fail to outperform the prevailing expanding historical mean benchmark out-of-sample:
  $$\mathbb{E}\left[L\left(y_{t+1}, \hat{y}_{t+1}^{\text{Model}}\right)\right] \ge \mathbb{E}\left[L\left(y_{t+1}, \hat{y}_{t+1}^{\text{Hist Mean}}\right)\right]$$
- **Alternative Hypothesis ($H_1$):** Multivariate macroeconomic models achieve strictly lower expected out-of-sample loss:
  $$\mathbb{E}\left[L\left(y_{t+1}, \hat{y}_{t+1}^{\text{Model}}\right)\right] < \mathbb{E}\left[L\left(y_{t+1}, \hat{y}_{t+1}^{\text{Hist Mean}}\right)\right]$$

### Falsification Standard
The alternative hypothesis $H_1$ is **empirically falsified** if:
1. The out-of-sample $R^2$ relative to the expanding training mean benchmark is non-positive ($R^2_{\text{OOS}} \le 0$):
   $$R^2_{\text{OOS}} = 1 - \frac{\sum_{t=1}^N (y_t - \hat{y}_t^{\text{Model}})^2}{\sum_{t=1}^N (y_t - \bar{y}_{\text{train}, t})^2} \le 0$$
2. The 95% moving block bootstrap confidence interval for the mean squared error differential $d_t = (y_t - \hat{y}_t^{\text{Hist Mean}})^2 - (y_t - \hat{y}_t^{\text{Model}})^2$ spans zero or is strictly negative ($p \ge 0.05$).

---

## 2. Empirical Findings & Result Interpretation

Evaluating 247 quarterly out-of-sample forecasts from **1964 Q3 through 2026 Q1** (with a 40-quarter initial training window) yields the following empirical results:

| Model | MAE (pp) | RMSE (pp) | Out-of-Sample $R^2$ vs Training Mean | Directional Accuracy (%) | Mean Bias (pp) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Historical Mean Benchmark** | 2.487 | 4.251 | 0.0000 | 88.3% | +0.000 |
| **Autoregressive AR(1)** | 2.443 | 4.530 | -0.1360 | 88.3% | -0.063 |
| **Macro Ridge Regression ($\alpha=10$)** | 2.609 | 4.490 | -0.1159 | 86.6% | -0.218 |
| **Constrained Gradient Boosting (GBR)** | 2.443 | 4.262 | -0.0056 | 87.4% | -0.088 |

### Paired Moving Block Bootstrap Hypothesis Test (Ridge vs. Historical Mean)
- **Mean Loss Differential ($\Delta \text{MSE}$):** $-2.095 \text{ pp}^2$ (negative denotes baseline has lower MSE)
- **95% Block Bootstrap Confidence Interval ($b=4$ quarters):** $[-7.232, +1.980] \text{ pp}^2$
- **Empirical $p$-value:** $0.540$
- **Verdict:** We fail to reject the null hypothesis $H_0$. There is no statistically significant evidence that the multivariate Ridge model beats the prevailing historical mean out-of-sample.

### Institutional Interpretation
This finding directly replicates the established consensus in modern empirical macroeconomics (e.g., Stock & Watson 2003, Faust & Wright 2013). Quarterly real GDP growth has an extremely low signal-to-noise ratio and exhibits recurring structural shifts (such as the transition into the Great Moderation and post-2008 policy regimes). In out-of-sample walk-forward tests, parameter estimation error in multivariate models often offsets any incremental theoretical predictive content, making simple expanding-mean baselines extraordinarily competitive.

---

## 3. Data Strategy & Provenance

The application is completely self-contained and operates **100% offline** without runtime web requests, external database requirements, or API keys.

### Bundled Historical Datasets (`data/`)
- `data/curated_macro_quarterly.csv`: Curated quarterly panel spanning **1947 Q1 to 2026 Q2** (complete predictors available from **1954 Q3 to 2026 Q2**, 288 quarters).
- `data/statsmodels_macrodata_fallback.csv`: Secondary benchmark panel extracted from `statsmodels.datasets.macrodata` (**1959 Q1 to 2009 Q3**, 203 quarters).
- `data/raw/`: Original downloadable series files for `GDPC1`, `CPIAUCSL`, `UNRATE`, `FEDFUNDS`, `INDPRO`, `GS10`, and `TB3MS`.
- `data/data_manifest.json`: Manifest recording series metadata, URLs, retrieval dates, and transformation formulas.

### Variable Definitions & Transformations
| Variable Symbol | Underlying FRED Series | Source Agency | Frequency & Resampling | Transformation Formula |
|---|---|---|---|---|
| `GDP_growth_t` | `GDPC1` | U.S. Bureau of Economic Analysis | Quarterly (native) | $400 \times [\ln(\text{GDP}_t) - \ln(\text{GDP}_{t-1})]$ |
| `CPI_yoy_t` | `CPIAUCSL` | U.S. Bureau of Labor Statistics | Monthly $\to$ Quarterly Mean | $100 \times [\text{CPI}_t / \text{CPI}_{t-4} - 1]$ |
| `CPI_qoq_t` | `CPIAUCSL` | U.S. Bureau of Labor Statistics | Monthly $\to$ Quarterly Mean | $400 \times [\ln(\text{CPI}_t) - \ln(\text{CPI}_{t-1})]$ |
| `UNRATE_t` | `UNRATE` | U.S. Bureau of Labor Statistics | Monthly $\to$ Quarterly Mean | Level in percent |
| `ΔUNRATE_t` | `UNRATE` | U.S. Bureau of Labor Statistics | Monthly $\to$ Quarterly Mean | $\text{UNRATE}_t - \text{UNRATE}_{t-1}$ |
| `IP_growth_t` | `INDPRO` | Federal Reserve Board | Monthly $\to$ Quarterly Mean | $400 \times [\ln(\text{IP}_t) - \ln(\text{IP}_{t-1})]$ |
| `FEDFUNDS_t` | `FEDFUNDS` | Federal Reserve Board | Monthly $\to$ Quarterly Mean | Level in percent per annum |
| `Spread_t` | `GS10`, `TB3MS` | Federal Reserve Board | Monthly $\to$ Quarterly Mean | $\text{GS10}_t - \text{TB3MS}_t$ (percentage points) |

### Retrospective Revised-Data Notice & ALFRED Documentation
> [!WARNING]
> **Data Revision Bias:** The bundled historical series represent the *latest revised historical values* (post-benchmark revisions and updated seasonal adjustment factors). Real-time forecasters at quarter $t$ only possessed preliminary (Advance) prints. Applying a 1-quarter feature lag aligns information timing but **does not eliminate data revision bias**. For real-time vintages, see the [ALFRED (ArchivaL Federal Reserve Economic Data) Documentation](https://alfred.stlouisfed.org/help).

---

## 4. Methodological Architecture

### 1. Chronological Walk-Forward Expanding Window Protocol
- Total available quarters: $t = 1, \dots, T$.
- Initial training window: $M$ quarters (default $M = 40$ quarters = 10 years).
- For step $k \in \{M, \dots, T-1\}$:
  1. Training features $X_{1:k-1}$ and target realizations $y_{2:k}$ are isolated.
  2. Feature standardizers and model parameters are fitted **exclusively on the training window**.
  3. Feature vector $X_k$ is standardized using training parameters and fed to the model to predict $\hat{y}_{k+1}$.
  4. Realized $y_{k+1}$ is recorded, producing out-of-sample residual $e_{k+1} = y_{k+1} - \hat{y}_{k+1}$.
- **Zero Lookahead Guarantee:** Every test prediction occurs strictly after all observations used to fit the model and scaler.

### 2. Descriptive Growth/Inflation Regimes
Regimes are classified without lookahead by comparing quarter $t$ growth and inflation against **trailing expanding-window medians** computed strictly from quarters $1$ through $t-1$:
- **Expansion:** Growth $\ge \text{Median}_{1:t-1}$, Inflation $< \text{Median}_{1:t-1}$
- **Reflation:** Growth $\ge \text{Median}_{1:t-1}$, Inflation $\ge \text{Median}_{1:t-1}$
- **Stagflation:** Growth $< \text{Median}_{1:t-1}$, Inflation $\ge \text{Median}_{1:t-1}$
- **Slowdown:** Growth $< \text{Median}_{1:t-1}$, Inflation $< \text{Median}_{1:t-1}$

*Notice:* These are descriptive empirical states, not official NBER recession dates or causal indicators.

### 3. Decoupled Rates Scenario Engine
Interest rate sensitivity is modeled via a second-order Taylor series local duration-convexity expansion:
$$\frac{\Delta P}{P} \approx \underbrace{-D \cdot \Delta y}_{\text{Linear Duration}} + \underbrace{\frac{1}{2} C \cdot (\Delta y)^2}_{\text{Convexity Adjustment}}$$
where $D$ is modified duration, $C$ is convexity, and $\Delta y = \Delta y_{\text{bps}} / 10{,}000$.

*Notice:* This calculator is **mathematically decoupled from the GDP model**. Bond returns are not inferred from GDP forecasts unless a validated asset-pricing model is specified.

---

## 5. Installation & Execution

### Prerequisites
- Python 3.11 or 3.12
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/MacroCycle.git
cd MacroCycle
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Test Suite
```bash
python -m pytest
```
*All 27 unit and integration tests run in ~13 seconds.*

### 5. Launch the Streamlit Research Lab
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 6. Project Structure

```
P2_MacroCycle/
│
├── .streamlit/
│   └── config.toml               # Institutional navy/slate UI theme configuration
├── data/
│   ├── raw/                      # Downloaded original series files from FRED
│   │   ├── GDPC1.csv
│   │   ├── CPIAUCSL.csv
│   │   ├── UNRATE.csv
│   │   ├── FEDFUNDS.csv
│   │   ├── INDPRO.csv
│   │   ├── GS10.csv
│   │   └── TB3MS.csv
│   ├── curated_macro_quarterly.csv            # Standardized quarterly research panel (1947–2026)
│   ├── statsmodels_macrodata_fallback.csv     # Fallback benchmark panel (1959–2009)
│   └── data_manifest.json                     # Metadata, sources, and transformation manifest
│
├── macrocycle/
│   ├── __init__.py
│   ├── data_pipeline.py          # Data ingestion, quarterly resampling, formulas, alignment
│   ├── models.py                 # MeanModel, AR(1), Ridge with training scaler, Constrained GBR
│   ├── evaluation.py             # Expanding-window walk-forward engine, OOS R², block bootstrap
│   ├── regimes.py                # Trailing-median 4-quadrant classification & error analysis
│   ├── fixed_income.py           # Duration-convexity Taylor calculator & scenario analysis
│   └── plotting.py               # Institutional Plotly charts and color palettes
│
├── tests/
│   ├── __init__.py
│   ├── test_app_integration.py   # Full Streamlit AppTest integration verification
│   ├── test_data_pipeline.py     # Formulas, edge cases, alignment tests
│   ├── test_models.py            # Model fitting, coefficients, scaler isolation
│   ├── test_evaluation.py        # Walk-forward chronological order, metrics, bootstrap
│   ├── test_regimes.py           # Non-anticipating median thresholds, sample flags
│   └── test_fixed_income.py      # Duration, convexity asymmetry, negative/positive shocks
│
├── scripts/
│   └── fetch_and_prepare_data.py # Development data acquisition and preprocessing script
│
├── app.py                        # Institutional Streamlit application (6 research tabs)
├── pyproject.toml                # Pytest configuration and pythonpath setup
├── requirements.txt              # Pinned production dependencies
├── .gitignore                    # Python, OS, and environment ignores
└── README.md                     # Comprehensive research documentation
```

---

## 7. Automated Test Suite Coverage

The automated test suite verifies 27 distinct properties:
- **Data Transformations:** Accurate log-difference annualized GDP and IP growth, YoY and QoQ inflation, and term spread.
- **Chronological Alignment:** Zero future lookahead between feature periods and target realization periods.
- **Data Fallbacks:** Flawless fallback dataset ingestion and structure.
- **Model Isolation:** StandardScalers and Ridge estimators refitted strictly on training slices without leakage.
- **Evaluation Integrity:** Verification that the first evaluation occurs only after $M$ periods and subsequent test steps are strictly out-of-sample.
- **Statistical Testing:** Moving block bootstrap loss differential estimation and empirical $p$-value calibration.
- **Fixed-Income Mathematics:** Accurate linear duration and convexity contributions, verifying positive convexity cushion under symmetric $\pm 100 \text{ bps}$ shifts.
- **App Integration:** Streamlit `AppTest` execution of the complete 6-tab interface without runtime errors.

---

## 8. Disclaimer & Scope of Research

This software is developed strictly for macroeconomic and fixed-income academic and quantitative research. It:
1. **Does not provide real-time nowcasts:** Historical series reflect revised BEA/BLS data.
2. **Does not make causal assertions:** Historical relationships between the yield curve and output growth are descriptive and cyclical, not structural guarantees.
3. **Does not constitute investment advice:** Hypothetical fixed-income price changes derive from local mathematical approximations and do not represent trading strategies, expected returns, or financial recommendations.
