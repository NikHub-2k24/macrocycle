"""
Forecasting Models for MacroCycle Research Lab.

Implements benchmark and challenger models strictly conforming to expanding-window
walk-forward training:
1. Historical Mean Forecast (Zero-parameter expanding mean)
2. Autoregressive Baseline AR(1) / AR(p) (Univariate lagged benchmark)
3. Ridge Regression with training-only standardization
4. Constrained Gradient Boosting Challenger (Nonlinear tree ensemble)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class BaseMacroModel(ABC):
    """Abstract base class for all macro forecasting models."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None):
        """Fit model strictly on provided training observations."""
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict for out-of-sample features."""
        pass

    def get_coefficients(self) -> Optional[Dict[str, float]]:
        """Return model coefficients or feature importances if available."""
        return None


class HistoricalMeanModel(BaseMacroModel):
    """
    Historical-mean forecast benchmark.
    Predicts the expanding training sample mean of the target variable:
    y_hat_{t+1} = mean(y_{1...t})
    """

    def __init__(self):
        super().__init__(name="Historical Mean")
        self.mean_val: float = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None):
        if len(y_train) == 0:
            raise ValueError("Cannot fit HistoricalMeanModel on empty training set.")
        self.mean_val = float(np.mean(y_train))
        self.is_fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        n_samples = len(X_test) if hasattr(X_test, "__len__") else 1
        return np.full(n_samples, self.mean_val, dtype=float)

    def get_coefficients(self) -> Optional[Dict[str, float]]:
        return {"Intercept (Historical Mean)": self.mean_val}


class AutoregressiveBaselineModel(BaseMacroModel):
    """
    Univariate Autoregressive AR(1) baseline using only lagged target growth:
    y_{t+1} = beta_0 + beta_1 * y_t + eps_{t+1}
    
    When features X_train are supplied, the first column (or column matching target name)
    is used as the lagged target.
    """

    def __init__(self, target_col_idx: int = 0):
        super().__init__(name="Autoregressive AR(1)")
        self.target_col_idx = target_col_idx
        self.beta_0: float = 0.0
        self.beta_1: float = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None):
        if len(y_train) < 5:
            raise ValueError("Insufficient observations to fit AR(1) model.")
        
        # Extract the lagged target predictor (default column 0: lagged GDP growth)
        if X_train.ndim == 1:
            x_lag = X_train
        else:
            x_lag = X_train[:, self.target_col_idx]

        # Fit OLS: y = beta_0 + beta_1 * x_lag
        X_mat = np.column_stack([np.ones_like(x_lag), x_lag])
        # Solve least squares via pinv for numerical stability
        betas, _, _, _ = np.linalg.lstsq(X_mat, y_train, rcond=None)
        self.beta_0 = float(betas[0])
        self.beta_1 = float(betas[1])
        self.is_fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        if X_test.ndim == 1:
            x_lag = X_test
        else:
            x_lag = X_test[:, self.target_col_idx]
        return self.beta_0 + self.beta_1 * x_lag

    def get_coefficients(self) -> Optional[Dict[str, float]]:
        return {
            "Intercept": self.beta_0,
            "AR(1) Lagged Target Coeff": self.beta_1,
        }


class RidgeMacroModel(BaseMacroModel):
    """
    Multivariate Ridge regression using standardized macroeconomic predictors.
    StandardScaler is fit strictly on training data X_train to prevent future lookahead.
    
    Objective:
    min_{beta} || y - X beta ||^2_2 + alpha * || beta ||^2_2
    """

    def __init__(self, alpha: float = 10.0, fit_intercept: bool = True):
        super().__init__(name="Macro Ridge Regression")
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
        self.feature_names: List[str] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None):
        if len(y_train) < X_train.shape[1] + 2:
            raise ValueError(f"Insufficient observations ({len(y_train)}) for {X_train.shape[1]} features.")
        
        self.feature_names = feature_names if feature_names is not None else [f"Feature_{i}" for i in range(X_train.shape[1])]
        # Fit scaler ONLY on training features
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        # Transform test features using the training-only scaler parameters
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

    def get_coefficients(self) -> Optional[Dict[str, float]]:
        if not self.is_fitted:
            return None
        coeffs = {"Intercept": float(self.model.intercept_)}
        for name, coef in zip(self.feature_names, self.model.coef_):
            coeffs[name] = float(coef)
        return coeffs


class ConstrainedGBRModel(BaseMacroModel):
    """
    Constrained Gradient Boosting Regressor challenger.
    Uses conservative tree depth (max_depth=2), high minimum leaf samples,
    and a small learning rate to avoid overfitting on limited quarterly samples.
    """

    def __init__(
        self,
        n_estimators: int = 35,
        max_depth: int = 2,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 6,
        random_state: int = 42,
    ):
        super().__init__(name="Constrained Gradient Boosting")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.feature_names: List[str] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: Optional[List[str]] = None):
        if len(y_train) < 20:
            raise ValueError(f"Insufficient observations ({len(y_train)}) for Gradient Boosting.")
        self.feature_names = feature_names if feature_names is not None else [f"Feature_{i}" for i in range(X_train.shape[1])]
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return self.model.predict(X_test)

    def get_coefficients(self) -> Optional[Dict[str, float]]:
        if not self.is_fitted:
            return None
        importances = {}
        for name, imp in zip(self.feature_names, self.model.feature_importances_):
            importances[name] = float(imp)
        return importances
