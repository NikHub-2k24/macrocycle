"""
Unit tests for macrocycle.models
"""

import numpy as np
import pytest

from macrocycle.models import (
    AutoregressiveBaselineModel,
    ConstrainedGBRModel,
    HistoricalMeanModel,
    RidgeMacroModel,
)


def test_historical_mean_model():
    """Verify HistoricalMeanModel computes training sample mean and predicts constant."""
    model = HistoricalMeanModel()
    X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_train = np.array([2.0, 4.0, 6.0])

    model.fit(X_train, y_train)
    preds = model.predict(np.array([[10.0, 20.0], [30.0, 40.0]]))
    assert len(preds) == 2
    assert np.allclose(preds, 4.0)
    coeffs = model.get_coefficients()
    assert coeffs is not None
    assert coeffs["Intercept (Historical Mean)"] == 4.0


def test_autoregressive_ar1_model():
    """Verify Autoregressive AR(1) OLS fitting."""
    model = AutoregressiveBaselineModel(target_col_idx=0)
    # y = 1.0 + 0.5 * y_{lag}
    x_lag = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    X_train = x_lag.reshape(-1, 1)
    y_train = 1.0 + 0.5 * x_lag

    model.fit(X_train, y_train)
    pred = model.predict(np.array([[10.0]]))
    assert np.isclose(pred[0], 6.0, atol=1e-5)
    coeffs = model.get_coefficients()
    assert coeffs is not None
    assert np.isclose(coeffs["Intercept"], 1.0, atol=1e-5)
    assert np.isclose(coeffs["AR(1) Lagged Target Coeff"], 0.5, atol=1e-5)


def test_ridge_training_only_scaling():
    """Verify Ridge scales using training data only, without leakage."""
    model = RidgeMacroModel(alpha=1.0)
    rng = np.random.default_rng(123)
    X_train = rng.normal(loc=100.0, scale=15.0, size=(50, 4))
    y_train = X_train[:, 0] * 0.5 + rng.normal(size=50)

    model.fit(X_train, y_train, feature_names=["F1", "F2", "F3", "F4"])
    # Scaler mean must match training mean
    np.testing.assert_allclose(model.scaler.mean_, np.mean(X_train, axis=0))

    X_test = rng.normal(loc=200.0, scale=15.0, size=(5, 4))
    preds = model.predict(X_test)
    assert len(preds) == 5
    assert not np.isnan(preds).any()


def test_constrained_gbr_reproducibility():
    """Verify ConstrainedGBRModel reproducibility with fixed random seed."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 3))
    y = X[:, 0] * 1.5 - X[:, 1] * 0.8 + rng.normal(scale=0.2, size=60)

    m1 = ConstrainedGBRModel(random_state=42)
    m1.fit(X, y)
    p1 = m1.predict(X[:5])

    m2 = ConstrainedGBRModel(random_state=42)
    m2.fit(X, y)
    p2 = m2.predict(X[:5])

    np.testing.assert_allclose(p1, p2)


def test_model_unfitted_raises():
    """Verify predicting on unfitted model raises RuntimeError."""
    m = HistoricalMeanModel()
    with pytest.raises(RuntimeError):
        m.predict(np.array([[1.0]]))

    r = RidgeMacroModel()
    with pytest.raises(RuntimeError):
        r.predict(np.array([[1.0, 2.0]]))
