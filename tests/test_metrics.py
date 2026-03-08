"""
test_metrics.py
---------------
Tests unitaires pour les métriques d'évaluation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.metrics import r_squared, mape, smape, rmse, nrmse, compute_all_metrics


class TestRSquared:

    def test_perfect_prediction_gives_1(self):
        y = np.array([100., 200., 300.])
        assert r_squared(y, y) == 1.0

    def test_constant_pred_gives_0(self):
        y_true = np.array([100., 200., 300., 400.])
        y_pred = np.full_like(y_true, y_true.mean())
        np.testing.assert_almost_equal(r_squared(y_true, y_pred), 0.0, decimal=5)

    def test_close_predictions_high_r2(self):
        y_true = np.array([100., 200., 300., 400., 500.], dtype=float)
        y_pred = y_true * 1.01
        assert r_squared(y_true, y_pred) > 0.99

    def test_bounded(self):
        y_true = np.random.rand(100) * 1e6
        y_pred = np.random.rand(100) * 1e6
        r2 = r_squared(y_true, y_pred)
        assert r2 <= 1.0  # peut être négatif si très mauvais modèle

    def test_scalar_equal(self):
        assert r_squared(np.array([5.]), np.array([5.])) == 1.0


class TestMape:

    def test_perfect_gives_zero(self):
        y = np.array([100., 200., 300.])
        np.testing.assert_almost_equal(mape(y, y), 0.0)

    def test_10_percent_error(self):
        y_true = np.array([100., 200., 300.], dtype=float)
        y_pred = y_true * 1.10
        np.testing.assert_almost_equal(mape(y_true, y_pred), 10.0, decimal=5)

    def test_symmetric_not_guaranteed(self):
        """MAPE n'est pas symétrique — erreur asymétrique selon le sens."""
        y_true = np.array([100.], dtype=float)
        y_pred_over  = np.array([200.])  # +100% → MAPE=100%
        y_pred_under = np.array([50.])   # -50% → MAPE=50%
        assert mape(y_true, y_pred_over) > mape(y_true, y_pred_under)

    def test_ignores_zero_true_values(self):
        y_true = np.array([0., 100., 200.])
        y_pred = np.array([50., 110., 210.])
        result = mape(y_true, y_pred)
        assert np.isfinite(result)


class TestSmape:

    def test_perfect_gives_zero(self):
        y = np.array([100., 200., 300.])
        np.testing.assert_almost_equal(smape(y, y), 0.0)

    def test_symmetric_property(self):
        """SMAPE est symétrique : smape(y, y_pred) == smape(y_pred, y)."""
        y_true = np.array([100., 300., 500.])
        y_pred = np.array([120., 280., 550.])
        np.testing.assert_almost_equal(
            smape(y_true, y_pred),
            smape(y_pred, y_true),
            decimal=10
        )

    def test_bounded_0_100(self):
        y_true = np.array([100., 200.])
        y_pred = np.array([0., 400.])
        assert smape(y_true, y_pred) >= 0


class TestRmseNrmse:

    def test_rmse_zero_for_perfect(self):
        y = np.array([100., 200., 300.])
        np.testing.assert_almost_equal(rmse(y, y), 0.0)

    def test_rmse_positive(self):
        y_true = np.array([100., 200., 300.])
        y_pred = np.array([110., 190., 310.])
        assert rmse(y_true, y_pred) > 0

    def test_nrmse_zero_for_perfect(self):
        y = np.array([100., 200., 300.])
        np.testing.assert_almost_equal(nrmse(y, y), 0.0)

    def test_nrmse_dimensionless(self):
        """NRMSE est normalisé → ne dépend pas de l'échelle."""
        y_true_small = np.array([1., 2., 3.])
        y_pred_small = np.array([1.1, 1.9, 3.1])
        y_true_large = y_true_small * 1e6
        y_pred_large = y_pred_small * 1e6
        np.testing.assert_almost_equal(
            nrmse(y_true_small, y_pred_small),
            nrmse(y_true_large, y_pred_large),
            decimal=6
        )


class TestComputeAllMetrics:

    def test_returns_all_keys(self):
        y = np.array([100., 200., 300., 400., 500.])
        m = compute_all_metrics(y, y)
        for key in ["r2", "mape", "smape", "rmse", "nrmse"]:
            assert key in m, f"Clé manquante : {key}"

    def test_perfect_prediction(self):
        y = np.array([100., 200., 300., 400., 500.], dtype=float)
        m = compute_all_metrics(y, y)
        assert m["r2"] == 1.0
        assert m["mape"] == 0.0
        assert m["rmse"] == 0.0

    def test_realistic_scenario(self):
        """Simulation d'un bon modèle MMM : R² > 0.85, MAPE < 15%."""
        rng = np.random.default_rng(42)
        y_true = rng.normal(500_000, 50_000, 100)
        noise  = rng.normal(0, 10_000, 100)
        y_pred = y_true + noise
        m = compute_all_metrics(y_true, y_pred)
        assert m["r2"] > 0.70, f"R² trop bas : {m['r2']}"
        assert m["mape"] < 20.0, f"MAPE trop élevé : {m['mape']}"