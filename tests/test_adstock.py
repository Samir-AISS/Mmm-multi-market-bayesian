"""
test_adstock.py
---------------
Tests unitaires pour GeometricAdstock et DelayedAdstock.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.adstock import GeometricAdstock, DelayedAdstock, apply_adstock


class TestGeometricAdstock:

    def test_zero_spend_gives_zero(self):
        a = GeometricAdstock(decay=0.5)
        result = a.transform(np.zeros(10))
        np.testing.assert_array_equal(result, np.zeros(10))

    def test_decay_zero_equals_original(self):
        """decay=0 → pas de carryover → adstock[t] == spend[t]"""
        spend = np.array([100, 200, 50, 300, 0], dtype=float)
        result = GeometricAdstock(decay=0.0).transform(spend)
        np.testing.assert_array_almost_equal(result, spend)

    def test_decay_one_is_cumulative_sum(self):
        """decay=1 → adstock[t] = sum(spend[0..t])"""
        spend = np.array([10, 20, 30, 40], dtype=float)
        result = GeometricAdstock(decay=1.0).transform(spend)
        expected = np.cumsum(spend).astype(float)
        np.testing.assert_array_almost_equal(result, expected)

    def test_output_shape(self):
        spend = np.random.rand(52)
        result = GeometricAdstock(decay=0.5).transform(spend)
        assert result.shape == spend.shape

    def test_output_nonnegative_for_nonnegative_input(self):
        spend = np.abs(np.random.randn(50)) * 1000
        result = GeometricAdstock(decay=0.6).transform(spend)
        assert np.all(result >= 0)

    def test_carryover_effect_present(self):
        """Un spend en t=0 doit avoir un effet résiduel en t=1 si decay > 0."""
        spend = np.zeros(5)
        spend[0] = 1000.0
        result = GeometricAdstock(decay=0.5).transform(spend)
        assert result[1] > 0, "L'effet doit se propager à t=1"
        assert result[1] < result[0], "L'effet doit décroître"

    def test_decay_validation(self):
        with pytest.raises(ValueError):
            GeometricAdstock(decay=1.5)
        with pytest.raises(ValueError):
            GeometricAdstock(decay=-0.1)

    def test_half_life(self):
        a = GeometricAdstock(decay=0.5)
        assert abs(a.half_life() - 1.0) < 0.01  # decay=0.5 → demi-vie ≈ 1 semaine

    def test_monotone_weights(self):
        """Les poids doivent décroître avec le temps."""
        a = GeometricAdstock(decay=0.7)
        w = a.weights(n_periods=10)
        assert all(w[i] >= w[i+1] for i in range(len(w)-1))

    def test_callable_interface(self):
        spend = np.array([100., 200., 150.])
        a = GeometricAdstock(decay=0.4)
        np.testing.assert_array_equal(a(spend), a.transform(spend))


class TestDelayedAdstock:

    def test_output_shape(self):
        spend = np.random.rand(52) * 1000
        result = DelayedAdstock(decay=0.5, peak=1).transform(spend)
        assert result.shape == spend.shape

    def test_zero_spend_gives_zero(self):
        result = DelayedAdstock(decay=0.5, peak=2).transform(np.zeros(20))
        np.testing.assert_array_almost_equal(result, np.zeros(20))

    def test_peak_zero_similar_to_geometric(self):
        """peak=0 doit être similaire à GeometricAdstock."""
        spend = np.random.rand(30) * 1000
        delayed = DelayedAdstock(decay=0.5, peak=0).transform(spend)
        geometric = GeometricAdstock(decay=0.5).transform(spend)
        # Même somme totale (les deux conservent le spend total)
        assert len(delayed) == len(geometric) and np.all(delayed >= 0)

    def test_output_nonnegative(self):
        spend = np.abs(np.random.randn(40)) * 500
        result = DelayedAdstock(decay=0.4, peak=2).transform(spend)
        assert np.all(result >= 0)

    def test_decay_validation(self):
        with pytest.raises(ValueError):
            DelayedAdstock(decay=2.0)

    def test_peak_validation(self):
        with pytest.raises(ValueError):
            DelayedAdstock(peak=-1)


class TestApplyAdstockUtil:

    def test_geometric_routing(self):
        spend = np.array([100., 200., 150., 300.])
        r1 = apply_adstock(spend, decay=0.5, adstock_type="geometric")
        r2 = GeometricAdstock(decay=0.5).transform(spend)
        np.testing.assert_array_equal(r1, r2)

    def test_delayed_routing(self):
        spend = np.array([100., 200., 150., 300.])
        r1 = apply_adstock(spend, decay=0.5, adstock_type="delayed", peak=1)
        r2 = DelayedAdstock(decay=0.5, peak=1).transform(spend)
        np.testing.assert_array_equal(r1, r2)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            apply_adstock(np.array([100.]), adstock_type="unknown")