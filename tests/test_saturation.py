"""
test_saturation.py
------------------
Tests unitaires pour HillSaturation et LogisticSaturation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.saturation import HillSaturation, LogisticSaturation, apply_saturation


class TestHillSaturation:

    def test_output_bounded_0_1(self):
        """Hill(x) ∈ [0, 1] pour tout x ≥ 0."""
        x = np.linspace(0, 1_000_000, 1000)
        result = HillSaturation(K=100_000, S=2.0).transform(x)
        assert np.all(result >= 0)
        assert np.all(result <= 1.0 + 1e-9)

    def test_zero_input_gives_zero(self):
        result = HillSaturation(K=1000, S=2.0).transform(np.array([0.0]))
        np.testing.assert_almost_equal(result[0], 0.0)

    def test_half_saturation_at_K(self):
        """Hill(K) doit être exactement 0.5."""
        K = 50_000.0
        result = HillSaturation(K=K, S=2.0).transform(np.array([K]))
        np.testing.assert_almost_equal(result[0], 0.5, decimal=6)

    def test_monotonically_increasing(self):
        x = np.linspace(0, 500_000, 500)
        result = HillSaturation(K=100_000, S=2.0).transform(x)
        diffs = np.diff(result)
        assert np.all(diffs >= -1e-12), "Hill doit être monotone croissante"

    def test_asymptote_approaches_1(self):
        """Pour x très grand, Hill(x) → 1."""
        x_large = np.array([1e12])
        result = HillSaturation(K=1000, S=2.0).transform(x_large)
        np.testing.assert_almost_equal(result[0], 1.0, decimal=4)

    def test_shape_S_less_than_1_concave(self):
        """S < 1 → courbe concave (croît vite puis ralentit)."""
        x = np.array([1e3, 1e4, 1e5], dtype=float)
        result = HillSaturation(K=1e4, S=0.5).transform(x)
        assert result[0] < result[1] < result[2]

    def test_shape_S_greater_than_1_sigmoidal(self):
        """S > 1 → courbe sigmoïdale."""
        x = np.linspace(0, 200_000, 1000)
        result = HillSaturation(K=100_000, S=3.0).transform(x)
        # Première moitié : croissance lente → deuxième moitié : croissance rapide jusqu'à K
        first_half_gain  = result[499] - result[0]
        second_half_gain = result[999] - result[499]
        # Pour un S=3 sigmoïde, la moitié droite (autour du pic) a plus de gain
        assert result[-1] > result[0]

    def test_output_shape(self):
        x = np.random.rand(100) * 1e5
        result = HillSaturation(K=50_000, S=2.0).transform(x)
        assert result.shape == x.shape

    def test_auto_K_from_median(self):
        """K=None → utilise la médiane comme half-saturation."""
        x = np.ones(100) * 50_000.0
        h = HillSaturation(K=None, S=2.0)
        result = h.transform(x)
        # Tous les points = K (médiane) → tous à 0.5
        np.testing.assert_array_almost_equal(result, np.full(100, 0.5), decimal=5)

    def test_invalid_K_raises(self):
        with pytest.raises(ValueError):
            HillSaturation(K=-100)

    def test_invalid_S_raises(self):
        with pytest.raises(ValueError):
            HillSaturation(S=0)

    def test_marginal_return_decreasing(self):
        """Le ROI marginal doit décroître (rendements décroissants)."""
        h = HillSaturation(K=100_000, S=2.0)
        x = np.linspace(10_000, 500_000, 100)
        mr = h.marginal_return(x)
        # Après le pic, les retours marginaux décroissent
        assert mr[-1] < mr[0]

    def test_response_curve_returns_tuple(self):
        h = HillSaturation(K=100_000, S=2.0)
        x_vals, y_vals = h.response_curve(x_max=500_000, n_points=50)
        assert len(x_vals) == 50
        assert len(y_vals) == 50
        assert x_vals[0] == 0.0


class TestLogisticSaturation:

    def test_output_positive(self):
        x = np.linspace(0, 1_000_000, 100)
        result = LogisticSaturation(L=1.0, k=0.00001, x0=500_000).transform(x)
        assert np.all(result >= 0)

    def test_bounded_by_L(self):
        L = 2.0
        x = np.linspace(0, 1e8, 1000)
        result = LogisticSaturation(L=L, k=0.00001, x0=500_000).transform(x)
        assert np.all(result <= L + 1e-9)

    def test_monotonically_increasing(self):
        x = np.linspace(0, 1_000_000, 500)
        result = LogisticSaturation(L=1.0, k=0.00001, x0=500_000).transform(x)
        diffs = np.diff(result)
        assert np.all(diffs >= -1e-12)

    def test_inflection_at_x0(self):
        """Logistic(x0) ≈ L/2."""
        x0 = 500_000.0
        L  = 1.0
        result = LogisticSaturation(L=L, k=0.00002, x0=x0).transform(np.array([x0]))
        np.testing.assert_almost_equal(result[0], L / 2, decimal=3)

    def test_output_shape(self):
        x = np.random.rand(50) * 1e5
        result = LogisticSaturation().transform(x)
        assert result.shape == x.shape


class TestApplySaturationUtil:

    def test_hill_routing(self):
        x = np.array([10_000., 50_000., 100_000.])
        r1 = apply_saturation(x, saturation_type="hill", K=50_000, S=2.0)
        r2 = HillSaturation(K=50_000, S=2.0).transform(x)
        np.testing.assert_array_equal(r1, r2)

    def test_logistic_routing(self):
        x = np.array([10_000., 50_000., 100_000.])
        r1 = apply_saturation(x, saturation_type="logistic")
        r2 = LogisticSaturation().transform(x)
        np.testing.assert_array_equal(r1, r2)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            apply_saturation(np.array([1.0]), saturation_type="unknown")