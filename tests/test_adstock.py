"""
test_adstock.py
---------------
Tests unitaires pour les transformations adstock.
TODO : à compléter après implémentation de src/models/adstock.py
"""
import pytest
import numpy as np

# from src.models.adstock import GeometricAdstock, DelayedAdstock


class TestGeometricAdstock:
    def test_zero_spend_gives_zero(self):
        # TODO
        pass

    def test_decay_zero_equals_original(self):
        # decay=0 → pas de carryover → adstock = spend
        # TODO
        pass

    def test_decay_one_is_cumulative_sum(self):
        # decay=1 → adstock[t] = sum(spend[0:t+1])
        # TODO
        pass

    def test_output_shape(self):
        # TODO
        pass


class TestDelayedAdstock:
    def test_peak_shift(self):
        # TODO
        pass
