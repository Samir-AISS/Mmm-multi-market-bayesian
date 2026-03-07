"""
test_data_generator.py
-----------------------
Tests automatiques pour multi_market_generator.py — 4 niveaux de validation.

Lancement : pytest tests/test_data_generator.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.multi_market_generator import generate_full_dataset, MARKETS, N_WEEKS
from src.data.data_validator import validate, EXPECTED_ROWS, EXPECTED_MARKETS


@pytest.fixture(scope="session")
def dataset():
    return generate_full_dataset()


# ── Niveau 1 : Structure ──────────────────────────────────────────────────────
class TestNiveau1Structure:
    COLS = ["market","date","week","revenue","tv_spend","facebook_spend",
            "search_spend","ooh_spend","print_spend","competitor_price",
            "events","trend","seasonality","promotions"]

    def test_toutes_colonnes_presentes(self, dataset):
        for col in self.COLS:
            assert col in dataset.columns, f"Colonne manquante: {col}"

    def test_pas_colonnes_extra(self, dataset):
        assert set(dataset.columns) == set(self.COLS)

    def test_types_numeriques(self, dataset):
        for col in ["revenue","tv_spend","facebook_spend","search_spend","ooh_spend","print_spend"]:
            assert pd.api.types.is_float_dtype(dataset[col]), f"{col} doit être float"

    def test_types_entiers(self, dataset):
        for col in ["week","events","promotions"]:
            assert pd.api.types.is_integer_dtype(dataset[col]), f"{col} doit être int"


# ── Niveau 2 : Complétude ─────────────────────────────────────────────────────
class TestNiveau2Completude:
    def test_2080_lignes_exactes(self, dataset):
        assert len(dataset) == 2_080

    def test_zero_nulls(self, dataset):
        assert dataset.isnull().sum().sum() == 0

    def test_10_marches(self, dataset):
        assert set(dataset["market"].unique()) == EXPECTED_MARKETS

    @pytest.mark.parametrize("market", ["FR","DE","UK","IT","ES","NL","BE","PL","SE","NO"])
    def test_208_semaines_par_marche(self, dataset, market):
        n = len(dataset[dataset["market"] == market])
        assert n == N_WEEKS, f"{market}: {n} semaines (attendu {N_WEEKS})"


# ── Niveau 3 : Cohérence ──────────────────────────────────────────────────────
class TestNiveau3Coherence:
    @pytest.mark.parametrize("col", ["revenue","tv_spend","facebook_spend",
                                      "search_spend","ooh_spend","print_spend"])
    def test_pas_negatifs(self, dataset, col):
        assert (dataset[col] < 0).sum() == 0

    def test_pas_doublons(self, dataset):
        assert dataset.duplicated(subset=["market","week"]).sum() == 0

    def test_events_binaires(self, dataset):
        assert dataset["events"].isin([0,1]).all()

    def test_promotions_binaires(self, dataset):
        assert dataset["promotions"].isin([0,1]).all()


# ── Niveau 4 : Logique métier ─────────────────────────────────────────────────
class TestNiveau4LogiqueMetier:
    def test_tv_superieur_print(self, dataset):
        assert (dataset["tv_spend"] <= dataset["print_spend"]).sum() == 0

    def test_revenus_noel_eleves(self, dataset):
        xmas = dataset[dataset["week"] % 52 >= 50]["revenue"].median()
        assert xmas > dataset["revenue"].median()

    def test_trend_croissant(self, dataset):
        assert dataset[dataset["week"] > 160]["trend"].mean() > \
               dataset[dataset["week"] < 50]["trend"].mean()

    def test_seasonality_bornee(self, dataset):
        assert ((dataset["seasonality"] < 0.7) | (dataset["seasonality"] > 1.5)).sum() == 0

    def test_rapport_validation_zero_erreur(self, dataset):
        report = validate(dataset)
        errors = [(r.level, r.name) for r in report.results if not r.passed]
        assert report.n_errors == 0, \
            f"{report.n_errors} erreur(s): " + str(errors)
