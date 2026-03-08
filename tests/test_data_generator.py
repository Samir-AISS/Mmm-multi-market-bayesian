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
from src.data.data_validator import validate, EXPECTED_MARKETS


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def dataset():
    return generate_full_dataset()


# ── Niveau 1 : Structure ──────────────────────────────────────────────────────

class TestNiveau1Structure:
    COLS = [
        "market", "date", "week", "revenue",
        "tv_spend", "facebook_spend", "search_spend", "ooh_spend", "print_spend",
        "competitor_price", "events", "trend", "seasonality", "promotions",
    ]

    def test_toutes_colonnes_presentes(self, dataset):
        for col in self.COLS:
            assert col in dataset.columns, f"Colonne manquante : {col}"

    def test_pas_colonnes_extra(self, dataset):
        assert set(dataset.columns) == set(self.COLS)

    def test_types_numeriques(self, dataset):
        for col in ["revenue", "tv_spend", "facebook_spend",
                    "search_spend", "ooh_spend", "print_spend"]:
            assert pd.api.types.is_float_dtype(dataset[col]), f"{col} doit être float"

    def test_date_parsable(self, dataset):
        try:
            pd.to_datetime(dataset["date"])
            ok = True
        except Exception:
            ok = False
        assert ok


# ── Niveau 2 : Complétude ─────────────────────────────────────────────────────

class TestNiveau2Completude:

    def test_2080_lignes_exactes(self, dataset):
        assert len(dataset) == 2_080

    def test_zero_nulls(self, dataset):
        assert dataset.isnull().sum().sum() == 0

    def test_10_marches(self, dataset):
        assert set(dataset["market"].unique()) == set(EXPECTED_MARKETS)

    @pytest.mark.parametrize("market", ["FR", "DE", "UK", "IT", "ES",
                                         "NL", "BE", "PL", "SE", "NO"])
    def test_208_semaines_par_marche(self, dataset, market):
        n = len(dataset[dataset["market"] == market])
        assert n == N_WEEKS, f"{market} : {n} semaines (attendu {N_WEEKS})"

    def test_semaines_1_a_208(self, dataset):
        for market, grp in dataset.groupby("market"):
            weeks = sorted(grp["week"].values)
            assert weeks == list(range(1, N_WEEKS + 1)), \
                f"{market} : semaines non consécutives"


# ── Niveau 3 : Cohérence ──────────────────────────────────────────────────────

class TestNiveau3Coherence:

    @pytest.mark.parametrize("col", [
        "revenue", "tv_spend", "facebook_spend",
        "search_spend", "ooh_spend", "print_spend",
    ])
    def test_pas_negatifs(self, dataset, col):
        assert (dataset[col] < 0).sum() == 0, f"{col} contient des valeurs négatives"

    def test_pas_doublons(self, dataset):
        assert dataset.duplicated(subset=["market", "week"]).sum() == 0

    def test_events_binaires(self, dataset):
        assert dataset["events"].isin([0, 1]).all()

    def test_promotions_binaires(self, dataset):
        assert dataset["promotions"].isin([0, 1]).all()

    def test_seasonality_bornee(self, dataset):
        assert dataset["seasonality"].between(0.3, 3.0).all(), \
            f"Min={dataset['seasonality'].min():.3f}, Max={dataset['seasonality'].max():.3f}"

    def test_trend_positif(self, dataset):
        assert (dataset["trend"] > 0).all()

    def test_competitor_price_positif(self, dataset):
        assert (dataset["competitor_price"] > 0).all()


# ── Niveau 4 : Logique métier ─────────────────────────────────────────────────

class TestNiveau4LogiqueMetier:

    def test_tv_superieur_print_en_moyenne(self, dataset):
        assert dataset["tv_spend"].mean() > dataset["print_spend"].mean(), \
            "TV doit avoir des dépenses moyennes supérieures à Print"

    def test_trend_croissant(self, dataset):
        mean_debut = dataset[dataset["week"] < 50]["trend"].mean()
        mean_fin   = dataset[dataset["week"] > 160]["trend"].mean()
        assert mean_fin > mean_debut, \
            f"Trend fin ({mean_fin:.3f}) doit être > trend début ({mean_debut:.3f})"

    def test_revenue_positif(self, dataset):
        assert (dataset["revenue"] > 0).all()

    def test_revenue_varie_par_marche(self, dataset):
        rev_by_market = dataset.groupby("market")["revenue"].mean()
        assert rev_by_market.std() > 0, "Revenue moyen identique pour tous les marchés"

    def test_uk_revenue_superieur_pl(self, dataset):
        rev_uk = dataset[dataset["market"] == "UK"]["revenue"].mean()
        rev_pl = dataset[dataset["market"] == "PL"]["revenue"].mean()
        assert rev_uk > rev_pl, f"UK ({rev_uk:.0f}) doit > PL ({rev_pl:.0f})"

    def test_search_decay_faible(self, dataset):
        """Search a un faible decay → forte corrélation semaine courante."""
        for market, grp in dataset.groupby("market"):
            corr = grp["search_spend"].corr(grp["revenue"])
            assert corr > 0, f"{market} : corrélation search/revenue négative"

    def test_rapport_validation_zero_erreur(self, dataset):
        report = validate(dataset)
        errors = [(r.level, r.name) for r in report.results if not r.passed]
        assert report.n_errors == 0, \
            f"{report.n_errors} erreur(s) : {errors}"