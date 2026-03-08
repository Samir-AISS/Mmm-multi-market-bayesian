"""
data_validator.py
-----------------
Validation en 4 niveaux des données MMM multi-marchés.

Niveaux :
  1. Structure   — colonnes, types, shape
  2. Intégrité   — valeurs manquantes, négatifs, doublons
  3. Cohérence   — plages de valeurs, corrélations
  4. Métier      — logique MMM (saisonnalité, tendance, etc.)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


REQUIRED_COLUMNS = [
    "market", "date", "week", "revenue",
    "tv_spend", "facebook_spend", "search_spend", "ooh_spend", "print_spend",
    "competitor_price", "events", "trend", "seasonality", "promotions",
]

EXPECTED_MARKETS = ["FR", "DE", "UK", "IT", "ES", "NL", "BE", "PL", "SE", "NO"]
EXPECTED_WEEKS   = 208
SPEND_COLS       = ["tv_spend", "facebook_spend", "search_spend", "ooh_spend", "print_spend"]


# ── Résultats de validation ───────────────────────────────────────────────────

@dataclass
class TestResult:
    level:  str
    name:   str
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    results: List[TestResult] = field(default_factory=list)

    def add(self, level, name, passed, detail=""):
        self.results.append(TestResult(level, name, passed, detail))

    @property
    def n_errors(self): return sum(1 for r in self.results if not r.passed)

    @property
    def n_tests(self): return len(self.results)

    def print_summary(self):
        print(f"\n{'═' * 60}")
        print(f"  RAPPORT DE VALIDATION — {self.n_tests} tests")
        print(f"{'═' * 60}")
        for r in self.results:
            s = " PASS" if r.passed else " FAIL"
            detail = f" — {r.detail}" if r.detail else ""
            print(f"  [{r.level:12s}] {s}  {r.name}{detail}")
        print(f"{'═' * 60}")
        print(f"  Résultat : {self.n_errors} erreur(s) / {self.n_tests} tests")
        print(f"{'═' * 60}\n")


# ── Niveau 1 : Structure ──────────────────────────────────────────────────────

def validate_structure(df: pd.DataFrame, report: ValidationReport):
    """Vérifie colonnes, types, shape."""

    # 1.1 Colonnes requises
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report.add("STRUCTURE", "Colonnes requises présentes",
               len(missing_cols) == 0,
               f"Manquantes : {missing_cols}" if missing_cols else "")

    # 1.2 Nombre de lignes
    report.add("STRUCTURE", "Nombre de lignes = 2080",
               len(df) == 2080,
               f"Trouvé : {len(df)}")

    # 1.3 Nombre de marchés
    n_markets = df["market"].nunique() if "market" in df.columns else 0
    report.add("STRUCTURE", "10 marchés présents",
               n_markets == 10,
               f"Trouvé : {n_markets}")

    # 1.4 Semaines par marché
    if "market" in df.columns and "week" in df.columns:
        weeks_per_market = df.groupby("market")["week"].count()
        all_208 = (weeks_per_market == EXPECTED_WEEKS).all()
        report.add("STRUCTURE", "208 semaines par marché",
                   all_208,
                   "" if all_208 else f"Min={weeks_per_market.min()}, Max={weeks_per_market.max()}")

    # 1.5 Types numériques spend
    for col in SPEND_COLS:
        if col in df.columns:
            report.add("STRUCTURE", f"Type numérique : {col}",
                       pd.api.types.is_numeric_dtype(df[col]))

    # 1.6 Colonne date parsable
    if "date" in df.columns:
        try:
            pd.to_datetime(df["date"])
            report.add("STRUCTURE", "Colonne date parsable", True)
        except Exception:
            report.add("STRUCTURE", "Colonne date parsable", False)


# ── Niveau 2 : Intégrité ──────────────────────────────────────────────────────

def validate_integrity(df: pd.DataFrame, report: ValidationReport):
    """Vérifie valeurs manquantes, négatifs, doublons."""

    # 2.1 Aucune valeur manquante
    n_null = df.isnull().sum().sum()
    report.add("INTÉGRITÉ", "Aucune valeur manquante",
               n_null == 0,
               f"{n_null} valeur(s) manquante(s)" if n_null else "")

    # 2.2 Revenue positif
    if "revenue" in df.columns:
        n_neg = (df["revenue"] <= 0).sum()
        report.add("INTÉGRITÉ", "Revenue > 0",
                   n_neg == 0,
                   f"{n_neg} valeur(s) ≤ 0" if n_neg else "")

    # 2.3 Spends non négatifs
    for col in SPEND_COLS:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            report.add("INTÉGRITÉ", f"Spend ≥ 0 : {col}",
                       n_neg == 0,
                       f"{n_neg} valeur(s) < 0" if n_neg else "")

    # 2.4 Aucun doublon (market, week)
    if "market" in df.columns and "week" in df.columns:
        n_dup = df.duplicated(subset=["market", "week"]).sum()
        report.add("INTÉGRITÉ", "Aucun doublon (market, week)",
                   n_dup == 0,
                   f"{n_dup} doublon(s)" if n_dup else "")

    # 2.5 Marchés attendus
    if "market" in df.columns:
        actual   = set(df["market"].unique())
        expected = set(EXPECTED_MARKETS)
        missing  = expected - actual
        extra    = actual - expected
        report.add("INTÉGRITÉ", "Marchés attendus présents",
                   len(missing) == 0,
                   f"Manquants : {missing}" if missing else "")


# ── Niveau 3 : Cohérence ──────────────────────────────────────────────────────

def validate_coherence(df: pd.DataFrame, report: ValidationReport):
    """Vérifie plages de valeurs et cohérences statistiques."""

    # 3.1 Saisonnalité dans [0.5, 2.0]
    if "seasonality" in df.columns:
        ok = (df["seasonality"].between(0.3, 3.0)).all()
        report.add("COHÉRENCE", "Saisonnalité dans [0.3, 3.0]",
                   ok,
                   f"Min={df['seasonality'].min():.2f}, Max={df['seasonality'].max():.2f}")

    # 3.2 Trend dans [0.5, 2.0]
    if "trend" in df.columns:
        ok = (df["trend"].between(0.5, 2.0)).all()
        report.add("COHÉRENCE", "Trend dans [0.5, 2.0]",
                   ok,
                   f"Min={df['trend'].min():.2f}, Max={df['trend'].max():.2f}")

    # 3.3 Events binaire
    if "events" in df.columns:
        unique_vals = set(df["events"].unique())
        ok = unique_vals.issubset({0, 1, 0.0, 1.0})
        report.add("COHÉRENCE", "Events binaire {0, 1}",
                   ok,
                   f"Valeurs : {unique_vals}" if not ok else "")

    # 3.4 Promotions binaire
    if "promotions" in df.columns:
        unique_vals = set(df["promotions"].unique())
        ok = unique_vals.issubset({0, 1, 0.0, 1.0})
        report.add("COHÉRENCE", "Promotions binaire {0, 1}",
                   ok,
                   f"Valeurs : {unique_vals}" if not ok else "")

    # 3.5 Revenue > somme spends (logique économique)
    if "revenue" in df.columns and all(c in df.columns for c in SPEND_COLS):
        total_spend = df[SPEND_COLS].sum(axis=1)
        ok = (df["revenue"] > total_spend).mean() > 0.8
        report.add("COHÉRENCE", "Revenue > total spend (80% des semaines)",
                   ok)

    # 3.6 Variance non nulle par marché
    if "market" in df.columns and "revenue" in df.columns:
        variances = df.groupby("market")["revenue"].var()
        ok = (variances > 0).all()
        report.add("COHÉRENCE", "Revenue non constant par marché",
                   ok)


# ── Niveau 4 : Métier ─────────────────────────────────────────────────────────

def validate_business(df: pd.DataFrame, report: ValidationReport):
    """Vérifie la logique métier MMM."""

    # 4.1 Corrélation spend/revenue positive
    if "revenue" in df.columns:
        for col in SPEND_COLS:
            if col in df.columns:
                corr = df[[col, "revenue"]].corr().iloc[0, 1]
                ok   = corr > 0
                report.add("MÉTIER", f"Corrélation positive {col}/revenue",
                           ok,
                           f"r={corr:.3f}")

    # 4.2 TV spend = canal le plus important en volume
    if all(c in df.columns for c in SPEND_COLS):
        means     = df[SPEND_COLS].mean()
        top_canal = means.idxmax()
        ok        = top_canal == "tv_spend"
        report.add("MÉTIER", "TV = canal dominant en volume",
                   ok,
                   f"Canal dominant : {top_canal}")

    # 4.3 Semaines consécutives par marché
    if "market" in df.columns and "week" in df.columns:
        ok = True
        for market, grp in df.groupby("market"):
            weeks = sorted(grp["week"].values)
            if weeks != list(range(min(weeks), max(weeks) + 1)):
                ok = False
                break
        report.add("MÉTIER", "Semaines consécutives par marché", ok)

    # 4.4 Revenue moyen cohérent entre marchés (pas de valeur aberrante)
    if "market" in df.columns and "revenue" in df.columns:
        rev_by_market = df.groupby("market")["revenue"].mean()
        cv = rev_by_market.std() / rev_by_market.mean()
        ok = cv < 1.5
        report.add("MÉTIER", "Revenue cohérent entre marchés (CV < 1.5)",
                   ok,
                   f"CV = {cv:.2f}")


# ── Point d'entrée principal ──────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> ValidationReport:
    """
    Lance les 4 niveaux de validation et retourne le rapport complet.

    Paramètres
    ----------
    df : DataFrame à valider

    Retourne
    --------
    ValidationReport avec tous les résultats
    """
    report = ValidationReport()

    validate_structure(df, report)
    validate_integrity(df, report)
    validate_coherence(df, report)
    validate_business(df, report)

    return report


if __name__ == "__main__":
    path = Path(__file__).parent.parent.parent / "data" / "synthetic" / "mmm_multi_market.csv"
    df   = pd.read_csv(path)
    rep  = validate(df)
    rep.print_summary()