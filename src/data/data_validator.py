"""
data_validator.py
-----------------
Validation en 4 niveaux du dataset MMM multi-marchés.

Niveau 1 — Structure    : colonnes, types
Niveau 2 — Complétude   : 2080 lignes, 0 NULL, 10 marchés × 208 semaines
Niveau 3 — Cohérence    : pas de négatifs, pas de doublons
Niveau 4 — Logique métier : tv > print, Noël élevé, trend positif

Usage:
    from src.data.data_validator import validate
    report = validate(df)
    report.print_summary()  # → 0 erreur
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List

# ── Types attendus ─────────────────────────────────────────────────────────────
EXPECTED_COLUMNS = {
    "market":           "object",
    "date":             "object",
    "week":             "int64",
    "revenue":          "float64",
    "tv_spend":         "float64",
    "facebook_spend":   "float64",
    "search_spend":     "float64",
    "ooh_spend":        "float64",
    "print_spend":      "float64",
    "competitor_price": "float64",
    "events":           "int64",
    "trend":            "float64",
    "seasonality":      "float64",
    "promotions":       "int64",
}

EXPECTED_ROWS     = 2_080
EXPECTED_MARKETS  = {"FR","DE","UK","IT","ES","NL","BE","PL","SE","NO"}
N_WEEKS           = 208
SPEND_COLS        = ["tv_spend","facebook_spend","search_spend","ooh_spend","print_spend"]


@dataclass
class TestResult:
    level:  int
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
    def n_tests(self):  return len(self.results)

    def print_summary(self):
        emojis = {1:"🏗️ ", 2:"📊", 3:"🔍", 4:"💼"}
        labels = {
            1:"NIVEAU 1 — Structure",
            2:"NIVEAU 2 — Complétude",
            3:"NIVEAU 3 — Cohérence",
            4:"NIVEAU 4 — Logique métier",
        }
        cur = None
        for r in self.results:
            if r.level != cur:
                cur = r.level
                print(f"\n{emojis[r.level]}  {labels[r.level]}")
                print("  " + "─" * 52)
            s = "✅ PASS" if r.passed else "❌ FAIL"
            d = f"  → {r.detail}" if r.detail else ""
            print(f"  {s}  {r.name}{d}")
        print("\n" + "═" * 57)
        c = "🟢" if self.n_errors == 0 else "🔴"
        print(f"{c}  RÉSULTAT : {self.n_errors} erreur(s) / {self.n_tests} tests")
        print("═" * 57)


def _check_type(df, col, expected):
    actual = str(df[col].dtype)
    return (actual == expected
            or (expected == "int64"   and actual in ("int32","int64","Int64"))
            or (expected == "object"  and actual in ("object","str","string")) or (expected == "object" and "datetime" in actual)
            or (expected == "float64" and actual in ("float32","float64")))


def validate(df: pd.DataFrame) -> ValidationReport:
    r = ValidationReport()

    # ── Niveau 1 : Structure ──────────────────────────────────────────────────
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    extra   = set(df.columns) - set(EXPECTED_COLUMNS)
    r.add(1, "Colonnes attendues présentes", not missing, f"Manquantes: {missing}" if missing else "")
    r.add(1, "Pas de colonnes inattendues",  not extra,   f"Extra: {extra}" if extra else "")
    for col, exp in EXPECTED_COLUMNS.items():
        if col not in df.columns: continue
        ok = _check_type(df, col, exp)
        r.add(1, f"Type '{col}'", ok, f"Attendu {exp}, obtenu {df[col].dtype}" if not ok else "")

    # ── Niveau 2 : Complétude ─────────────────────────────────────────────────
    r.add(2, f"Exactement {EXPECTED_ROWS} lignes", len(df)==EXPECTED_ROWS, f"Obtenu: {len(df)}")
    nulls = df.isnull().sum().sum()
    r.add(2, "0 valeur NULL", nulls==0, f"{nulls} NULL" if nulls else "")
    actual_m = set(df["market"].unique()) if "market" in df.columns else set()
    r.add(2, "10 marchés attendus", actual_m==EXPECTED_MARKETS,
          f"Diff: {EXPECTED_MARKETS ^ actual_m}" if actual_m!=EXPECTED_MARKETS else "")
    if "market" in df.columns and "week" in df.columns:
        wpm = df.groupby("market")["week"].count()
        wrong = wpm[wpm != N_WEEKS]
        r.add(2, f"{N_WEEKS} semaines/marché", len(wrong)==0, wrong.to_dict() if len(wrong) else "")

    # ── Niveau 3 : Cohérence ──────────────────────────────────────────────────
    for col in ["revenue"] + SPEND_COLS:
        if col not in df.columns: continue
        n = (df[col] < 0).sum()
        r.add(3, f"Pas de négatifs '{col}'", n==0, f"{n} négatifs" if n else "")
    if "market" in df.columns and "week" in df.columns:
        dups = df.duplicated(subset=["market","week"]).sum()
        r.add(3, "Pas de doublons (market, week)", dups==0, f"{dups} doublons" if dups else "")
    for col in ["events","promotions"]:
        if col not in df.columns: continue
        bad = (~df[col].isin([0,1])).sum()
        r.add(3, f"'{col}' ∈ {{0,1}}", bad==0, f"{bad} invalides" if bad else "")

    # ── Niveau 4 : Logique métier ─────────────────────────────────────────────
    if "tv_spend" in df.columns and "print_spend" in df.columns:
        v = (df["tv_spend"] <= df["print_spend"]).sum()
        r.add(4, "tv_spend > print_spend", v==0, f"{v} violations" if v else "")
    if "week" in df.columns and "revenue" in df.columns:
        xmas = df[df["week"] % 52 >= 50]["revenue"].median()
        med  = df["revenue"].median()
        r.add(4, "Revenus Noël > médiane", xmas > med,
              f"Noël={xmas:,.0f} vs médiane={med:,.0f}")
    if "trend" in df.columns:
        trend_end   = df[df["week"] > N_WEEKS * 0.8]["trend"].mean()
        trend_start = df[df["week"] < N_WEEKS * 0.2]["trend"].mean()
        r.add(4, "Trend croissant", trend_end > trend_start,
              f"Début={trend_start:.4f}, Fin={trend_end:.4f}")
    if "seasonality" in df.columns:
        out = ((df["seasonality"] < 0.7) | (df["seasonality"] > 1.5)).sum()
        r.add(4, "Saisonnalité ∈ [0.7, 1.5]", out==0, f"{out} hors-plage" if out else "")
    if "revenue" in df.columns:
        low = (df["revenue"] < 50_000).sum()
        r.add(4, "Revenue > 50 000 €", low==0, f"{low} trop faibles" if low else "")

    return r


if __name__ == "__main__":
    from pathlib import Path
    path = Path(__file__).parent.parent.parent / "data" / "synthetic" / "mmm_multi_market.csv"
    if not path.exists():
        print("❌ Fichier introuvable. Lancez d'abord multi_market_generator.py")
        exit(1)
    df = pd.read_csv(path)
    report = validate(df)
    report.print_summary()
    exit(report.n_errors)
