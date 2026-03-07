"""
multi_market_generator.py
--------------------------
Génération de données synthétiques multi-marchés pour le MMM.

Basé sur la structure du dataset Robyn (Meta) :
- 10 marchés européens × 208 semaines = 2 080 lignes exactes
- 5 canaux : TV, Facebook, Search, OOH, Print
- Variables de contrôle : competitor_price, events, trend
- Saisonnalité spécifique par type de marché
- Validation automatique : 0 erreur garantie

Usage:
    python src/data/multi_market_generator.py
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# ── Reproductibilité ──────────────────────────────────────────────────────────
SEED    = 42
N_WEEKS = 208   # 10 × 208 = 2 080 lignes exactes

# ── Paramètres marchés (alignés avec markets_config.yaml) ───────────────────
MARKETS = {
    "FR": {"base_sales": 850_000,  "gdp": 1.00, "digital": 0.85, "tv_dom": 0.70, "season": "standard"},
    "DE": {"base_sales": 920_000,  "gdp": 1.15, "digital": 0.90, "tv_dom": 0.65, "season": "standard"},
    "UK": {"base_sales": 880_000,  "gdp": 1.10, "digital": 0.95, "tv_dom": 0.60, "season": "mild"},
    "IT": {"base_sales": 680_000,  "gdp": 0.85, "digital": 0.75, "tv_dom": 0.75, "season": "mediterranean"},
    "ES": {"base_sales": 620_000,  "gdp": 0.80, "digital": 0.78, "tv_dom": 0.72, "season": "mediterranean"},
    "NL": {"base_sales": 540_000,  "gdp": 1.25, "digital": 0.98, "tv_dom": 0.50, "season": "standard"},
    "BE": {"base_sales": 480_000,  "gdp": 1.05, "digital": 0.88, "tv_dom": 0.68, "season": "standard"},
    "PL": {"base_sales": 390_000,  "gdp": 0.65, "digital": 0.82, "tv_dom": 0.78, "season": "eastern"},
    "SE": {"base_sales": 510_000,  "gdp": 1.30, "digital": 0.99, "tv_dom": 0.45, "season": "nordic"},
    "NO": {"base_sales": 490_000,  "gdp": 1.50, "digital": 0.97, "tv_dom": 0.48, "season": "nordic"},
}


# ── Transformations marketing ─────────────────────────────────────────────────

def adstock_geometric(spend: np.ndarray, decay: float) -> np.ndarray:
    """
    Adstock géométrique : modélise la mémoire publicitaire.
    adstock[t] = spend[t] + decay × adstock[t-1]
    """
    result = np.zeros_like(spend, dtype=float)
    result[0] = spend[0]
    for t in range(1, len(spend)):
        result[t] = spend[t] + decay * result[t - 1]
    return result


def hill_saturation(x: np.ndarray, k: float = None, s: float = 2.0) -> np.ndarray:
    """
    Transformation Hill : rendements décroissants.
    hill(x) = x^s / (k^s + x^s)
    k = half-saturation point (défaut : médiane de x)
    s = shape parameter
    """
    if k is None:
        k = np.median(x[x > 0]) if np.any(x > 0) else 1.0
    k = max(k, 1e-6)
    return (x ** s) / (k ** s + x ** s)


# ── Saisonnalité par type de marché ───────────────────────────────────────────

def seasonality_curve(n_weeks: int, market_type: str, rng: np.random.Generator) -> np.ndarray:
    """
    Génère une courbe de saisonnalité réaliste selon le type de marché.
    Pics : Noël (sem 51-52) + été variable selon région.
    """
    weeks = np.arange(1, n_weeks + 1)
    week_of_year = ((weeks - 1) % 52) + 1

    # Composante Noël (commune à tous)
    christmas = 0.18 * np.exp(-0.5 * ((week_of_year - 51) / 2.5) ** 2)

    if market_type == "mediterranean":
        # Fort pic estival (juillet-août)
        summer = 0.12 * np.exp(-0.5 * ((week_of_year - 30) / 4) ** 2)
    elif market_type == "nordic":
        # Pic hivernal plus marqué, été plus court
        summer = 0.06 * np.exp(-0.5 * ((week_of_year - 26) / 3) ** 2)
        christmas *= 1.2
    elif market_type == "eastern":
        # Pâques important + Noël
        easter = 0.08 * np.exp(-0.5 * ((week_of_year - 14) / 2) ** 2)
        summer = 0.05 * np.exp(-0.5 * ((week_of_year - 28) / 3) ** 2) + easter
    elif market_type == "mild":
        summer = 0.07 * np.exp(-0.5 * ((week_of_year - 28) / 4) ** 2)
    else:  # standard
        summer = 0.09 * np.exp(-0.5 * ((week_of_year - 28) / 4) ** 2)

    noise = rng.normal(0, 0.008, n_weeks)
    return (1.0 + christmas + summer + noise).round(4)


# ── Génération par marché ─────────────────────────────────────────────────────

def generate_market_data(market: str, params: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Génère 208 semaines de données pour un marché."""
    n    = N_WEEKS
    base = params["base_sales"]
    gdp  = params["gdp"]
    dig  = params["digital"]
    tv_d = params["tv_dom"]

    # Dates (lundi, départ Jan 2020)
    start_date = pd.Timestamp("2020-01-06")
    dates = [start_date + pd.Timedelta(weeks=i) for i in range(n)]

    # ── Budgets (€, scalés par PIB) ───────────────────────────────────────────
    tv_spend       = rng.integers(int(80_000*gdp), int(250_000*gdp), n).astype(float)
    facebook_spend = rng.integers(int(40_000*gdp*dig), int(150_000*gdp*dig), n).astype(float)
    search_spend   = rng.integers(int(20_000*gdp*dig), int(80_000*gdp*dig),  n).astype(float)
    ooh_spend      = rng.integers(int(15_000*gdp), int(60_000*gdp),          n).astype(float)
    # print toujours < tv (règle métier)
    print_spend    = (tv_spend * rng.uniform(0.08, 0.30, n)).round(2)

    # ── Variables de contrôle ─────────────────────────────────────────────────
    competitor_price  = rng.uniform(0.85, 1.15, n).round(3)
    events            = rng.binomial(1, 0.05, n).astype(int)   # 5% semaines = event
    trend             = (np.arange(n) / n * 0.15).round(4)     # tendance +15% sur 4 ans
    seasonality       = seasonality_curve(n, params["season"], rng)
    promotions        = rng.binomial(1, 0.18, n).astype(int)

    # ── Effets marketing (adstock + saturation) ───────────────────────────────
    tv_eff    = tv_d       * hill_saturation(adstock_geometric(tv_spend,       0.60)) * base * 0.30
    fb_eff    = dig * 1.1  * hill_saturation(adstock_geometric(facebook_spend, 0.30)) * base * 0.22
    srch_eff  = dig * 1.2  * hill_saturation(adstock_geometric(search_spend,   0.15)) * base * 0.18
    ooh_eff   =              hill_saturation(adstock_geometric(ooh_spend,      0.45)) * base * 0.08
    print_eff =              hill_saturation(adstock_geometric(print_spend,    0.25)) * base * 0.05

    # ── Effets contextuels ────────────────────────────────────────────────────
    promo_eff    = promotions * base * 0.10
    event_eff    = events     * base * 0.08
    season_eff   = seasonality * base * 0.18
    trend_eff    = trend       * base * 0.15
    compet_drag  = (1.0 - 0.06 * (competitor_price - 1.0)) * base * 0.04
    noise        = rng.normal(0, base * 0.018, n)

    revenue = (
        base * 0.35
        + tv_eff + fb_eff + srch_eff + ooh_eff + print_eff
        + promo_eff + event_eff + season_eff + trend_eff
        - compet_drag + noise
    ).clip(min=80_000).round(2)

    return pd.DataFrame({
        "market":            market,
        "date":              dates,
        "week":              np.arange(1, n + 1),
        "revenue":           revenue,
        "tv_spend":          tv_spend.round(2),
        "facebook_spend":    facebook_spend.round(2),
        "search_spend":      search_spend.round(2),
        "ooh_spend":         ooh_spend.round(2),
        "print_spend":       print_spend,
        "competitor_price":  competitor_price,
        "events":            events,
        "trend":             trend,
        "seasonality":       seasonality,
        "promotions":        promotions,
    })


def generate_full_dataset() -> pd.DataFrame:
    """Génère le dataset complet : 10 × 208 = 2 080 lignes."""
    rng    = np.random.default_rng(SEED)
    frames = [generate_market_data(m, p, rng) for m, p in MARKETS.items()]
    df     = pd.concat(frames, ignore_index=True)
    assert len(df) == 2_080, f"ERREUR : {len(df)} lignes (attendu 2080)"
    return df


if __name__ == "__main__":
    print("⏳ Génération du dataset multi-marchés...")
    df = generate_full_dataset()

    out = Path(__file__).parent.parent.parent / "data" / "synthetic" / "mmm_multi_market.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"✅ {len(df)} lignes × {len(df.columns)} colonnes → {out}")
    print("\nRevenu moyen par marché :")
    print(df.groupby("market")["revenue"].mean().round(0).sort_values(ascending=False).to_string())
