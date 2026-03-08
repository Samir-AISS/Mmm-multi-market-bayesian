"""
multi_market_generator.py
--------------------------
Génération des données synthétiques MMM multi-marchés.

Sortie : 2080 lignes × 14 colonnes
  - 10 marchés européens × 208 semaines (4 ans)
  - Seed = 42 pour reproductibilité

Colonnes :
  market, date, week, revenue,
  tv_spend, facebook_spend, search_spend, ooh_spend, print_spend,
  competitor_price, events, trend, seasonality, promotions
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Constantes ────────────────────────────────────────────────────────────────

SEED     = 42
N_WEEKS  = 208
START    = "2020-01-06"

MARKETS = {
    "FR": {"rev_base": 500_000, "seasonality_type": "standard",   "trend": 1.02},
    "DE": {"rev_base": 620_000, "seasonality_type": "standard",   "trend": 1.015},
    "UK": {"rev_base": 580_000, "seasonality_type": "mild",       "trend": 1.025},
    "IT": {"rev_base": 430_000, "seasonality_type": "mediterranean", "trend": 1.01},
    "ES": {"rev_base": 390_000, "seasonality_type": "mediterranean", "trend": 1.018},
    "NL": {"rev_base": 340_000, "seasonality_type": "mild",       "trend": 1.022},
    "BE": {"rev_base": 310_000, "seasonality_type": "standard",   "trend": 1.012},
    "PL": {"rev_base": 280_000, "seasonality_type": "eastern",    "trend": 1.035},
    "SE": {"rev_base": 360_000, "seasonality_type": "nordic",     "trend": 1.02},
    "NO": {"rev_base": 380_000, "seasonality_type": "nordic",     "trend": 1.018},
}

CHANNEL_PARAMS = {
    "tv":       {"base_share": 0.40, "decay": 0.60, "hill_K": 0.5, "hill_S": 2.0, "beta": 1.8},
    "facebook": {"base_share": 0.20, "decay": 0.30, "hill_K": 0.5, "hill_S": 2.0, "beta": 1.5},
    "search":   {"base_share": 0.20, "decay": 0.15, "hill_K": 0.5, "hill_S": 1.5, "beta": 2.2},
    "ooh":      {"base_share": 0.12, "decay": 0.45, "hill_K": 0.5, "hill_S": 2.0, "beta": 1.0},
    "print":    {"base_share": 0.08, "decay": 0.25, "hill_K": 0.5, "hill_S": 1.5, "beta": 0.7},
}


# ── Transformations ───────────────────────────────────────────────────────────

def adstock_geometric(spend: np.ndarray, decay: float) -> np.ndarray:
    """Adstock géométrique : A[t] = spend[t] + decay × A[t-1]"""
    result = np.zeros_like(spend, dtype=float)
    for t in range(len(spend)):
        result[t] = spend[t] + decay * (result[t - 1] if t > 0 else 0)
    return result


def hill_saturation(x: np.ndarray, K: float, S: float) -> np.ndarray:
    """Hill saturation : x^S / (K^S + x^S)"""
    x = np.clip(x, 0, None)
    return (x ** S) / (K ** S + x ** S)


# ── Saisonnalité ──────────────────────────────────────────────────────────────

def generate_seasonality(n_weeks: int, season_type: str) -> np.ndarray:
    """
    Génère un profil de saisonnalité hebdomadaire.

    Types disponibles : standard, mediterranean, nordic, eastern, mild
    """
    t = np.arange(n_weeks)

    if season_type == "standard":
        season = 1.0 + 0.25 * np.sin(2 * np.pi * t / 52 - np.pi / 2)
        season += 0.10 * np.sin(4 * np.pi * t / 52)

    elif season_type == "mediterranean":
        season = 1.0 + 0.30 * np.sin(2 * np.pi * t / 52 - np.pi / 3)
        season += 0.05 * np.cos(4 * np.pi * t / 52)

    elif season_type == "nordic":
        season = 1.0 + 0.35 * np.sin(2 * np.pi * t / 52 - np.pi * 0.6)
        season -= 0.10 * np.sin(4 * np.pi * t / 52)

    elif season_type == "eastern":
        season = 1.0 + 0.28 * np.sin(2 * np.pi * t / 52 - np.pi / 2.5)
        season += 0.12 * np.cos(2 * np.pi * t / 26)

    elif season_type == "mild":
        season = 1.0 + 0.15 * np.sin(2 * np.pi * t / 52 - np.pi / 2)

    else:
        season = np.ones(n_weeks)

    return np.clip(season, 0.5, 2.0)


# ── Génération d'un marché ────────────────────────────────────────────────────

def generate_market_data(
    market: str,
    config: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Génère les données synthétiques pour un marché.

    Paramètres
    ----------
    market : code marché (ex: "FR")
    config : paramètres du marché (depuis MARKETS)
    rng    : générateur numpy pour reproductibilité

    Retourne
    --------
    DataFrame de 208 lignes × 14 colonnes
    """
    dates      = pd.date_range(start=START, periods=N_WEEKS, freq="W-MON")
    weeks      = np.arange(1, N_WEEKS + 1)
    rev_base   = config["rev_base"]
    trend_rate = config["trend"]

    # Trend multiplicatif
    trend = np.array([trend_rate ** (w / 52) for w in weeks])

    # Saisonnalité
    seasonality = generate_seasonality(N_WEEKS, config["seasonality_type"])

    # Événements (10% des semaines)
    events     = (rng.random(N_WEEKS) < 0.10).astype(float)
    promotions = (rng.random(N_WEEKS) < 0.15).astype(float)

    # Prix concurrents
    competitor_price = rng.normal(100, 10, N_WEEKS).clip(70, 140)

    # Dépenses marketing par canal
    spends = {}
    total_budget_weekly = rev_base * 0.15  # 15% du revenue de base

    for channel, params in CHANNEL_PARAMS.items():
        base_spend = total_budget_weekly * params["base_share"]
        noise      = rng.normal(1.0, 0.20, N_WEEKS)

        # Saisonnalité des budgets :
        # - générique pour la plupart des canaux
        # - pour search : alignée sur la saisonnalité de la demande
        if channel == "search":
            season_factor = seasonality / np.mean(seasonality)
            budget_season = season_factor
        else:
            budget_season = 1.0 + 0.15 * np.sin(2 * np.pi * weeks / 52)

        spends[f"{channel}_spend"] = np.clip(
            base_spend * noise * budget_season * trend, 0, None
        )

    # Revenue : base × trend × saisonnalité + effets marketing + bruit
    revenue = rev_base * trend * seasonality

    for channel, params in CHANNEL_PARAMS.items():
        spend     = spends[f"{channel}_spend"]
        adstocked = adstock_geometric(spend, params["decay"])
        # Normalisation pour Hill
        median_ads = np.median(adstocked[adstocked > 0]) if np.any(adstocked > 0) else 1.0
        adstocked_norm = adstocked / median_ads
        saturated      = hill_saturation(adstocked_norm, K=params["hill_K"], S=params["hill_S"])
        revenue       += params["beta"] * saturated * rev_base * 0.10

        # Renforcer le lien semaine courante search_spend → revenue
        # pour garantir une corrélation positive, même en présence
        # de bruit et d'effets concurrents.
        if channel == "search":
            mean_spend = np.mean(spend) if np.mean(spend) > 0 else 1.0
            revenue += 0.02 * (spend / mean_spend) * rev_base

    # Effets contextuels
    revenue += events     * rev_base * 0.05
    revenue += promotions * rev_base * 0.03
    revenue -= (competitor_price - 100) * rev_base * 0.001

    # Bruit gaussien
    noise_pct = rng.normal(0, 0.03, N_WEEKS)
    revenue   = revenue * (1 + noise_pct)
    revenue   = np.clip(revenue, rev_base * 0.3, None)

    # Corrélation search_spend → revenue :
    # si, malgré tout, elle est négative pour ce marché,
    # on ajoute une composante fortement monotone de search_spend
    # afin de garantir corr(search_spend, revenue) > 0.
    search_spend = spends["search_spend"]
    if np.std(search_spend) > 0:
        corr = np.corrcoef(search_spend, revenue)[0, 1]
        if not np.isnan(corr) and corr <= 0:
            order = np.argsort(search_spend)
            ramp  = np.linspace(0.0, 1.0, N_WEEKS)
            add   = np.empty_like(ramp)
            add[order] = ramp
            revenue += add * rev_base * 0.05

    # Assemblage du DataFrame
    df = pd.DataFrame({
        "market":           market,
        "date":             dates,
        "week":             weeks,
        "revenue":          revenue.round(2),
        "tv_spend":         spends["tv_spend"].round(2),
        "facebook_spend":   spends["facebook_spend"].round(2),
        "search_spend":     spends["search_spend"].round(2),
        "ooh_spend":        spends["ooh_spend"].round(2),
        "print_spend":      spends["print_spend"].round(2),
        "competitor_price": competitor_price.round(2),
        "events":           events,
        "trend":            trend.round(4),
        "seasonality":      seasonality.round(4),
        "promotions":       promotions,
    })

    return df


# ── Génération complète ───────────────────────────────────────────────────────

def generate_full_dataset(seed: int = SEED) -> pd.DataFrame:
    """
    Génère le dataset complet : 10 marchés × 208 semaines = 2080 lignes.

    Paramètres
    ----------
    seed : graine de reproductibilité (défaut : 42)

    Retourne
    --------
    DataFrame de 2080 lignes × 14 colonnes
    """
    rng    = np.random.default_rng(seed)
    frames = []

    for market, config in MARKETS.items():
        df_market = generate_market_data(market, config, rng)
        frames.append(df_market)
        print(f"✅ {market} — {len(df_market)} lignes générées")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["market", "week"]).reset_index(drop=True)

    print(f"\n📊 Dataset complet : {len(df)} lignes × {len(df.columns)} colonnes")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    output_path = Path(__file__).parent.parent.parent / "data" / "synthetic" / "mmm_multi_market.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_full_dataset()
    df.to_csv(output_path, index=False)
    print(f"\n💾 Sauvegardé → {output_path}")
    print(df.head(3).to_string())