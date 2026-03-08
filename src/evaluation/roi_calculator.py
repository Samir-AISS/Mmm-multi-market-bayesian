"""
roi_calculator.py
-----------------
Calcul du ROI par canal marketing.
ROI = Revenus générés par le canal / Dépenses du canal.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


SPEND_COLS = ["tv_spend", "facebook_spend", "search_spend", "ooh_spend", "print_spend"]


def compute_channel_contributions(
    df: pd.DataFrame,
    channel_effects: Dict[str, np.ndarray],
    base_effect: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Construit un DataFrame de décomposition des revenus.

    Paramètres
    ----------
    df              : données originales (avec revenue)
    channel_effects : {channel_name: array of weekly contributions}
    base_effect     : array des ventes de base (non-marketing)

    Retourne
    --------
    DataFrame avec colonnes : week, market, revenue, base, tv, facebook, ...
    """
    result = pd.DataFrame({
        "market":         df["market"].values if "market" in df.columns else "ALL",
        "week":           df["week"].values,
        "revenue_actual": df["revenue"].values,
    })

    total_explained = np.zeros(len(df))

    # Base
    if base_effect is not None:
        result["base"]   = base_effect
        total_explained += base_effect

    # Canaux marketing
    for channel, effect in channel_effects.items():
        result[channel]  = effect
        total_explained += effect

    # Résidu
    result["residual"] = df["revenue"].values - total_explained

    return result


def compute_roi(
    contributions_df: pd.DataFrame,
    spends_df: pd.DataFrame,
    channels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Calcule le ROI par canal : revenus générés / dépenses.

    Paramètres
    ----------
    contributions_df : résultat de compute_channel_contributions()
    spends_df        : DataFrame avec les colonnes de dépenses originales
    channels         : liste des canaux (défaut : tous)

    Retourne
    --------
    DataFrame : [channel, total_revenue, total_spend, roi, roi_per_1k]
    """
    channels = channels or [c.replace("_spend", "") for c in SPEND_COLS]
    records  = []

    for ch in channels:
        contrib_col = ch if ch in contributions_df.columns else f"{ch}_contribution"
        spend_col   = f"{ch}_spend"

        if contrib_col not in contributions_df.columns:
            continue
        if spend_col not in spends_df.columns:
            continue

        total_revenue = contributions_df[contrib_col].sum()
        total_spend   = spends_df[spend_col].sum()
        roi           = total_revenue / total_spend if total_spend > 0 else np.nan

        records.append({
            "channel":       ch,
            "total_revenue": round(total_revenue, 2),
            "total_spend":   round(total_spend, 2),
            "roi":           round(roi, 4),
            "roi_per_1k":    round(roi * 1000, 2),
        })

    return pd.DataFrame(records).sort_values("roi", ascending=False).reset_index(drop=True)


def compute_marginal_roi(
    channel: str,
    current_spend: float,
    hill_K: float,
    hill_S: float,
    beta: float,
    delta: float = 1000.0,
) -> float:
    """
    ROI marginal d'un canal : gain de revenu pour +delta€ supplémentaire.

    Utilise la dérivée numérique de la fonction Hill.

    Paramètres
    ----------
    channel       : nom du canal (pour les logs)
    current_spend : spend actuel (€)
    hill_K        : half-saturation point
    hill_S        : shape parameter
    beta          : coefficient du canal (posterior mean)
    delta         : incrément de test (défaut : 1 000€)
    """
    def hill(x: float) -> float:
        return (x ** hill_S) / (hill_K ** hill_S + x ** hill_S)

    revenue_now   = beta * hill(current_spend)
    revenue_after = beta * hill(current_spend + delta)
    return (revenue_after - revenue_now) / delta


def compute_roi_with_uncertainty(
    contributions_samples: np.ndarray,
    total_spend: float,
    credible_interval: float = 0.94,
) -> Dict[str, float]:
    """
    ROI avec intervalles de crédibilité bayésiens.

    Paramètres
    ----------
    contributions_samples : array (n_samples,) — somme des contributions postérieures
    total_spend           : dépenses totales du canal (€)
    credible_interval     : largeur de l'intervalle (défaut : 94%)

    Retourne
    --------
    dict avec roi_mean, roi_median, roi_lower, roi_upper, roi_std
    """
    if total_spend <= 0:
        return {"roi_mean": np.nan, "roi_median": np.nan,
                "roi_lower": np.nan, "roi_upper": np.nan}

    roi_samples = contributions_samples / total_spend
    alpha       = (1 - credible_interval) / 2

    return {
        "roi_mean":   float(np.mean(roi_samples)),
        "roi_median": float(np.median(roi_samples)),
        "roi_lower":  float(np.percentile(roi_samples, alpha * 100)),
        "roi_upper":  float(np.percentile(roi_samples, (1 - alpha) * 100)),
        "roi_std":    float(np.std(roi_samples)),
    }


def budget_recommendation(
    roi_df: pd.DataFrame,
    total_budget: float,
    min_alloc: float = 0.05,
    max_alloc: float = 0.60,
) -> pd.DataFrame:
    """
    Allocation budgétaire simple basée sur les ROI (proportionnelle).

    Contraintes :
      - min_alloc : % minimum par canal actif
      - max_alloc : % maximum pour un seul canal

    Retourne
    --------
    DataFrame : [channel, roi, roi_per_1k, recommended_budget, recommended_share]
    """
    df = roi_df.copy().dropna(subset=["roi"])
    df = df[df["roi"] > 0].reset_index(drop=True)

    if len(df) == 0:
        return pd.DataFrame()

    # Poids proportionnels aux ROI
    roi_vals = df["roi"].values
    weights  = roi_vals / roi_vals.sum()

    # Contraintes min/max
    weights = np.clip(weights, min_alloc, max_alloc)
    weights = weights / weights.sum()  # renormaliser

    df["recommended_budget"] = (weights * total_budget).round(2)
    df["recommended_share"]  = (weights * 100).round(1)

    return df[["channel", "roi", "roi_per_1k", "recommended_budget", "recommended_share"]]


def roi_summary_all_markets(
    results_by_market: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Consolide les ROI de tous les marchés en un seul DataFrame.

    Paramètres
    ----------
    results_by_market : {market: roi_df} — un roi_df par marché

    Retourne
    --------
    DataFrame long avec colonnes : [market, channel, roi, total_spend, total_revenue]
    """
    rows = []
    for market, roi_df in results_by_market.items():
        for _, row in roi_df.iterrows():
            rows.append({
                "market":        market,
                "channel":       row["channel"],
                "roi":           row.get("roi"),
                "total_spend":   row.get("total_spend"),
                "total_revenue": row.get("total_revenue"),
            })
    return pd.DataFrame(rows).sort_values(["market", "roi"], ascending=[True, False])


if __name__ == "__main__":
    # Test rapide avec données fictives
    roi_data = pd.DataFrame({
        "channel":   ["tv", "facebook", "search", "ooh", "print"],
        "roi":       [1.8, 2.4, 3.1, 1.2, 0.9],
        "roi_per_1k": [1800, 2400, 3100, 1200, 900],
    })
    rec = budget_recommendation(roi_data, total_budget=1_000_000)
    print("Budget recommendation (1M€ total) :")
    print(rec.to_string(index=False))
    print("✅ roi_calculator.py OK")