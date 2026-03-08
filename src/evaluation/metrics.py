"""
metrics.py
----------
Métriques d'évaluation pour le modèle MMM.

Métriques disponibles :
  - r_squared()        : coefficient de détermination R²
  - mape()             : Mean Absolute Percentage Error
  - smape()            : Symmetric MAPE
  - rmse()             : Root Mean Squared Error
  - nrmse()            : Normalized RMSE
  - compute_all_metrics() : toutes les métriques en une fois
  - print_metrics_report(): affichage formaté
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


# ── Métriques individuelles ───────────────────────────────────────────────────

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient de détermination R².
    R² = 1 - SS_res / SS_tot

    R² = 1.0 → prédiction parfaite
    R² = 0.0 → modèle équivalent à la moyenne
    R² < 0.0 → modèle pire que la moyenne
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return float(1 - ss_res / ss_tot)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (%).
    Ignore les valeurs y_true == 0 pour éviter la division par zéro.

    Retourne un pourcentage (ex: 8.5 pour 8.5%).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0
    if not np.any(mask):
        return 0.0

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (%).
    Symétrique contrairement à MAPE — pénalise également sur/sous-estimation.

    Formule : 100 × mean(|y - ŷ| / ((|y| + |ŷ|) / 2))
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask        = denominator != 0

    if not np.any(mask):
        return 0.0

    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.
    Même unité que y_true (€ pour les revenus MMM).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalized RMSE — RMSE divisé par l'étendue (max - min).
    Sans dimension → comparable entre marchés de tailles différentes.

    NRMSE = RMSE / (max(y_true) - min(y_true))
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    range_y = y_true.max() - y_true.min()
    if range_y == 0:
        return 0.0

    return float(rmse(y_true, y_pred) / range_y)


# ── Agrégation ────────────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calcule toutes les métriques en une seule fois.

    Retourne
    --------
    dict avec clés : r2, mape, smape, rmse, nrmse
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "r2":    r_squared(y_true, y_pred),
        "mape":  mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "rmse":  rmse(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
    }


def print_metrics_report(
    metrics: Dict[str, float],
    label: str = "Modèle",
) -> None:
    """
    Affiche un rapport formaté des métriques dans la console.

    Paramètres
    ----------
    metrics : dict retourné par compute_all_metrics()
    label   : titre du rapport (ex: "Train [FR]")
    """
    r2    = metrics.get("r2",    0)
    mape_ = metrics.get("mape",  0)
    smape_= metrics.get("smape", 0)
    rmse_ = metrics.get("rmse",  0)
    nrmse_= metrics.get("nrmse", 0)

    r2_icon    = "✅" if r2    >= 0.85 else "⚠️ "
    mape_icon  = "✅" if mape_ <= 15   else "⚠️ "
    nrmse_icon = "✅" if nrmse_ <= 0.15 else "⚠️ "

    print(f"\n{'─' * 50}")
    print(f"  📊 Métriques — {label}")
    print(f"{'─' * 50}")
    print(f"  {r2_icon}  R²     : {r2:.4f}   (objectif ≥ 0.85)")
    print(f"  {mape_icon}  MAPE   : {mape_:.2f}%  (objectif ≤ 15%)")
    print(f"      sMAPE  : {smape_:.2f}%")
    print(f"      RMSE   : €{rmse_:,.0f}")
    print(f"  {nrmse_icon}  NRMSE  : {nrmse_:.4f}  (objectif ≤ 0.15)")
    print(f"{'─' * 50}\n")


def metrics_to_dataframe(
    metrics_by_market: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Convertit un dict de métriques multi-marchés en DataFrame.

    Paramètres
    ----------
    metrics_by_market : {market: {r2, mape, smape, rmse, nrmse}}

    Retourne
    --------
    DataFrame avec une ligne par marché, trié par R² décroissant
    """
    rows = []
    for market, m in metrics_by_market.items():
        rows.append({
            "market": market,
            "r2":     round(m.get("r2",    0), 4),
            "mape":   round(m.get("mape",  0), 2),
            "smape":  round(m.get("smape", 0), 2),
            "rmse":   round(m.get("rmse",  0), 0),
            "nrmse":  round(m.get("nrmse", 0), 4),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("r2", ascending=False)
        .reset_index(drop=True)
    )