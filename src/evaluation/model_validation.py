"""
model_validation.py
-------------------
Validation statistique du modèle MMM.

Tests disponibles :
  - walk_forward_validation() : validation temporelle glissante
  - posterior_predictive_check() : PPC bayésien
  - cross_market_consistency() : cohérence entre marchés
  - full_validation_report() : rapport complet
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.evaluation.metrics import compute_all_metrics
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ── 1. Walk-Forward Validation ────────────────────────────────────────────────

def walk_forward_validation(
    df: pd.DataFrame,
    model_class,
    config: dict,
    n_splits: int = 5,
    min_train_size: int = 100,
) -> pd.DataFrame:
    """
    Validation temporelle glissante (time-series cross-validation).

    Principe : on entraîne sur [0..t] et on prédit [t+1..t+h],
    en faisant glisser la fenêtre n_splits fois.

    Paramètres
    ----------
    df            : DataFrame trié par semaine (une seule marché)
    model_class   : classe du modèle (ex: BayesianMMM)
    config        : configuration du modèle
    n_splits      : nombre de découpages temporels
    min_train_size: taille minimale du set d'entraînement

    Retourne
    --------
    DataFrame avec colonnes [split, n_train, n_test, r2, mape, nrmse, rmse]
    """
    df = df.sort_values("week").reset_index(drop=True)
    n  = len(df)

    step     = (n - min_train_size) // n_splits
    results  = []

    for i in range(n_splits):
        train_end = min_train_size + i * step
        if train_end >= n:
            break

        df_train = df.iloc[:train_end].copy()
        df_test  = df.iloc[train_end: train_end + step].copy()

        if len(df_test) == 0:
            continue

        logger.info(f"Split {i+1}/{n_splits} — train: {len(df_train)}, test: {len(df_test)}")

        try:
            model = model_class(config=config)
            model.build_model(df_train)
            model.fit(df_train)

            y_pred  = model.predict(df_test)
            y_true  = df_test["revenue"].values
            metrics = compute_all_metrics(y_true, y_pred)

            results.append({
                "split":   i + 1,
                "n_train": len(df_train),
                "n_test":  len(df_test),
                **metrics,
            })

        except Exception as e:
            logger.error(f"Split {i+1} erreur : {e}")
            results.append({
                "split":   i + 1,
                "n_train": len(df_train),
                "n_test":  len(df_test),
                "error":   str(e),
            })

    return pd.DataFrame(results)


# ── 2. Posterior Predictive Check ─────────────────────────────────────────────

def posterior_predictive_check(
    idata,
    df: pd.DataFrame,
    n_samples: int = 200,
) -> Dict:
    """
    Posterior Predictive Check (PPC) bayésien.

    Compare la distribution des données observées avec celle
    des données simulées à partir de la distribution postérieure.

    Paramètres
    ----------
    idata     : az.InferenceData retourné par PyMC
    df        : DataFrame avec la colonne 'revenue'
    n_samples : nombre d'échantillons postérieurs à utiliser

    Retourne
    --------
    dict avec : mean_ppc, std_ppc, coverage_94, p_value_mean, p_value_std
    """
    try:
        import arviz as az
    except ImportError:
        logger.error("arviz non installé — pip install arviz")
        return {}

    y_obs = df["revenue"].values

    # Récupérer les prédictions postérieures
    if not hasattr(idata, "posterior_predictive"):
        logger.warning("idata ne contient pas de posterior_predictive. "
                       "Appelez model.sample_posterior_predictive() d'abord.")
        return {}

    # Extraire les échantillons (shape : chains × draws × obs)
    ppc_samples = idata.posterior_predictive["revenue"].values
    ppc_flat    = ppc_samples.reshape(-1, ppc_samples.shape[-1])

    # Sous-échantillonner si nécessaire
    idx     = np.random.choice(len(ppc_flat), min(n_samples, len(ppc_flat)), replace=False)
    samples = ppc_flat[idx]

    # Statistiques
    mean_ppc = samples.mean(axis=0)
    std_ppc  = samples.std(axis=0)

    # Couverture : % de points observés dans l'intervalle HDI 94%
    lower = np.percentile(samples, 3,  axis=0)
    upper = np.percentile(samples, 97, axis=0)
    coverage_94 = np.mean((y_obs >= lower) & (y_obs <= upper)) * 100

    # Bayesian p-values
    p_value_mean = np.mean(samples.mean(axis=1) >= y_obs.mean())
    p_value_std  = np.mean(samples.std(axis=1)  >= y_obs.std())

    result = {
        "mean_ppc":      float(mean_ppc.mean()),
        "mean_observed": float(y_obs.mean()),
        "std_ppc":       float(std_ppc.mean()),
        "std_observed":  float(y_obs.std()),
        "coverage_94":   float(coverage_94),
        "p_value_mean":  float(p_value_mean),
        "p_value_std":   float(p_value_std),
        "ppc_samples":   samples,
    }

    # Évaluation qualitative
    result["assessment"] = _assess_ppc(coverage_94, p_value_mean, p_value_std)
    logger.info(f"PPC — Coverage 94%: {coverage_94:.1f}% | "
                f"p_mean: {p_value_mean:.3f} | p_std: {p_value_std:.3f}")
    return result


def _assess_ppc(coverage: float, p_mean: float, p_std: float) -> str:
    """Évalue qualitativement le PPC."""
    issues = []
    if coverage < 80:
        issues.append(f"couverture faible ({coverage:.1f}% < 80%)")
    if p_mean < 0.05 or p_mean > 0.95:
        issues.append(f"p_mean extrême ({p_mean:.3f})")
    if p_std < 0.05 or p_std > 0.95:
        issues.append(f"p_std extrême ({p_std:.3f})")
    if issues:
        return "⚠️  Problèmes détectés : " + " | ".join(issues)
    return "✅ PPC satisfaisant"


# ── 3. Cross-Market Consistency ───────────────────────────────────────────────

def cross_market_consistency(
    results_by_market: Dict[str, dict],
) -> pd.DataFrame:
    """
    Vérifie la cohérence des métriques et ROI entre marchés.

    Détecte les marchés outliers (R² anormalement bas, ROI aberrant, etc.)

    Paramètres
    ----------
    results_by_market : {market: {"metrics": {...}, "roi": pd.DataFrame}}

    Retourne
    --------
    DataFrame avec une ligne par marché + flag outlier
    """
    rows = []
    for market, data in results_by_market.items():
        row = {"market": market}
        if "metrics" in data:
            row.update(data["metrics"])
        if "roi" in data and isinstance(data["roi"], pd.DataFrame):
            roi_df = data["roi"]
            if "roi" in roi_df.columns:
                row["roi_mean"]  = roi_df["roi"].mean()
                row["roi_max"]   = roi_df["roi"].max()
                row["roi_min"]   = roi_df["roi"].min()
        rows.append(row)

    df = pd.DataFrame(rows)

    # Détection outliers sur R²
    if "r2" in df.columns:
        r2_mean = df["r2"].mean()
        r2_std  = df["r2"].std()
        df["r2_outlier"] = df["r2"] < (r2_mean - 2 * r2_std)

    # Détection outliers sur MAPE
    if "mape" in df.columns:
        mape_mean = df["mape"].mean()
        mape_std  = df["mape"].std()
        df["mape_outlier"] = df["mape"] > (mape_mean + 2 * mape_std)

    df["needs_review"] = df.get("r2_outlier", False) | df.get("mape_outlier", False)

    return df.sort_values("r2", ascending=False).reset_index(drop=True)


# ── 4. Rapport complet ────────────────────────────────────────────────────────

def full_validation_report(
    idata,
    df: pd.DataFrame,
    y_pred: np.ndarray,
    market: str = "unknown",
    output_path: Optional[Path] = None,
) -> dict:
    """
    Génère un rapport de validation complet pour un marché.

    Inclut :
      - Métriques de performance (R², MAPE, NRMSE)
      - Diagnostics MCMC (R-hat, ESS)
      - Posterior Predictive Check
      - Résumé textuel

    Paramètres
    ----------
    idata       : az.InferenceData
    df          : données de test
    y_pred      : prédictions centrales
    market      : code marché
    output_path : si fourni, sauvegarde le rapport en CSV

    Retourne
    --------
    dict avec toutes les métriques et diagnostics
    """
    from src.training.model_diagnostics import check_rhat, check_ess

    logger.info(f"[{market}] Génération du rapport de validation...")

    report = {"market": market}

    # 1. Métriques prédictives
    y_true = df["revenue"].values
    report["metrics"] = compute_all_metrics(y_true, y_pred)

    # 2. Diagnostics MCMC
    if idata is not None:
        report["rhat_issues"] = check_rhat(idata)
        report["ess_issues"]  = check_ess(idata)
        report["convergence_ok"] = (
            len(report["rhat_issues"]) == 0 and
            len(report["ess_issues"])  == 0
        )

    # 3. PPC
    report["ppc"] = posterior_predictive_check(idata, df)

    # 4. Résumé
    m = report["metrics"]
    report["summary"] = (
        f"Marché {market} | "
        f"R²={m.get('r2', 0):.3f} | "
        f"MAPE={m.get('mape', 0):.1f}% | "
        f"NRMSE={m.get('nrmse', 0):.3f} | "
        f"Convergence={'✅' if report.get('convergence_ok') else '⚠️'}"
    )

    logger.info(report["summary"])

    # 5. Sauvegarde
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        flat = {
            "market": market,
            **report["metrics"],
            "convergence_ok": report.get("convergence_ok", None),
            "ppc_coverage_94": report.get("ppc", {}).get("coverage_94"),
        }
        pd.DataFrame([flat]).to_csv(output_path, index=False)
        logger.info(f"Rapport sauvegardé → {output_path}")

    return report