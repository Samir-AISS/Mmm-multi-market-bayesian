"""
model_diagnostics.py
--------------------
Diagnostics de convergence MCMC et validation du modèle bayésien.
Fonctionne avec arviz.InferenceData retourné par PyMC.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def check_rhat(idata, threshold: float = 1.01) -> Dict[str, float]:
    """
    Vérifie la convergence R-hat (Gelman-Rubin) de tous les paramètres.
    R-hat < 1.01 = bonne convergence.

    Retourne les paramètres problématiques {param: rhat_value}.
    """
    try:
        import arviz as az
        summary      = az.summary(idata, var_names=None)
        problematic  = summary[summary["r_hat"] >= threshold]["r_hat"]
        result       = problematic.to_dict()
        if result:
            logger.warning(
                f"⚠️  {len(result)} paramètre(s) avec R-hat ≥ {threshold}: "
                f"{list(result.keys())}"
            )
        else:
            logger.info(f"✅ R-hat OK : tous < {threshold}")
        return result
    except ImportError:
        logger.error("arviz non installé. pip install arviz")
        return {}


def check_ess(idata, min_ess: int = 400) -> Dict[str, float]:
    """
    Vérifie l'Effective Sample Size (ESS).
    ESS > 400 = estimation fiable.

    Retourne les paramètres avec ESS insuffisant.
    """
    try:
        import arviz as az
        summary = az.summary(idata)
        low_ess = summary[summary["ess_bulk"] < min_ess]["ess_bulk"]
        result  = low_ess.to_dict()
        if result:
            logger.warning(f"⚠️  {len(result)} paramètre(s) avec ESS < {min_ess}")
        else:
            logger.info(f"✅ ESS OK : tous > {min_ess}")
        return result
    except ImportError:
        logger.error("arviz non installé")
        return {}


def compute_loo(idata) -> dict:
    """
    Leave-One-Out Cross Validation (LOO-CV) via arviz.
    Retourne elpd_loo, p_loo et les warnings éventuels.
    """
    try:
        import arviz as az
        loo_result = az.loo(idata)
        logger.info(
            f"LOO: elpd={loo_result.elpd_loo:.2f}, "
            f"p_loo={loo_result.p_loo:.2f}"
        )
        return {
            "elpd_loo": float(loo_result.elpd_loo),
            "p_loo":    float(loo_result.p_loo),
            "se":       float(loo_result.se),
        }
    except Exception as e:
        logger.error(f"LOO échoué : {e}")
        return {}


def compute_ppc_metrics(
    y_true: np.ndarray,
    ppc_samples: np.ndarray,
) -> dict:
    """
    Posterior Predictive Check : compare y_true aux prédictions postérieures.

    Paramètres
    ----------
    y_true      : array (n_obs,) — valeurs observées
    ppc_samples : array (n_samples, n_obs) — échantillons postérieurs

    Retourne
    --------
    dict avec mean_pred, coverage_94, coverage_50, mae, mape
    """
    mean_pred = ppc_samples.mean(axis=0)
    lower_94  = np.percentile(ppc_samples,  3, axis=0)
    upper_94  = np.percentile(ppc_samples, 97, axis=0)
    lower_50  = np.percentile(ppc_samples, 25, axis=0)
    upper_50  = np.percentile(ppc_samples, 75, axis=0)

    coverage_94 = float(np.mean((y_true >= lower_94) & (y_true <= upper_94)) * 100)
    coverage_50 = float(np.mean((y_true >= lower_50) & (y_true <= upper_50)) * 100)
    mae         = float(np.mean(np.abs(y_true - mean_pred)))
    mape_val    = float(
        np.mean(np.abs((y_true - mean_pred) / np.where(y_true != 0, y_true, 1))) * 100
    )

    logger.info(
        f"PPC — Couverture 94%: {coverage_94:.1f}% | "
        f"50%: {coverage_50:.1f}% | MAE: {mae:,.0f}"
    )

    return {
        "mean_prediction": mean_pred,
        "coverage_94pct":  coverage_94,
        "coverage_50pct":  coverage_50,
        "mae":             mae,
        "mape":            mape_val,
    }


def plot_trace(idata, params: Optional[List[str]] = None) -> None:
    """
    Trace plots pour diagnostiquer visuellement la convergence MCMC.
    Affiche les chaînes et les distributions postérieures.

    Paramètres
    ----------
    idata  : az.InferenceData
    params : liste de paramètres à tracer (défaut : tous)
    """
    try:
        import arviz as az
        import matplotlib.pyplot as plt

        kwargs = {"var_names": params} if params else {}
        az.plot_trace(idata, **kwargs)
        plt.tight_layout()
        plt.show()
        logger.info("Trace plots générés")
    except ImportError:
        logger.error("arviz ou matplotlib non installé")
    except Exception as e:
        logger.error(f"plot_trace échoué : {e}")


def plot_energy(idata) -> None:
    """
    Energy plot — détecte les problèmes de géométrie MCMC (divergences HMC).
    Un bon modèle a une distribution d'énergie centrée et symétrique.
    """
    try:
        import arviz as az
        import matplotlib.pyplot as plt

        az.plot_energy(idata)
        plt.tight_layout()
        plt.show()
        logger.info("Energy plot généré")
    except ImportError:
        logger.error("arviz ou matplotlib non installé")
    except Exception as e:
        logger.error(f"plot_energy échoué : {e}")


def check_divergences(idata) -> dict:
    """
    Compte les divergences MCMC (HMC/NUTS).
    Toute divergence indique un problème de modélisation ou de prior.

    Retourne
    --------
    dict avec n_divergences, pct_divergences, has_divergences
    """
    try:
        import arviz as az

        divergences = idata.sample_stats["diverging"].values
        n_div       = int(divergences.sum())
        n_total     = int(divergences.size)
        pct_div     = n_div / n_total * 100

        if n_div > 0:
            logger.warning(
                f"⚠️  {n_div} divergences ({pct_div:.2f}%) — "
                "reparamétrisez ou réduisez step_size"
            )
        else:
            logger.info("✅ Aucune divergence MCMC")

        return {
            "n_divergences":   n_div,
            "pct_divergences": round(pct_div, 4),
            "has_divergences": n_div > 0,
        }

    except Exception as e:
        logger.error(f"check_divergences échoué : {e}")
        return {"n_divergences": -1, "has_divergences": None}


def full_diagnostics_report(
    idata,
    df: pd.DataFrame,
    y_col: str = "revenue",
) -> dict:
    """
    Rapport complet de diagnostics MCMC.

    Inclut : R-hat, ESS, divergences, LOO-CV.
    Retourne un dict structuré avec toutes les métriques.
    """
    logger.info("═" * 50)
    logger.info("RAPPORT DE DIAGNOSTICS MCMC")
    logger.info("═" * 50)

    report = {
        "rhat_issues":  check_rhat(idata),
        "ess_issues":   check_ess(idata),
        "divergences":  check_divergences(idata),
        "loo":          compute_loo(idata),
        "converged":    False,
    }

    # Convergence globale = 0 R-hat + 0 ESS + 0 divergence
    report["converged"] = (
        len(report["rhat_issues"]) == 0
        and len(report["ess_issues"]) == 0
        and not report["divergences"].get("has_divergences", True)
    )

    status = "✅ CONVERGÉ" if report["converged"] else "⚠️  NON CONVERGÉ"
    logger.info(f"Statut final : {status}")
    logger.info("═" * 50)

    return report