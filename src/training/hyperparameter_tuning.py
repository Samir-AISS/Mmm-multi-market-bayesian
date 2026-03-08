"""
hyperparameter_tuning.py
------------------------
Optimisation des hyperparamètres du modèle bayésien MMM.

Approche :
  - Grid search sur les priors (decay, K, S)
  - Évaluation via LOO-CV (Leave-One-Out Cross Validation)
  - Sélection du meilleur modèle par marché
  - Sauvegarde des résultats dans results/reports/

Usage :
    from src.training.hyperparameter_tuning import tune_market
    best_config = tune_market("FR", df_fr)
"""

import itertools
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all_metrics
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

ROOT        = Path(__file__).parent.parent.parent
REPORTS_DIR = ROOT / "results" / "reports"


# ── Grille de recherche par défaut ────────────────────────────────────────────

DEFAULT_PARAM_GRID = {
    "tv_decay":       [0.3, 0.5, 0.7],
    "facebook_decay": [0.1, 0.3, 0.5],
    "search_decay":   [0.0, 0.1, 0.3],
    "ooh_decay":      [0.2, 0.4, 0.6],
    "print_decay":    [0.1, 0.3, 0.5],
    "hill_S":         [1.5, 2.0, 3.0],
}

# Grille réduite pour tests rapides
FAST_PARAM_GRID = {
    "tv_decay":       [0.3, 0.6],
    "facebook_decay": [0.2, 0.4],
    "search_decay":   [0.1, 0.3],
    "ooh_decay":      [0.3, 0.5],
    "print_decay":    [0.2, 0.4],
    "hill_S":         [1.5, 2.0],
}


# ── Évaluation d'une configuration ───────────────────────────────────────────

def evaluate_config(
    config: dict,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> dict:
    """
    Entraîne le modèle avec une configuration donnée et évalue sur df_val.

    Paramètres
    ----------
    config   : dict de paramètres (decay par canal, hill_S, etc.)
    df_train : données d'entraînement
    df_val   : données de validation

    Retourne
    --------
    dict avec : config, r2, mape, nrmse, duration_s
    """
    from src.models.bayesian_mmm import BayesianMMM

    start = time.time()

    try:
        model = BayesianMMM(config=config)
        model.build_model(df_train)
        model.fit(
            df_train,
            draws=500,    # réduit pour la tuning (vitesse)
            tune=500,
            chains=2,
        )

        y_pred  = model.predict(df_val)
        y_true  = df_val["revenue"].values
        metrics = compute_all_metrics(y_true, y_pred)

        return {
            "config":     config,
            "r2":         metrics["r2"],
            "mape":       metrics["mape"],
            "nrmse":      metrics["nrmse"],
            "duration_s": round(time.time() - start, 1),
            "status":     "success",
        }

    except Exception as e:
        return {
            "config":     config,
            "error":      str(e),
            "duration_s": round(time.time() - start, 1),
            "status":     "failed",
        }


# ── LOO-CV pour sélection du modèle ──────────────────────────────────────────

def loo_cv_score(idata) -> float:
    """
    Calcule le score LOO-CV (Leave-One-Out Cross Validation) via arviz.
    Plus le score elpd_loo est élevé, meilleur est le modèle.

    Paramètres
    ----------
    idata : az.InferenceData

    Retourne
    --------
    float : elpd_loo (Expected Log Pointwise Predictive Density)
    """
    try:
        import arviz as az
        loo_result = az.loo(idata, pointwise=False)
        return float(loo_result.elpd_loo)
    except Exception as e:
        logger.warning(f"LOO-CV échoué : {e}")
        return -np.inf


# ── Grid Search ───────────────────────────────────────────────────────────────

def grid_search(
    df: pd.DataFrame,
    param_grid: Optional[dict] = None,
    val_ratio: float = 0.2,
    market: str = "unknown",
    fast_mode: bool = False,
) -> pd.DataFrame:
    """
    Grid search sur les hyperparamètres du modèle bayésien.

    Paramètres
    ----------
    df         : DataFrame complet (un seul marché)
    param_grid : grille de paramètres (défaut : DEFAULT_PARAM_GRID)
    val_ratio  : proportion des données pour la validation
    market     : code marché (pour les logs)
    fast_mode  : utilise FAST_PARAM_GRID (moins de combinaisons)

    Retourne
    --------
    DataFrame trié par R² décroissant avec toutes les combinaisons testées
    """
    if param_grid is None:
        param_grid = FAST_PARAM_GRID if fast_mode else DEFAULT_PARAM_GRID

    # Split temporel train/val
    df = df.sort_values("week").reset_index(drop=True)
    n_val   = int(len(df) * val_ratio)
    df_train = df.iloc[:-n_val].copy()
    df_val   = df.iloc[-n_val:].copy()

    # Générer toutes les combinaisons
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    n_total = len(combinations)

    logger.info(
        f"[{market}] Grid search — {n_total} combinaisons | "
        f"train={len(df_train)} val={len(df_val)}"
    )

    results = []
    for i, combo in enumerate(combinations, 1):
        config = dict(zip(keys, combo))
        logger.info(f"[{market}] Combinaison {i}/{n_total} : {config}")

        result = evaluate_config(config, df_train, df_val)
        results.append(result)

        if result["status"] == "success":
            logger.info(
                f"[{market}]   → R²={result['r2']:.3f} | "
                f"MAPE={result['mape']:.1f}% | "
                f"durée={result['duration_s']}s"
            )
        else:
            logger.warning(f"[{market}]   → ERREUR : {result.get('error')}")

    # Construire le DataFrame résultats
    rows = []
    for r in results:
        row = {k: v for k, v in r.get("config", {}).items()}
        row["status"]     = r["status"]
        row["r2"]         = r.get("r2")
        row["mape"]       = r.get("mape")
        row["nrmse"]      = r.get("nrmse")
        row["duration_s"] = r.get("duration_s")
        if r["status"] == "failed":
            row["error"] = r.get("error")
        rows.append(row)

    results_df = pd.DataFrame(rows)
    if "r2" in results_df.columns:
        results_df = results_df.sort_values("r2", ascending=False).reset_index(drop=True)

    return results_df


# ── Sélection du meilleur modèle ──────────────────────────────────────────────

def select_best_config(
    results_df: pd.DataFrame,
    metric: str = "r2",
    min_r2: float = 0.70,
) -> Optional[dict]:
    """
    Sélectionne la meilleure configuration selon une métrique.

    Paramètres
    ----------
    results_df : DataFrame retourné par grid_search()
    metric     : métrique de sélection ("r2" | "mape" | "nrmse")
    min_r2     : R² minimum acceptable (sinon retourne None)

    Retourne
    --------
    dict de la meilleure configuration, ou None si aucune satisfaisante
    """
    success_df = results_df[results_df["status"] == "success"].copy()

    if success_df.empty:
        logger.error("Aucune configuration valide trouvée")
        return None

    ascending = metric in ("mape", "nrmse")
    best_row  = success_df.sort_values(metric, ascending=ascending).iloc[0]

    if best_row.get("r2", 0) < min_r2:
        logger.warning(
            f"Meilleur R²={best_row['r2']:.3f} < seuil {min_r2}. "
            "Considérez d'élargir la grille de recherche."
        )

    param_keys = [k for k in results_df.columns
                  if k not in ("status", "r2", "mape", "nrmse", "duration_s", "error")]
    best_config = {k: best_row[k] for k in param_keys if k in best_row}

    logger.info(
        f"✅ Meilleure config : {best_config} | "
        f"R²={best_row.get('r2'):.3f} | "
        f"MAPE={best_row.get('mape'):.1f}%"
    )
    return best_config


# ── Tuning complet pour un marché ─────────────────────────────────────────────

def tune_market(
    market: str,
    df: pd.DataFrame,
    param_grid: Optional[dict] = None,
    fast_mode: bool = True,
    save_results: bool = True,
) -> Tuple[Optional[dict], pd.DataFrame]:
    """
    Pipeline complet de tuning pour un marché.

    Paramètres
    ----------
    market      : code marché (ex: "FR")
    df          : DataFrame du marché
    param_grid  : grille personnalisée (optionnel)
    fast_mode   : utilise la grille réduite
    save_results: sauvegarde les résultats en CSV

    Retourne
    --------
    (best_config, results_df) :
      - best_config : dict de la meilleure configuration
      - results_df  : tous les résultats du grid search
    """
    logger.info(f"🔍 Démarrage hyperparameter tuning — Marché {market}")
    start = time.time()

    results_df  = grid_search(df, param_grid, market=market, fast_mode=fast_mode)
    best_config = select_best_config(results_df)

    # Sauvegarde
    if save_results:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REPORTS_DIR / f"tuning_{market}.csv"
        results_df.to_csv(out_path, index=False)
        logger.info(f"Résultats tuning → {out_path}")

    logger.info(f"✅ Tuning {market} terminé en {time.time()-start:.1f}s")
    return best_config, results_df


# ── Tuning multi-marchés ──────────────────────────────────────────────────────

def tune_all_markets(
    markets: Optional[List[str]] = None,
    fast_mode: bool = True,
    n_jobs: int = 1,
) -> Dict[str, Optional[dict]]:
    """
    Lance le tuning pour tous les marchés.

    Paramètres
    ----------
    markets   : liste des marchés (défaut : tous)
    fast_mode : grille réduite pour aller plus vite
    n_jobs    : parallélisation (1 = séquentiel)

    Retourne
    --------
    dict {market: best_config}
    """
    import pandas as pd
    from src.data.data_loader import load_market_data

    if markets is None:
        markets = ["FR", "DE", "UK", "IT", "ES", "NL", "BE", "PL", "SE", "NO"]

    best_configs = {}

    if n_jobs == 1:
        # Séquentiel
        for market in markets:
            df = load_market_data(market)
            best_config, _ = tune_market(market, df, fast_mode=fast_mode)
            best_configs[market] = best_config
    else:
        # Parallèle
        from joblib import Parallel, delayed

        def _tune(market):
            df = load_market_data(market)
            best_config, _ = tune_market(market, df, fast_mode=fast_mode)
            return market, best_config

        results = Parallel(n_jobs=n_jobs)(delayed(_tune)(m) for m in markets)
        best_configs = dict(results)

    # Résumé
    logger.info("=" * 50)
    logger.info("RÉSUMÉ TUNING MULTI-MARCHÉS")
    for market, config in best_configs.items():
        if config:
            logger.info(f"  {market} : {config}")
        else:
            logger.warning(f"  {market} : aucune config valide")
    logger.info("=" * 50)

    # Sauvegarde globale
    summary_rows = [
        {"market": m, **(c or {"error": "no valid config"})}
        for m, c in best_configs.items()
    ]
    summary_df = pd.DataFrame(summary_rows)
    out_path   = REPORTS_DIR / "tuning_summary_all_markets.csv"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    logger.info(f"Résumé global → {out_path}")

    return best_configs