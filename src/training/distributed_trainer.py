"""
distributed_trainer.py
-----------------------
Entraînement parallèle du modèle MMM sur tous les marchés.
"""

import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from joblib import Parallel, delayed

from src.data.data_loader import load_market_data, split_train_test
from src.evaluation.metrics import compute_all_metrics
from src.models.bayesian_mmm import BayesianMMM
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"


def train_single_market(market: str, config: dict) -> dict:
    """Entraîne le modèle pour un marché unique."""
    t0 = time.time()
    logger.info(f"[{market}] Démarrage entraînement...")

    try:
        df = load_market_data(market)
        df_train, df_test = split_train_test(df, test_ratio=0.2)
        logger.info(f"[{market}] Données : {len(df_train)} train / {len(df_test)} test")

        draws  = config.get("draws",  500)
        tune   = config.get("tune",   500)
        chains = config.get("chains", 2)
        seed   = config.get("random_seed", 42)

        model = BayesianMMM(market=market)
        model.fit(df_train, draws=draws, tune=tune, chains=chains, random_seed=seed)

        y_pred  = model.predict(df_test)
        metrics = compute_all_metrics(df_test["revenue"].values, y_pred)
        roi_df  = model.get_roi(df_train)
        best_ch  = roi_df.iloc[0]["channel"] if not roi_df.empty else "N/A"
        best_roi = roi_df.iloc[0]["roi"]     if not roi_df.empty else 0.0

        _save_model(model, market)

        duration = time.time() - t0
        logger.info(
            f"[{market}] SUCCESS R²={metrics['r2']:.3f} | "
            f"MAPE={metrics['mape']:.1f}% | Durée={duration:.1f}s"
        )

        return {
            "market": market, "status": "success",
            "r2": metrics["r2"], "mape": metrics["mape"],
            "nrmse": metrics["nrmse"], "rmse": metrics["rmse"],
            "best_channel": best_ch, "best_roi": best_roi,
            "duration": duration, "convergence_ok": True, "error": None,
        }

    except Exception as e:
        duration = time.time() - t0
        logger.error(f"[{market}] ERREUR : {e}")
        logger.debug(traceback.format_exc())
        return {
            "market": market, "status": "error",
            "r2": None, "mape": None, "nrmse": None, "rmse": None,
            "best_channel": None, "best_roi": None,
            "duration": duration, "convergence_ok": False, "error": str(e),
        }


def _save_model(model: BayesianMMM, market: str) -> Optional[Path]:
    """Sauvegarde le modèle dans results/models/."""
    try:
        models_dir = RESULTS_DIR / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        if model.idata is not None:
            import arviz as az
            path = models_dir / f"mmm_{market}.nc"
            az.to_netcdf(model.idata, str(path))
            logger.info(f"[{market}] Modèle sauvegardé → {path}")
            return path
    except Exception as e:
        logger.warning(f"[{market}] Sauvegarde ignorée : {e}")
    return None


def train_all_markets(
    markets: List[str],
    config: Optional[dict] = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Entraîne en parallèle tous les marchés spécifiés."""
    if config is None:
        config = {"draws": 500, "tune": 500, "chains": 2, "random_seed": 42}

    logger.info(f"Démarrage entraînement parallèle — {len(markets)} marchés | n_jobs={n_jobs}")
    t0 = time.time()

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(train_single_market)(market, config)
        for market in markets
    )

    df_results = pd.DataFrame(results)
    duration   = time.time() - t0
    n_success  = (df_results["status"] == "success").sum()
    n_errors   = (df_results["status"] == "error").sum()

    logger.info("=" * 55)
    logger.info(f"Entraînement terminé en {duration:.1f}s")
    logger.info(f"RÉSUMÉ : {n_success} succès / {n_errors} échec(s)")
    logger.info("=" * 55)

    reports_dir = RESULTS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "training_summary.csv"
    df_results.to_csv(report_path, index=False)
    logger.info(f"Rapport sauvegardé → {report_path}")

    return df_results


if __name__ == "__main__":
    import argparse
    ALL_MARKETS = ["FR", "DE", "UK", "IT", "ES", "NL", "BE", "PL", "SE", "NO"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", default="all")
    parser.add_argument("--n-jobs",  type=int, default=-1)
    parser.add_argument("--draws",   type=int, default=500)
    parser.add_argument("--tune",    type=int, default=500)
    parser.add_argument("--chains",  type=int, default=2)
    args = parser.parse_args()

    markets = ALL_MARKETS if args.markets == "all" else args.markets.split(",")
    config  = {"draws": args.draws, "tune": args.tune,
               "chains": args.chains, "random_seed": 42}

    df = train_all_markets(markets, config=config, n_jobs=args.n_jobs)
    print(df[["market", "status", "r2", "mape", "duration"]].to_string(index=False))