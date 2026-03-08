"""
run_pipeline.py
---------------
Script d'orchestration du pipeline complet MMM.
Alternative légère à Airflow pour exécution locale.

Usage:
    python pipelines/orchestration/run_pipeline.py
    python pipelines/orchestration/run_pipeline.py --markets FR,DE
    python pipelines/orchestration/run_pipeline.py --force --draws 1000
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Résolution du path projet ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.multi_market_generator import generate_full_dataset
from src.data.data_validator          import validate
from src.training.distributed_trainer import train_all_markets
from src.utils.logging_config          import get_logger

logger = get_logger(__name__)

ALL_MARKETS  = ["FR", "DE", "UK", "IT", "ES", "NL", "BE", "PL", "SE", "NO"]
DATA_PATH    = PROJECT_ROOT / "data" / "synthetic" / "mmm_multi_market.csv"
REPORTS_DIR  = PROJECT_ROOT / "results" / "reports"


# ── Étape 1 : Génération des données ──────────────────────────────────────
def step_generate(force: bool = False) -> pd.DataFrame:
    if DATA_PATH.exists() and not force:
        logger.info(
            f"✅ Données existantes → {DATA_PATH} "
            f"(utilisez --force pour régénérer)"
        )
        return pd.read_csv(DATA_PATH, parse_dates=["date"])

    logger.info("⏳ Génération des données synthétiques...")
    t0 = time.time()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_full_dataset(output_path=str(DATA_PATH))
    logger.info(f"✅ Données générées en {time.time()-t0:.1f}s — {len(df)} lignes")
    return df


# ── Étape 2 : Validation des données ──────────────────────────────────────
def step_validate(df: pd.DataFrame) -> bool:
    logger.info("⏳ Étape 2/5 — Validation des données...")
    t0 = time.time()
    report = validate(df)
    report.print_summary()
    ok = report.n_errors == 0
    status = "OK" if ok else f"{report.n_errors} ERREUR(S)"
    logger.info(f"✅ Validation {status} — {report.n_tests} tests en {time.time()-t0:.1f}s")
    return ok


# ── Étape 3 : Entraînement ─────────────────────────────────────────────────
def step_train(markets: list, config: dict, n_jobs: int) -> pd.DataFrame:
    logger.info(f"⏳ Étape 3/5 — Entraînement sur {len(markets)} marché(s) | n_jobs={n_jobs}")
    t0 = time.time()
    df_results = train_all_markets(markets, config=config, n_jobs=n_jobs)
    n_ok = (df_results["status"] == "success").sum()
    logger.info(f"✅ Entraînement terminé en {time.time()-t0:.1f}s — {n_ok} succès / {len(markets)} échec(s)" if n_ok < len(markets) else
                f"✅ Entraînement terminé en {time.time()-t0:.1f}s — {n_ok}/{len(markets)} succès")
    return df_results


# ── Étape 4 : Évaluation ──────────────────────────────────────────────────
def step_evaluate(df_results: pd.DataFrame) -> pd.DataFrame:
    logger.info("⏳ Étape 4/5 — Évaluation des modèles...")
    t0 = time.time()
    successful = df_results[df_results["status"] == "success"]
    if successful.empty:
        logger.warning("⚠️  Aucun modèle entraîné avec succès — évaluation ignorée")
    else:
        logger.info(f"  R²   moyen : {successful['r2'].mean():.3f}")
        logger.info(f"  MAPE moyen : {successful['mape'].mean():.1f}%")
    logger.info(f"✅ Évaluation terminée en {time.time()-t0:.1f}s")
    return df_results


# ── Étape 5 : Rapport final ────────────────────────────────────────────────
def step_report(df_results: pd.DataFrame) -> Path:
    logger.info("⏳ Étape 5/5 — Génération du rapport...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_path = REPORTS_DIR / f"pipeline_report_{timestamp}.csv"
    df_results.to_csv(report_path, index=False)

    # Affichage console
    sep = "═" * 65
    print(f"\n{sep}")
    print(f"  📊 MMM MULTI-MARKET — RAPPORT PIPELINE")
    print(f"  📅 {timestamp}")
    print(sep)

    for _, row in df_results.iterrows():
        if row["status"] == "success":
            r2_str   = f"{row['r2']:.3f}" if row["r2"]   is not None else "  N/A"
            mape_str = f"{row['mape']:.1f}%" if row["mape"] is not None else "  N/A"
            conv     = "✅" if row["convergence_ok"] else "⚠️ "
            icon     = "✅"
        else:
            r2_str, mape_str, conv, icon = "  N/A", "  N/A", "⚠️ ", "❌"

        print(f"  {icon} {row['market']:3s} | R²={r2_str:>6} | MAPE={mape_str:>6} | Conv={conv}")

    n_ok = (df_results["status"] == "success").sum()
    print(sep)
    print(f"  TOTAL : {n_ok}/{len(df_results)} marchés entraînés avec succès")
    print(sep + "\n")

    logger.info(f"✅ Rapport sauvegardé → {report_path}")
    return report_path


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pipeline MMM Multi-Market")
    parser.add_argument("--markets",   default="all",
                        help="Marchés (ex: FR,DE ou 'all')")
    parser.add_argument("--n-jobs",    type=int, default=-1,
                        help="Nombre de workers parallèles (-1 = tous)")
    parser.add_argument("--draws",     type=int, default=500)
    parser.add_argument("--tune",      type=int, default=500)
    parser.add_argument("--chains",    type=int, default=2)
    parser.add_argument("--force",     action="store_true",
                        help="Régénérer les données même si elles existent")
    parser.add_argument("--skip-train", action="store_true",
                        help="Sauter l'entraînement (validation seule)")
    args = parser.parse_args()

    markets = ALL_MARKETS if args.markets == "all" else args.markets.split(",")
    config  = {
        "draws":       args.draws,
        "tune":        args.tune,
        "chains":      args.chains,
        "random_seed": 42,
    }

    t_total = time.time()
    logger.info("🚀 Démarrage du pipeline MMM Multi-Market")
    logger.info(f"   Marchés : {markets}")
    logger.info(f"   n_jobs  : {args.n_jobs}")

    # Étape 1 — Données
    df = step_generate(force=args.force)

    # Étape 2 — Validation
    valid = step_validate(df)
    if not valid:
        logger.error("❌ Données invalides — pipeline interrompu")
        sys.exit(1)

    # Étape 3 — Entraînement
    if args.skip_train:
        logger.info("⏭️  Entraînement ignoré (--skip-train)")
        df_results = pd.DataFrame([
            {"market": m, "status": "skipped", "r2": None, "mape": None,
             "nrmse": None, "rmse": None, "best_channel": None,
             "best_roi": None, "duration": 0, "convergence_ok": False, "error": None}
            for m in markets
        ])
    else:
        df_results = step_train(markets, config, n_jobs=args.n_jobs)

    # Étape 4 — Évaluation
    df_results = step_evaluate(df_results)

    # Étape 5 — Rapport
    step_report(df_results)

    logger.info(f"🏁 Pipeline terminé en {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()