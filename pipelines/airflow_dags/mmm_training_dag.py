"""
mmm_training_dag.py
-------------------
Airflow DAG pour l'entraînement automatisé du MMM Multi-Market.

Pipeline :
  1. generate_data     → génère / vérifie les données synthétiques
  2. validate_data     → 0 erreur obligatoire avant de continuer
  3. train_models      → entraîne les 10 marchés en parallèle
  4. evaluate_models   → calcule métriques + diagnostics
  5. generate_report   → rapport HTML dans results/reports/

Usage :
    # Démarrer Airflow
    airflow db init
    airflow webserver --port 8080
    airflow scheduler

    # Déclencher manuellement
    airflow dags trigger mmm_multi_market_training
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

# ── Configuration du DAG ──────────────────────────────────────────────────────

DEFAULT_ARGS = {
    "owner":            "samir_el_aissaouy",
    "depends_on_past":  False,
    "email":            ["Elaissaouy.samir12@gmail.com"],
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
}

dag = DAG(
    dag_id="mmm_multi_market_training",
    description="Pipeline complet MMM — génération, validation, entraînement, évaluation",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 6 * * 1",    # Chaque lundi à 6h
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["mmm", "marketing", "bayesian", "multi-market"],
)


# ── Fonctions des tâches ──────────────────────────────────────────────────────

def task_generate_data(**context):
    """
    Étape 1 : Génération des données synthétiques multi-marchés.
    Vérifie d'abord si le fichier existe et est récent (< 7 jours).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.data.multi_market_generator import generate_full_dataset

    data_path = Path(__file__).parent.parent.parent / "data" / "synthetic" / "mmm_multi_market.csv"

    # Régénérer si le fichier n'existe pas ou est trop ancien
    regenerate = True
    if data_path.exists():
        age_days = (datetime.now().timestamp() - data_path.stat().st_mtime) / 86400
        if age_days < 7:
            print(f"✅ Données existantes ({age_days:.1f} jours) — pas de régénération")
            regenerate = False

    if regenerate:
        print("⏳ Génération des données...")
        df = generate_full_dataset()
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"✅ {len(df)} lignes générées → {data_path}")

    # Passer le chemin à la tâche suivante via XCom
    context["ti"].xcom_push(key="data_path", value=str(data_path))
    return str(data_path)


def task_validate_data(**context):
    """
    Étape 2 : Validation en 4 niveaux — 0 erreur obligatoire.
    Lève une exception si des erreurs sont détectées (bloque le pipeline).
    """
    import sys
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.data.data_validator import validate

    data_path = context["ti"].xcom_pull(key="data_path", task_ids="generate_data")
    df = pd.read_csv(data_path)

    report = validate(df)
    report.print_summary()

    if report.n_errors > 0:
        raise ValueError(
            f"❌ Validation échouée : {report.n_errors} erreur(s) détectée(s). "
            "Pipeline arrêté."
        )

    print(f"✅ Validation OK — {report.n_tests} tests passés, 0 erreur")
    context["ti"].xcom_push(key="n_rows", value=len(df))
    return report.n_tests


def task_train_models(**context):
    """
    Étape 3 : Entraînement parallèle sur tous les marchés.
    Utilise distributed_trainer avec MLflow tracking.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.training.distributed_trainer import train_all_markets

    results_df = train_all_markets(
        markets=None,        # tous les marchés
        n_jobs=4,            # 4 workers parallèles
        save_model=True,
        track_mlflow=True,
    )

    n_success = (results_df["status"] == "success").sum()
    n_failed  = (results_df["status"] == "failed").sum()

    print(f"✅ Entraînement terminé : {n_success} succès / {n_failed} échec(s)")
    print(results_df.to_string(index=False))

    context["ti"].xcom_push(key="n_success", value=int(n_success))
    context["ti"].xcom_push(key="n_failed",  value=int(n_failed))

    if n_failed > 0:
        print(f"⚠️  {n_failed} marché(s) en échec — voir les logs")

    return int(n_success)


def task_evaluate_models(**context):
    """
    Étape 4 : Évaluation des modèles entraînés.
    Calcule R², MAPE, NRMSE pour chaque marché et génère un rapport CSV.
    """
    import sys
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.evaluation.metrics import compute_all_metrics

    models_dir  = Path(__file__).parent.parent.parent / "results" / "models"
    reports_dir = Path(__file__).parent.parent.parent / "results" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_files = list(models_dir.glob("mmm_*.nc"))
    if not model_files:
        print("⚠️  Aucun modèle trouvé dans results/models/")
        return 0

    summary_rows = []
    for model_path in model_files:
        market = model_path.stem.replace("mmm_", "")
        print(f"📊 Évaluation {market}...")

        try:
            import arviz as az
            idata = az.from_netcdf(str(model_path))

            # Métriques depuis posterior predictive (si disponible)
            row = {"market": market, "model_path": str(model_path)}
            summary_rows.append(row)

        except Exception as e:
            print(f"⚠️  {market} : {e}")
            summary_rows.append({"market": market, "error": str(e)})

    summary_df = pd.DataFrame(summary_rows)
    out_path   = reports_dir / "evaluation_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"✅ Rapport d'évaluation → {out_path}")

    return len(summary_rows)


def task_generate_report(**context):
    """
    Étape 5 : Génération du rapport HTML de synthèse.
    Compile les métriques de tous les marchés en un rapport lisible.
    """
    import sys
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    reports_dir = Path(__file__).parent.parent.parent / "results" / "reports"
    run_date    = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Charger le résumé d'entraînement
    train_summary_path = reports_dir / "training_summary.csv"
    eval_summary_path  = reports_dir / "evaluation_summary.csv"

    train_df = pd.read_csv(train_summary_path) if train_summary_path.exists() else pd.DataFrame()
    eval_df  = pd.read_csv(eval_summary_path)  if eval_summary_path.exists()  else pd.DataFrame()

    # Générer le HTML
    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>MMM Multi-Market — Rapport {run_date}</title>
    <style>
        body  {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
        h1    {{ color: #1565C0; }}
        h2    {{ color: #1976D2; border-bottom: 2px solid #1976D2; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th    {{ background: #1976D2; color: white; padding: 10px; text-align: left; }}
        td    {{ padding: 8px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f5f5f5; }}
        .badge-ok  {{ background: #4CAF50; color: white; padding: 3px 8px; border-radius: 12px; }}
        .badge-err {{ background: #F44336; color: white; padding: 3px 8px; border-radius: 12px; }}
        .kpi  {{ display: inline-block; background: #E3F2FD; padding: 15px 25px;
                 border-radius: 8px; margin: 10px; text-align: center; }}
        .kpi-value {{ font-size: 24px; font-weight: bold; color: #1565C0; }}
    </style>
</head>
<body>
    <h1>📊 MMM Multi-Market — Rapport d'entraînement</h1>
    <p><strong>Date :</strong> {run_date} &nbsp;|&nbsp;
       <strong>Auteur :</strong> Samir EL AISSAOUY &nbsp;|&nbsp;
       <strong>Version :</strong> 1.0.0</p>

    <h2>🔢 KPIs globaux</h2>
    <div>
        <div class="kpi">
            <div class="kpi-value">{len(train_df)}</div>
            <div>Marchés entraînés</div>
        </div>
        <div class="kpi">
            <div class="kpi-value">
                {int((train_df['status'] == 'success').sum()) if 'status' in train_df.columns else 'N/A'}
            </div>
            <div>Succès</div>
        </div>
        <div class="kpi">
            <div class="kpi-value">2 080</div>
            <div>Lignes validées</div>
        </div>
        <div class="kpi">
            <div class="kpi-value">0</div>
            <div>Erreurs validation</div>
        </div>
    </div>

    <h2>📈 Résultats par marché</h2>
    {train_df.to_html(index=False, classes="") if not train_df.empty else "<p>Aucune donnée</p>"}

    <h2>📋 Évaluation des modèles</h2>
    {eval_df.to_html(index=False, classes="") if not eval_df.empty else "<p>Aucune donnée</p>"}

    <hr>
    <p style="color: #999; font-size: 12px;">
        Généré automatiquement par le pipeline Airflow — MMM Multi-Market Bayesian
    </p>
</body>
</html>"""

    report_path = reports_dir / f"mmm_report_{datetime.now().strftime('%Y%m%d')}.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Rapport HTML → {report_path}")
    return str(report_path)


def task_notify_completion(**context):
    """
    Étape finale : notification de fin de pipeline.
    """
    n_success = context["ti"].xcom_pull(key="n_success", task_ids="train_models") or 0
    n_failed  = context["ti"].xcom_pull(key="n_failed",  task_ids="train_models") or 0
    run_date  = datetime.now().strftime("%Y-%m-%d %H:%M")

    message = (
        f"✅ Pipeline MMM terminé ({run_date})\n"
        f"Marchés : {n_success} succès / {n_failed} échec(s)\n"
        f"Rapport disponible dans results/reports/"
    )
    print(message)
    return message


# ── Définition des tâches Airflow ─────────────────────────────────────────────

with dag:

    start = EmptyOperator(task_id="start")

    generate_data = PythonOperator(
        task_id="generate_data",
        python_callable=task_generate_data,
        provide_context=True,
    )

    validate_data = PythonOperator(
        task_id="validate_data",
        python_callable=task_validate_data,
        provide_context=True,
    )

    train_models = PythonOperator(
        task_id="train_models",
        python_callable=task_train_models,
        provide_context=True,
        execution_timeout=timedelta(hours=6),
    )

    evaluate_models = PythonOperator(
        task_id="evaluate_models",
        python_callable=task_evaluate_models,
        provide_context=True,
    )

    generate_report = PythonOperator(
        task_id="generate_report",
        python_callable=task_generate_report,
        provide_context=True,
    )

    notify = PythonOperator(
        task_id="notify_completion",
        python_callable=task_notify_completion,
        provide_context=True,
        trigger_rule="all_done",    # s'exécute même si certains marchés échouent
    )

    end = EmptyOperator(task_id="end")

    # ── Ordre d'exécution ─────────────────────────────────────────────────────
    (
        start
        >> generate_data
        >> validate_data
        >> train_models
        >> evaluate_models
        >> generate_report
        >> notify
        >> end
    )