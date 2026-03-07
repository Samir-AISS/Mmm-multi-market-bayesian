"""
distributed_trainer.py
-----------------------
Entraînement parallèle du modèle sur tous les marchés.

Stratégie :
- joblib.Parallel pour entraîner N marchés en parallèle
- MLflow pour tracker chaque run (paramètres, métriques, artefacts)
- Sauvegarde des modèles dans results/models/

Usage:
    python src/training/distributed_trainer.py --markets all
    python src/training/distributed_trainer.py --markets FR,DE,UK

TODO : implémenter (Partie 3)
"""

# import argparse
# import mlflow
# from joblib import Parallel, delayed
# from src.models.bayesian_mmm import BayesianMMM
# from src.data.data_loader import load_all_markets


def train_single_market(market: str, df, config: dict) -> dict:
    """
    Entraîne le modèle pour un marché, log dans MLflow.
    Retourne les métriques de performance.
    TODO
    """
    raise NotImplementedError


def train_all_markets(markets: list, n_jobs: int = -1) -> dict:
    """
    Entraîne en parallèle tous les marchés spécifiés.
    TODO
    """
    raise NotImplementedError


if __name__ == "__main__":
    # TODO : argparse + appel train_all_markets()
    raise NotImplementedError
