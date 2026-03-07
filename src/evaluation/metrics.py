"""
metrics.py
----------
Métriques d'évaluation pour le modèle MMM.

Métriques :
- R²  (coefficient de détermination)
- MAPE (Mean Absolute Percentage Error)
- NRMSE (Normalized Root Mean Square Error)
- SMAPE (Symmetric MAPE)

TODO : implémenter (Partie 2)
"""

import numpy as np


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² — proportion de variance expliquée."""
    # TODO
    raise NotImplementedError


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE — erreur absolue moyenne en pourcentage."""
    # TODO
    raise NotImplementedError


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NRMSE — RMSE normalisé par la moyenne."""
    # TODO
    raise NotImplementedError


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """SMAPE — MAPE symétrique."""
    # TODO
    raise NotImplementedError


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcule toutes les métriques d'un coup."""
    # TODO
    raise NotImplementedError
