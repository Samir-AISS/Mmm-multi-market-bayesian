"""
roi_calculator.py
-----------------
Calcul du ROI par canal marketing.

ROI = (Revenue généré par le canal) / (Spend du canal)
      avec intervalles de crédibilité HDI 94%

TODO : implémenter (Partie 2/3)
"""

import pandas as pd
import numpy as np


def compute_channel_roi(contributions: pd.DataFrame, spends: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le ROI moyen par canal.

    Paramètres
    ----------
    contributions : df avec colonnes [channel, revenue_contribution]
    spends        : df avec colonnes [channel, total_spend]

    Retourne
    --------
    df avec colonnes [channel, roi_mean, roi_hdi_low, roi_hdi_high]
    """
    # TODO
    raise NotImplementedError


def compute_marginal_roi(idata, spend_range: np.ndarray, channel: str) -> np.ndarray:
    """
    Calcule le ROI marginal pour différents niveaux de spend.
    Utile pour construire les courbes de réponse.
    TODO
    """
    raise NotImplementedError


def budget_recommendation(roi_df: pd.DataFrame, total_budget: float) -> pd.DataFrame:
    """
    Recommandation d'allocation optimale basée sur les ROI.
    Maximise le revenu sous contrainte de budget total.
    TODO
    """
    raise NotImplementedError
