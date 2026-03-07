"""
feature_engineering.py
-----------------------
Transformations des features avant modélisation.

TODO :
- apply_adstock() : wrapper sur les dépenses
- apply_saturation() : wrapper Hill
- add_fourier_seasonality() : composantes Fourier pour la saisonnalité
- normalize_spends() : normalisation pour le modèle bayésien
- create_lagged_features() : variables décalées
"""

import numpy as np
import pandas as pd
from typing import Optional


def apply_adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """Adstock géométrique."""
    # TODO : implémenter (voir multi_market_generator.py)
    raise NotImplementedError


def apply_hill_saturation(x: np.ndarray, k: Optional[float] = None, s: float = 2.0) -> np.ndarray:
    """Transformation Hill."""
    # TODO : implémenter
    raise NotImplementedError


def add_fourier_seasonality(df: pd.DataFrame, n_components: int = 2, period: int = 52) -> pd.DataFrame:
    """
    Ajoute des composantes de Fourier pour modéliser la saisonnalité.
    Retourne df avec colonnes sin_1, cos_1, sin_2, cos_2...
    """
    # TODO : implémenter
    raise NotImplementedError


def normalize_spends(df: pd.DataFrame, spend_cols: list) -> pd.DataFrame:
    """Normalise les dépenses (division par la médiane) pour stabilité MCMC."""
    # TODO : implémenter
    raise NotImplementedError
