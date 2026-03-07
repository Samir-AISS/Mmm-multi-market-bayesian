"""
base_mmm.py
-----------
Classe de base abstraite pour les modèles MMM.
Définit l'interface commune : fit(), predict(), get_contributions().

TODO : implémenter la classe abstraite complète.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseMMM(ABC):
    """Interface commune pour tous les modèles MMM."""

    def __init__(self, config: dict = None):
        self.config       = config or {}
        self.is_fitted    = False
        self.channels     = []

    @abstractmethod
    def build_model(self, df: pd.DataFrame):
        """Construit le graphe probabiliste / structure du modèle."""
        pass

    @abstractmethod
    def fit(self, df: pd.DataFrame, **kwargs):
        """Entraîne le modèle sur les données."""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Génère des prédictions (revenu attendu)."""
        pass

    @abstractmethod
    def get_contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Décompose les ventes par composante :
        base, tv, facebook, search, ooh, print, seasonality, trend.
        """
        pass

    def get_roi(self) -> pd.DataFrame:
        """Calcule le ROI par canal (revenu généré / spend)."""
        # TODO : implémenter à partir de get_contributions()
        raise NotImplementedError
