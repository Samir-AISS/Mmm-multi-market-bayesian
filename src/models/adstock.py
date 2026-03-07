"""
adstock.py
----------
Transformations adstock pour modéliser la mémoire publicitaire.

Modèles disponibles :
- GeometricAdstock  : décroissance exponentielle simple
- DelayedAdstock    : pic différé avant décroissance (ex: campagnes TV)

Référence : Jin et al. (2017) "Bayesian Methods for Media Mix Modeling
            with Carryover and Shape Effects"

TODO : implémenter les classes complètes
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseAdstock(ABC):
    """Classe de base pour les transformations adstock."""

    @abstractmethod
    def transform(self, spend: np.ndarray) -> np.ndarray:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"


class GeometricAdstock(BaseAdstock):
    """
    Adstock géométrique classique.
    adstock[t] = spend[t] + decay × adstock[t-1]

    Paramètres
    ----------
    decay : float ∈ [0, 1]
        Taux de rétention hebdomadaire. Ex: 0.5 = 50% de l'effet retenu.
    """
    def __init__(self, decay: float = 0.5):
        assert 0 <= decay <= 1, "decay doit être dans [0, 1]"
        self.decay = decay

    def transform(self, spend: np.ndarray) -> np.ndarray:
        # TODO : implémenter
        raise NotImplementedError


class DelayedAdstock(BaseAdstock):
    """
    Adstock avec pic différé (peak).
    Utile pour la TV où l'effet maximum est décalé de quelques semaines.

    Paramètres
    ----------
    decay : float ∈ [0, 1]
    peak  : int — semaines avant le pic d'effet
    """
    def __init__(self, decay: float = 0.5, peak: int = 1):
        self.decay = decay
        self.peak  = peak

    def transform(self, spend: np.ndarray) -> np.ndarray:
        # TODO : implémenter
        raise NotImplementedError
