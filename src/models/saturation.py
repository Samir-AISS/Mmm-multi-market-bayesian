"""
saturation.py
-------------
Fonctions de saturation pour modéliser les rendements décroissants.

Modèles disponibles :
- HillSaturation    : courbe Hill (la plus utilisée en MMM)
- LogisticSaturation: courbe logistique symétrique

Référence : Rao & Bharani (2022) "Saturation Effects in MMM"

TODO : implémenter les classes complètes
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseSaturation(ABC):
    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        pass


class HillSaturation(BaseSaturation):
    """
    Transformation Hill : hill(x) = x^s / (K^s + x^s)

    Paramètres
    ----------
    K : float — demi-saturation (spend auquel l'effet = 50% du max)
    s : float — forme de la courbe (s > 1 = sigmoïde, s < 1 = concave)
    """
    def __init__(self, K: float = None, s: float = 2.0):
        self.K = K
        self.s = s

    def transform(self, x: np.ndarray) -> np.ndarray:
        # TODO : implémenter
        raise NotImplementedError


class LogisticSaturation(BaseSaturation):
    """
    Saturation logistique : logistic(x) = L / (1 + exp(-k(x - x0)))

    Paramètres
    ----------
    L  : float — valeur maximale
    k  : float — pente
    x0 : float — point d'inflexion
    """
    def __init__(self, L: float = 1.0, k: float = 1.0, x0: float = 0.5):
        self.L  = L
        self.k  = k
        self.x0 = x0

    def transform(self, x: np.ndarray) -> np.ndarray:
        # TODO : implémenter
        raise NotImplementedError
