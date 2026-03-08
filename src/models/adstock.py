"""
adstock.py
----------
Transformations adstock pour modéliser la mémoire publicitaire.

Deux modèles disponibles :
  - GeometricAdstock : décroissance exponentielle simple (le plus courant)
  - DelayedAdstock   : pic différé avant décroissance (TV, OOH)

Référence : Jin et al. (2017) "Bayesian Methods for Media Mix Modeling
            with Carryover and Shape Effects"
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union


class BaseAdstock(ABC):
    """Classe de base pour les transformations adstock."""

    @abstractmethod
    def transform(self, spend: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, spend: np.ndarray) -> np.ndarray:
        return self.transform(spend)

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({params})"


class GeometricAdstock(BaseAdstock):
    """
    Adstock géométrique classique.

    Formule : adstock[t] = spend[t] + decay × adstock[t-1]

    Propriétés :
      - decay = 0 → pas de carryover (adstock = spend)
      - decay = 1 → somme cumulative
      - La somme des poids = 1 / (1 - decay)

    Paramètres
    ----------
    decay : float ∈ [0, 1]
        Taux de rétention hebdomadaire.
        Ex: 0.5 → la moitié de l'effet est retenu la semaine suivante.
    normalize : bool
        Si True, normalise l'output pour avoir la même échelle que l'input.
    """

    def __init__(self, decay: float = 0.5, normalize: bool = False):
        if not 0 <= decay <= 1:
            raise ValueError(f"decay doit être dans [0, 1], reçu : {decay}")
        self.decay = decay
        self.normalize = normalize

    def transform(self, spend: np.ndarray) -> np.ndarray:
        """Applique la transformation adstock géométrique."""
        spend = np.asarray(spend, dtype=float)
        n = len(spend)
        result = np.zeros(n, dtype=float)
        result[0] = spend[0]
        for t in range(1, n):
            result[t] = spend[t] + self.decay * result[t - 1]

        if self.normalize and result.max() > 0:
            # Ramène à la même échelle que spend (utile pour interprétation)
            result = result * (spend.sum() / result.sum())

        return result

    def half_life(self) -> float:
        """Nombre de semaines pour que l'effet tombe à 50%."""
        if self.decay <= 0:
            return 0.0
        return np.log(0.5) / np.log(self.decay)

    def weights(self, n_periods: int = 20) -> np.ndarray:
        """Retourne les poids des n_periods semaines précédentes."""
        return np.array([self.decay ** t for t in range(n_periods)])


class DelayedAdstock(BaseAdstock):
    """
    Adstock avec pic différé (delayed peak).

    Utile pour la TV et l'OOH où l'effet maximum arrive
    quelques semaines après la diffusion.

    Approche : pondération par une distribution Gaussienne centrée sur `peak`.

    Paramètres
    ----------
    decay : float ∈ [0, 1]
        Taux de décroissance après le pic.
    peak : int ≥ 0
        Semaine du pic d'effet (0 = immédiat, comme GeometricAdstock).
    """

    def __init__(self, decay: float = 0.5, peak: int = 1):
        if not 0 <= decay <= 1:
            raise ValueError(f"decay doit être dans [0, 1], reçu : {decay}")
        if peak < 0:
            raise ValueError(f"peak doit être ≥ 0, reçu : {peak}")
        self.decay = decay
        self.peak  = peak

    def transform(self, spend: np.ndarray) -> np.ndarray:
        """Applique la transformation adstock avec pic différé."""
        spend  = np.asarray(spend, dtype=float)
        n      = len(spend)
        result = np.zeros(n, dtype=float)

        for t in range(n):
            for lag in range(min(t + 1, n)):
                # Poids = decay^|lag - peak| (max au pic, décroît autour)
                weight    = self.decay ** abs(lag - self.peak)
                result[t] += weight * spend[t - lag]

        # Normalisation : même somme que spend (évite explosion)
        total_spend = spend.sum()
        if result.sum() > 0 and total_spend > 0:
            result = result * (total_spend / result.sum())

        return result

    def half_life(self) -> float:
        """Demi-vie approximative après le pic."""
        if self.decay <= 0:
            return 0.0
        return np.log(0.5) / np.log(self.decay)


def apply_adstock(spend: np.ndarray,
                  decay: float = 0.5,
                  adstock_type: str = "geometric",
                  peak: int = 0) -> np.ndarray:
    """
    Fonction utilitaire pour appliquer l'adstock.

    Paramètres
    ----------
    spend        : array de dépenses hebdomadaires
    decay        : taux de décroissance ∈ [0, 1]
    adstock_type : "geometric" | "delayed"
    peak         : semaine du pic (DelayedAdstock uniquement)
    """
    if adstock_type == "geometric":
        return GeometricAdstock(decay=decay).transform(spend)
    elif adstock_type == "delayed":
        return DelayedAdstock(decay=decay, peak=peak).transform(spend)
    else:
        raise ValueError(f"adstock_type inconnu : {adstock_type!r}. Choisir 'geometric' ou 'delayed'.")