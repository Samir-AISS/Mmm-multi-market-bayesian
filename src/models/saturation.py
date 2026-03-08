"""
saturation.py
-------------
Fonctions de saturation pour modéliser les rendements décroissants.

Modèles disponibles :
- HillSaturation    : courbe Hill (la plus utilisée en MMM)
- LogisticSaturation: courbe logistique symétrique

Référence : Rao & Bharani (2022) "Saturation Effects in MMM"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseSaturation(ABC):
    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        ...


class HillSaturation(BaseSaturation):
    """
    Transformation Hill : hill(x) = x^S / (K^S + x^S)

    Paramètres
    ----------
    K : float | None
        Demi-saturation (spend auquel l'effet = 50% du max).
        Si ``None``, on utilise la médiane de x (pour x > 0).
    S : float
        Forme de la courbe (S > 1 = sigmoïde, S < 1 = concave).
    """

    def __init__(self, K: float | None = None, S: float = 2.0):
        if K is not None and K <= 0:
            raise ValueError("K must be positive when provided.")
        if S <= 0:
            raise ValueError("S must be strictly positive.")

        self.K = K
        self.S = float(S)

    def _ensure_array(self, x: np.ndarray | float) -> np.ndarray:
        return np.asarray(x, dtype=float)

    def _get_K(self, x: np.ndarray) -> float:
        if self.K is not None:
            return float(self.K)
        # K auto : médiane des valeurs positives (sinon médiane globale)
        positive = x[x > 0]
        if positive.size > 0:
            return float(np.median(positive))
        return float(np.median(x))

    def transform(self, x: np.ndarray | float) -> np.ndarray:
        x_arr = self._ensure_array(x)
        K = self._get_K(x_arr)

        # hill(x) = x^S / (K^S + x^S), borné entre 0 et 1
        x_pow = np.power(np.clip(x_arr, a_min=0.0, a_max=None), self.S)
        K_pow = K**self.S
        denom = K_pow + x_pow
        # éviter les divisions 0/0 si tout est nul
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(denom > 0, x_pow / denom, 0.0)

        # assurer les bornes numériques
        y = np.clip(y, 0.0, 1.0)
        return y

    def marginal_return(self, x: np.ndarray | float) -> np.ndarray:
        """
        Retour marginal (dérivée de hill par rapport à x).

        Utilisé uniquement dans les tests pour vérifier
        la décroissance des rendements marginaux.
        """
        x_arr = self._ensure_array(x)
        K = self._get_K(x_arr)
        x_pos = np.clip(x_arr, a_min=0.0, a_max=None)

        S = self.S
        K_pow = K**S
        x_pow = np.power(x_pos, S)
        denom = (K_pow + x_pow) ** 2

        with np.errstate(divide="ignore", invalid="ignore"):
            mr = np.where(
                denom > 0,
                S * np.power(x_pos, S - 1.0) * K_pow / denom,
                0.0,
            )
        return mr

    def response_curve(self, x_max: float, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Génère une courbe réponse (x, hill(x)) de 0 à x_max.
        """
        x_vals = np.linspace(0.0, float(x_max), int(n_points))
        y_vals = self.transform(x_vals)
        return x_vals, y_vals


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
        self.L = float(L)
        self.k = float(k)
        self.x0 = float(x0)

    def transform(self, x: np.ndarray | float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        # logistic(x) = L / (1 + exp(-k(x - x0)))
        z = -self.k * (x_arr - self.x0)
        # clamp z pour stabilité numérique
        z = np.clip(z, -60.0, 60.0)
        y = self.L / (1.0 + np.exp(z))
        # bornage numérique léger (<= L)
        y = np.clip(y, 0.0, self.L)
        return y


def apply_saturation(
    x: np.ndarray | float,
    saturation_type: str = "hill",
    **kwargs,
) -> np.ndarray:
    """
    Utilitaire pour appliquer une saturation à un vecteur de spend.

    Paramètres
    ----------
    x : array-like
        Valeurs de dépense.
    saturation_type : {'hill', 'logistic'}
        Type de saturation à utiliser.
    **kwargs :
        Paramètres passés au constructeur de la classe correspondante.
    """
    sat_type = saturation_type.lower()

    if sat_type == "hill":
        saturator = HillSaturation(**kwargs)
    elif sat_type == "logistic":
        saturator = LogisticSaturation(**kwargs)
    else:
        raise ValueError(f"Unknown saturation_type '{saturation_type}'")

    return saturator.transform(x)
