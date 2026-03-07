"""
bayesian_mmm.py
---------------
Modèle MMM bayésien avec PyMC.

Architecture :
  - Priors sur les coefficients β (HalfNormal)
  - Priors sur les paramètres adstock (Beta)
  - Priors sur les paramètres Hill (Gamma)
  - Likelihood : Normal(μ, σ)
  - Inférence : NUTS (No-U-Turn Sampler)

TODO : implémenter le modèle complet (Partie 2)
"""

# import pymc as pm
# import pytensor.tensor as pt
# import numpy as np
# import pandas as pd
# import arviz as az
# from src.models.base_mmm import BaseMMM


class BayesianMMM:
    """
    Modèle MMM bayésien hiérarchique avec PyMC.

    Paramètres
    ----------
    config : dict — depuis model_config.yaml
    market : str  — code marché (ex: "FR")
    """

    def __init__(self, config: dict = None, market: str = "ALL"):
        self.config  = config or {}
        self.market  = market
        self.model   = None   # pm.Model
        self.idata   = None   # az.InferenceData
        self.channels = ["tv","facebook","search","ooh","print"]

    def build_model(self, df):
        """
        Construit le modèle PyMC.

        Structure :
            revenue ~ Normal(mu, sigma)
            mu = base + Σ beta_i * adstock_i * hill_i + seasonality + trend
        """
        # TODO — Partie 2
        raise NotImplementedError

    def fit(self, df, draws=1000, tune=1000, chains=4, target_accept=0.9):
        """Lance l'inférence MCMC (NUTS)."""
        # TODO — Partie 2
        raise NotImplementedError

    def predict(self, df):
        """Posterior predictive : génère des prédictions avec incertitude."""
        # TODO — Partie 2
        raise NotImplementedError

    def get_contributions(self, df):
        """Décompose les revenus par canal marketing."""
        # TODO — Partie 2
        raise NotImplementedError

    def get_roi(self):
        """ROI par canal avec intervalles de crédibilité HDI 94%."""
        # TODO — Partie 2
        raise NotImplementedError

    def save(self, path):
        """Sauvegarde idata (NetCDF) + metadata."""
        # TODO — Partie 2
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        """Charge un modèle sauvegardé."""
        # TODO — Partie 2
        raise NotImplementedError
