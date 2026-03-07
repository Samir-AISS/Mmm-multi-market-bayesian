"""
model_diagnostics.py
--------------------
Diagnostics de convergence MCMC et validation du modèle.

Métriques :
- R-hat (Gelman-Rubin) : doit être < 1.01 pour tous les paramètres
- ESS (Effective Sample Size) : doit être > 400
- Posterior Predictive Check (PPC)
- LOO-CV (Leave-One-Out Cross Validation)

TODO : implémenter les fonctions (Partie 2)
"""

# import arviz as az
# import numpy as np
# import pandas as pd


def check_rhat(idata, threshold: float = 1.01) -> dict:
    """
    Vérifie la convergence R-hat de tous les paramètres.
    Retourne {param: rhat_value} pour les paramètres > threshold.
    TODO
    """
    raise NotImplementedError


def check_ess(idata, min_ess: int = 400) -> dict:
    """
    Vérifie l'Effective Sample Size.
    Retourne {param: ess_value} pour les paramètres < min_ess.
    TODO
    """
    raise NotImplementedError


def posterior_predictive_check(idata, df) -> dict:
    """
    Génère des statistiques de PPC.
    Compare la distribution observée vs prédite.
    TODO
    """
    raise NotImplementedError


def compute_loo(idata) -> dict:
    """
    Leave-One-Out Cross Validation via arviz.
    Retourne elpd_loo, p_loo, et les warnings eventuels.
    TODO
    """
    raise NotImplementedError


def full_diagnostics_report(idata, df) -> dict:
    """
    Rapport complet de diagnostics.
    Retourne un dict avec toutes les métriques.
    TODO
    """
    raise NotImplementedError
