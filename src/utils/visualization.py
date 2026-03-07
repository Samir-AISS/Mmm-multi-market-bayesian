"""
visualization.py
----------------
Fonctions de visualisation pour le projet MMM.

Graphiques disponibles :
- plot_channel_contributions()  : waterfall par canal
- plot_saturation_curves()      : courbes Hill par canal
- plot_actual_vs_predicted()    : série temporelle réel vs prédit
- plot_roi_comparison()         : ROI comparé entre marchés
- plot_posterior_distributions(): distributions postérieures
- plot_budget_optimizer()       : courbes ROI marginal

TODO : implémenter (Partie 3)
"""

# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import pandas as pd
# import arviz as az
# import numpy as np


def plot_channel_contributions(contributions_df, market: str = None, output_path=None):
    """Waterfall chart des contributions par canal. TODO"""
    raise NotImplementedError


def plot_saturation_curves(idata, channels: list, output_path=None):
    """Courbes de saturation Hill par canal. TODO"""
    raise NotImplementedError


def plot_actual_vs_predicted(df, idata, market: str, output_path=None):
    """Série temporelle : réel vs prédit (avec intervalle de crédibilité). TODO"""
    raise NotImplementedError


def plot_roi_comparison(roi_df, output_path=None):
    """Comparaison ROI entre marchés et canaux. TODO"""
    raise NotImplementedError


def plot_posterior_distributions(idata, params: list = None, output_path=None):
    """Distributions postérieures des paramètres. TODO"""
    raise NotImplementedError


def plot_budget_optimizer(marginal_roi, total_budget: float, output_path=None):
    """Courbes de ROI marginal pour l'optimisation budgétaire. TODO"""
    raise NotImplementedError
