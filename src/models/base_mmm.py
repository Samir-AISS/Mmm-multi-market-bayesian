"""
base_mmm.py
-----------
Classe de base abstraite pour tous les modèles MMM.
Définit l'interface commune : build_model(), fit(), predict(),
get_contributions(), get_roi().
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class BaseMMM(ABC):
    """
    Interface commune pour tous les modèles MMM.
    Toute implémentation concrète (BayesianMMM, etc.)
    doit hériter de cette classe et implémenter les méthodes abstraites.
    """

    CHANNELS: List[str] = ["tv", "facebook", "search", "ooh", "print"]

    def __init__(self, config: dict = None, market: str = "ALL"):
        self.config    = config or {}
        self.market    = market
        self.is_fitted = False
        self._contributions: Optional[pd.DataFrame] = None

    # ── Méthodes abstraites (obligatoires) ────────────────────────────────────

    @abstractmethod
    def build_model(self, df: pd.DataFrame):
        """Construit la structure du modèle à partir des données."""
        pass

    @abstractmethod
    def fit(self, df: pd.DataFrame, **kwargs):
        """Entraîne le modèle sur les données."""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Retourne les revenus prédits (valeur centrale)."""
        pass

    @abstractmethod
    def get_contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Décompose les revenus par composante.

        Retourne un DataFrame avec les colonnes :
          week, base, tv, facebook, search, ooh, print,
          seasonality, trend, events, total_predicted
        """
        pass

    # ── Méthode concrète partagée ─────────────────────────────────────────────

    def get_roi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule le ROI par canal marketing.
        ROI_canal = Σ(contribution_canal) / Σ(spend_canal)

        Retourne un DataFrame avec les colonnes :
          channel | total_contribution | total_spend | roi
        """
        self._check_fitted()

        contrib = self.get_contributions(df)
        rows = []

        for channel in self.CHANNELS:
            spend_col   = f"{channel}_spend"
            contrib_col = channel

            if contrib_col not in contrib.columns:
                continue
            if spend_col not in df.columns:
                continue

            total_contribution = contrib[contrib_col].sum()
            total_spend        = df[spend_col].sum()
            roi = total_contribution / total_spend if total_spend > 0 else np.nan

            rows.append({
                "channel":            channel,
                "total_contribution": round(total_contribution, 2),
                "total_spend":        round(total_spend, 2),
                "roi":                round(roi, 4),
            })

        return pd.DataFrame(rows).sort_values("roi", ascending=False).reset_index(drop=True)

    def summary(self, df: pd.DataFrame) -> dict:
        """
        Résumé rapide : ROI par canal + part de contribution de chaque canal.
        """
        self._check_fitted()
        roi_df = self.get_roi(df)
        total_marketing = roi_df["total_contribution"].sum()

        roi_df["contribution_share_pct"] = (
            roi_df["total_contribution"] / total_marketing * 100
        ).round(1)

        return {
            "market":           self.market,
            "n_obs":            len(df),
            "roi_by_channel":   roi_df.to_dict(orient="records"),
            "best_channel":     roi_df.iloc[0]["channel"] if len(roi_df) > 0 else None,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} n'est pas encore entraîné. "
                "Appelez .fit(df) d'abord."
            )

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(market={self.market!r}, status={status})"