"""
bayesian_mmm.py
---------------
Modèle MMM bayésien avec PyMC.

Architecture du modèle
-----------------------
  revenue(t) ~ Normal(μ(t), σ)

  μ(t) = base_sales
       + Σ_i [ β_i × Hill(Adstock(spend_i(t), θ_i), K_i, S_i) ]
       + γ_season × seasonality(t)
       + γ_trend  × trend(t)
       + γ_events × events(t)

Priors
------
  β_i       ~ HalfNormal(σ=1)       # coefficients canaux (positifs)
  θ_i       ~ Beta(α=2, β=2)        # decay adstock ∈ [0,1]
  K_i       ~ Gamma(α=3, β=1)       # half-saturation (normalisé)
  S_i       ~ Gamma(α=2, β=1)       # shape Hill > 0
  γ_*       ~ Normal(0, 0.5)        # effets contrôles
  σ         ~ HalfNormal(0.5)       # bruit observation

Sampling
--------
  NUTS (No-U-Turn Sampler) via PyMC
  4 chains × 1000 draws (+ 1000 tune)
  target_accept = 0.9

Usage
-----
  model = BayesianMMM(market="FR")
  model.fit(df_fr)
  roi      = model.get_roi()
  contribs = model.get_contributions(df_fr)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from src.models.adstock import GeometricAdstock
from src.models.saturation import HillSaturation
from src.evaluation.metrics import compute_all_metrics, print_metrics_report

# Import PyMC optionnel
try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False


# ── Constantes ────────────────────────────────────────────────────────────────
CHANNELS = ["tv", "facebook", "search", "ooh", "print"]

CHANNEL_CONFIG = {
    "tv":       {"decay": 0.60, "K_scale": 1.0, "S": 2.0, "adstock_type": "geometric"},
    "facebook": {"decay": 0.30, "K_scale": 1.0, "S": 2.0, "adstock_type": "geometric"},
    "search":   {"decay": 0.15, "K_scale": 1.0, "S": 2.0, "adstock_type": "geometric"},
    "ooh":      {"decay": 0.45, "K_scale": 1.0, "S": 2.0, "adstock_type": "geometric"},
    "print":    {"decay": 0.25, "K_scale": 1.0, "S": 2.0, "adstock_type": "geometric"},
}


class BayesianMMM:
    """
    Modèle MMM bayésien avec PyMC.

    Paramètres
    ----------
    market : str
        Code du marché (ex: "FR", "DE"). Utilisé pour les logs.
    config : dict
        Hyperparamètres (depuis model_config.yaml). Utilise les défauts si None.
    channels : list
        Liste des canaux à modéliser. Par défaut : les 5 canaux.
    """

    def __init__(self,
                 market: str = "ALL",
                 config: Optional[dict] = None,
                 channels: Optional[List[str]] = None):
        self.market   = market
        self.config   = config or {}
        self.channels = channels or CHANNELS

        self.model          = None
        self.idata          = None
        self.trace_         = None
        self.is_fitted      = False
        self._scaler        = {}
        self._metrics       = {}
        self._contributions = None

    # ── Préparation des données ───────────────────────────────────────────────

    def _prepare_data(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Prépare les arrays numpy depuis le DataFrame.
        Applique adstock + normalisation pour stabilité MCMC.
        """
        data = {}
        data["revenue"]     = df["revenue"].values.astype(float)
        data["seasonality"] = df["seasonality"].values.astype(float)
        data["trend"]       = df["trend"].values.astype(float)
        data["events"]      = df["events"].values.astype(float)

        for ch in self.channels:
            col   = f"{ch}_spend"
            spend = df[col].values.astype(float)
            cfg   = CHANNEL_CONFIG[ch]

            # 1. Adstock
            adstocked = GeometricAdstock(decay=cfg["decay"]).transform(spend)

            # 2. Normalisation
            median = np.median(adstocked[adstocked > 0]) if np.any(adstocked > 0) else 1.0
            self._scaler[ch] = median
            adstocked_norm   = adstocked / median

            # 3. Hill saturation
            data[f"{ch}_transformed"] = HillSaturation(K=1.0, S=cfg["S"]).transform(adstocked_norm)
            data[f"{ch}_spend_raw"]   = spend
            data[f"{ch}_adstocked"]   = adstocked

        # Normalisation revenue
        rev_median = np.median(data["revenue"])
        self._scaler["revenue"] = rev_median
        data["revenue_norm"]    = data["revenue"] / rev_median

        return data

    # ── Construction du modèle PyMC ───────────────────────────────────────────

    def build_model(self, data: Dict[str, np.ndarray]):
        """
        Construit le graphe probabiliste PyMC.
        """
        if not PYMC_AVAILABLE:
            raise ImportError(
                "PyMC n'est pas installé. Lancez : pip install pymc\n"
                "Vous pouvez utiliser .fit_numpy() pour une estimation OLS rapide."
            )

        n = len(data["revenue_norm"])

        with pm.Model() as self.model:

            # ── Priors canaux ─────────────────────────────────────────────────
            betas = {}
            for ch in self.channels:
                betas[ch] = pm.HalfNormal(f"beta_{ch}", sigma=1.0)

            # ── Priors contrôles ──────────────────────────────────────────────
            gamma_season = pm.Normal("gamma_seasonality", mu=0.5, sigma=0.3)
            gamma_trend  = pm.Normal("gamma_trend",       mu=0.3, sigma=0.3)
            gamma_events = pm.Normal("gamma_events",      mu=0.1, sigma=0.2)

            # ── Base (intercept) ──────────────────────────────────────────────
            base = pm.HalfNormal("base", sigma=1.0)

            # ── Contribution de chaque canal ──────────────────────────────────
            mu = base
            for ch in self.channels:
                mu = mu + betas[ch] * data[f"{ch}_transformed"]

            # ── Effets de contrôle ────────────────────────────────────────────
            mu = mu + gamma_season * data["seasonality"]
            mu = mu + gamma_trend  * data["trend"]
            mu = mu + gamma_events * data["events"]

            # ── Bruit d'observation ───────────────────────────────────────────
            sigma = pm.HalfNormal("sigma", sigma=0.5)

            # ── Likelihood ────────────────────────────────────────────────────
            _ = pm.Normal(
                "revenue_obs",
                mu=mu,
                sigma=sigma,
                observed=data["revenue_norm"]
            )

        return self.model

    # ── Inférence MCMC ────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame,
            draws: int = 1000,
            tune: int = 1000,
            chains: int = 4,
            target_accept: float = 0.9,
            random_seed: int = 42) -> "BayesianMMM":
        """
        Entraîne le modèle par inférence MCMC (NUTS).
        Si PyMC non disponible, bascule automatiquement en mode OLS.
        """
        print(f"🔧 Préparation des données — Marché : {self.market}")
        data = self._prepare_data(df)

        if PYMC_AVAILABLE:
            print(f"⛓️  Sampling MCMC — {chains} chains × {draws} draws + {tune} tune")
            self.build_model(data)

            with self.model:
                self.idata = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    random_seed=random_seed,
                    return_inferencedata=True,
                    progressbar=True,
                )
                # Posterior predictive
                pm.sample_posterior_predictive(
                    self.idata,
                    extend_inferencedata=True,
                    random_seed=random_seed,
                )

            print(f"✅ Sampling terminé")

        else:
            print("⚠️  PyMC non disponible — mode OLS (fallback)")
            self._fit_numpy(data)

        self.is_fitted = True

        # Métriques sur les données d'entraînement
        y_pred = self.predict(df)
        y_true = df["revenue"].values
        self._metrics = compute_all_metrics(y_true, y_pred)
        print_metrics_report(self._metrics, label=f"Train [{self.market}]")

        return self

    # ── Fallback OLS ──────────────────────────────────────────────────────────

    def _fit_numpy(self, data: Dict[str, np.ndarray]) -> None:
        """
        Estimation OLS rapide (fallback sans PyMC).
        Utilisé pour tests et démos sans dépendances lourdes.
        """
        from numpy.linalg import lstsq

        n = len(data["revenue_norm"])

        # Matrice de features : [intercept, canaux..., seasonality, trend, events]
        X_cols = [np.ones(n)]
        for ch in self.channels:
            X_cols.append(data[f"{ch}_transformed"])
        X_cols += [data["seasonality"], data["trend"], data["events"]]

        X = np.column_stack(X_cols)
        y = data["revenue_norm"]

        coeffs, _, _, _ = lstsq(X, y, rcond=None)

        # Stocker les coefficients dans trace_ (même interface que MCMC)
        self.trace_ = {"base": max(coeffs[0], 0)}
        for i, ch in enumerate(self.channels):
            self.trace_[f"beta_{ch}"] = max(coeffs[i + 1], 0)
        self.trace_["gamma_seasonality"] = coeffs[-3]
        self.trace_["gamma_trend"]       = coeffs[-2]
        self.trace_["gamma_events"]      = coeffs[-1]

    # ── Prédictions ───────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Génère des prédictions (revenus estimés en €).

        Si PyMC : utilise la moyenne posterior.
        Si fallback : utilise les coefficients OLS.
        """
        if not self.is_fitted:
            raise RuntimeError("Modèle non entraîné. Lancez .fit() d'abord.")

        data      = self._prepare_data(df)
        rev_scale = self._scaler["revenue"]

        if PYMC_AVAILABLE and self.idata is not None:
            post    = self.idata.posterior
            base    = float(post["base"].values.mean())
            mu_norm = base

            for ch in self.channels:
                beta     = float(post[f"beta_{ch}"].values.mean())
                mu_norm += beta * data[f"{ch}_transformed"]

            g_s = float(post["gamma_seasonality"].values.mean())
            g_t = float(post["gamma_trend"].values.mean())
            g_e = float(post["gamma_events"].values.mean())

        else:
            base    = self.trace_["base"]
            mu_norm = np.full(len(data["revenue"]), base)

            for ch in self.channels:
                beta     = self.trace_[f"beta_{ch}"]
                mu_norm  = mu_norm + beta * data[f"{ch}_transformed"]

            g_s = self.trace_["gamma_seasonality"]
            g_t = self.trace_["gamma_trend"]
            g_e = self.trace_["gamma_events"]

        mu_norm = (mu_norm
                   + g_s * data["seasonality"]
                   + g_t * data["trend"]
                   + g_e * data["events"])

        return mu_norm * rev_scale

    # ── Contributions ─────────────────────────────────────────────────────────

    def get_contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Décompose les revenus prédits par composante.

        Retourne un DataFrame avec colonnes :
          date, market, week, revenue_actual, revenue_predicted,
          base, tv, facebook, search, ooh, print,
          seasonality_contrib, trend_contrib, events_contrib
        """
        if not self.is_fitted:
            raise RuntimeError("Modèle non entraîné.")

        data      = self._prepare_data(df)
        rev_scale = self._scaler["revenue"]
        n         = len(df)

        if PYMC_AVAILABLE and self.idata is not None:
            post  = self.idata.posterior
            base  = float(post["base"].values.mean()) * rev_scale
            betas = {ch: float(post[f"beta_{ch}"].values.mean()) for ch in self.channels}
            g_s   = float(post["gamma_seasonality"].values.mean())
            g_t   = float(post["gamma_trend"].values.mean())
            g_e   = float(post["gamma_events"].values.mean())
        else:
            base  = self.trace_["base"] * rev_scale
            betas = {ch: self.trace_[f"beta_{ch}"] for ch in self.channels}
            g_s   = self.trace_["gamma_seasonality"]
            g_t   = self.trace_["gamma_trend"]
            g_e   = self.trace_["gamma_events"]

        result = pd.DataFrame({
            "date":              df["date"].values if "date" in df.columns else range(n),
            "market":            self.market,
            "week":              df["week"].values,
            "revenue_actual":    data["revenue"],
            "revenue_predicted": self.predict(df),
            "base":              np.full(n, base),
        })

        for ch in self.channels:
            result[ch] = betas[ch] * data[f"{ch}_transformed"] * rev_scale

        result["seasonality_contrib"] = g_s * data["seasonality"] * rev_scale
        result["trend_contrib"]       = g_t * data["trend"]       * rev_scale
        result["events_contrib"]      = g_e * data["events"]       * rev_scale

        self._contributions = result
        return result

    # ── ROI ───────────────────────────────────────────────────────────────────

    def get_roi(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calcule le ROI par canal marketing.
        ROI = Σ(contribution du canal) / Σ(spend du canal)
        """
        if not self.is_fitted:
            raise RuntimeError("Modèle non entraîné.")

        if df is None and self._contributions is None:
            raise RuntimeError("Passez df en argument ou appelez get_contributions() d'abord.")

        if self._contributions is None:
            self.get_contributions(df)

        contribs = self._contributions
        rows     = []

        for ch in self.channels:
            total_spend  = df[f"{ch}_spend"].sum() if df is not None else 0.0
            total_contri = contribs[ch].sum()
            roi          = total_contri / total_spend if total_spend > 0 else 0.0

            rows.append({
                "channel":              ch,
                "total_spend_€":        round(total_spend, 2),
                "total_contribution_€": round(total_contri, 2),
                "roi":                  round(roi, 4),
                "roi_per_1k€":          round(roi * 1000, 2),
            })

        return pd.DataFrame(rows).sort_values("roi", ascending=False).reset_index(drop=True)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        """
        Retourne les diagnostics de convergence MCMC (R-hat, ESS).
        """
        if not PYMC_AVAILABLE or self.idata is None:
            return {"message": "PyMC non disponible — diagnostics MCMC non calculables"}

        summary   = az.summary(self.idata, round_to=4)
        r_hat_max = summary["r_hat"].max()
        ess_min   = summary["ess_bulk"].min()

        return {
            "r_hat_max": float(r_hat_max),
            "ess_min":   float(ess_min),
            "converged": r_hat_max < 1.01 and ess_min > 400,
            "n_params":  len(summary),
            "summary":   summary,
        }

    def get_metrics(self) -> dict:
        """Retourne les métriques d'entraînement."""
        if not self._metrics:
            raise RuntimeError("Modèle non entraîné.")
        return self._metrics

    # ── Sauvegarde / Chargement ───────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Sauvegarde le modèle (idata NetCDF + trace numpy)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if PYMC_AVAILABLE and self.idata is not None:
            self.idata.to_netcdf(str(path / f"idata_{self.market}.nc"))

        if self.trace_:
            np.save(str(path / f"trace_{self.market}.npy"), self.trace_)

        print(f"💾 Modèle sauvegardé → {path}")

    @classmethod
    def load(cls, path: str, market: str) -> "BayesianMMM":
        """Charge un modèle sauvegardé."""
        path  = Path(path)
        model = cls(market=market)

        nc_path  = path / f"idata_{market}.nc"
        npy_path = path / f"trace_{market}.npy"

        if PYMC_AVAILABLE and nc_path.exists():
            model.idata = az.from_netcdf(str(nc_path))

        if npy_path.exists():
            model.trace_ = np.load(str(npy_path), allow_pickle=True).item()

        model.is_fitted = True
        print(f"📂 Modèle chargé ← {path}")
        return model

    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"BayesianMMM(market={self.market!r}, "
            f"channels={self.channels}, status={status})"
        )