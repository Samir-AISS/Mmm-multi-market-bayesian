"""
feature_engineering.py
-----------------------
Transformations des features avant modélisation bayésienne.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from src.models.adstock import GeometricAdstock, DelayedAdstock
from src.models.saturation import HillSaturation, LogisticSaturation


# ── Constantes ────────────────────────────────────────────────────────────────

SPEND_COLS = ["tv_spend", "facebook_spend", "search_spend", "ooh_spend", "print_spend"]

DEFAULT_ADSTOCK_CONFIG = {
    "tv":       {"decay": 0.60, "type": "geometric"},
    "facebook": {"decay": 0.30, "type": "geometric"},
    "search":   {"decay": 0.15, "type": "geometric"},
    "ooh":      {"decay": 0.45, "type": "geometric"},
    "print":    {"decay": 0.25, "type": "geometric"},
}

DEFAULT_SATURATION_CONFIG = {
    "tv":       {"type": "hill", "S": 2.0},
    "facebook": {"type": "hill", "S": 2.0},
    "search":   {"type": "hill", "S": 1.5},
    "ooh":      {"type": "hill", "S": 2.0},
    "print":    {"type": "hill", "S": 1.5},
}


# ── Adstock ───────────────────────────────────────────────────────────────────

def apply_adstock_all_channels(
    df: pd.DataFrame,
    config: Optional[Dict] = None,
    suffix: str = "_adstocked",
) -> pd.DataFrame:
    """
    Applique la transformation adstock à tous les canaux.

    Paramètres
    ----------
    df     : DataFrame avec les colonnes *_spend
    config : {channel: {decay, type}} — défaut : DEFAULT_ADSTOCK_CONFIG
    suffix : suffixe des nouvelles colonnes (défaut : "_adstocked")

    Retourne
    --------
    DataFrame enrichi avec les colonnes *_adstocked
    """
    config = config or DEFAULT_ADSTOCK_CONFIG
    df     = df.copy()

    for channel, cfg in config.items():
        spend_col = f"{channel}_spend"
        if spend_col not in df.columns:
            continue

        spend    = df[spend_col].values.astype(float)
        decay    = cfg.get("decay", 0.5)
        ads_type = cfg.get("type", "geometric")

        if ads_type == "geometric":
            adstocked = GeometricAdstock(decay=decay).transform(spend)
        elif ads_type == "delayed":
            peak      = cfg.get("peak", 1)
            adstocked = DelayedAdstock(decay=decay, peak=peak).transform(spend)
        else:
            raise ValueError(f"Type adstock inconnu : {ads_type!r}")

        df[f"{channel}{suffix}"] = adstocked

    return df


# ── Saturation ────────────────────────────────────────────────────────────────

def apply_saturation_all_channels(
    df: pd.DataFrame,
    adstock_suffix: str = "_adstocked",
    sat_suffix: str = "_saturated",
    config: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Applique la saturation Hill/Logistic sur les colonnes adstockées.

    Paramètres
    ----------
    df             : DataFrame avec les colonnes *_adstocked
    adstock_suffix : suffixe des colonnes adstockées (input)
    sat_suffix     : suffixe des colonnes saturées (output)
    config         : {channel: {type, S, K}} — défaut : DEFAULT_SATURATION_CONFIG

    Retourne
    --------
    DataFrame enrichi avec les colonnes *_saturated
    """
    config = config or DEFAULT_SATURATION_CONFIG
    df     = df.copy()

    for channel, cfg in config.items():
        input_col = f"{channel}{adstock_suffix}"
        if input_col not in df.columns:
            continue

        x        = df[input_col].values.astype(float)
        sat_type = cfg.get("type", "hill")
        S        = cfg.get("S", 2.0)
        K        = cfg.get("K", None)

        if sat_type == "hill":
            saturated = HillSaturation(K=K, S=S).transform(x)
        elif sat_type == "logistic":
            saturated = LogisticSaturation().transform(x)
        else:
            raise ValueError(f"Type saturation inconnu : {sat_type!r}")

        df[f"{channel}{sat_suffix}"] = saturated

    return df


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "median",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Normalise les features pour stabiliser le sampling MCMC.

    Paramètres
    ----------
    df      : DataFrame à normaliser
    columns : colonnes à normaliser (défaut : toutes les colonnes numériques)
    method  : "median" | "mean" | "minmax"

    Retourne
    --------
    (df_normalized, scalers) :
      - df_normalized : DataFrame avec colonnes normalisées
      - scalers       : {col: valeur de normalisation} pour dénormaliser
    """
    df      = df.copy()
    scalers = {}

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [c for c in columns if c != "week"]

    for col in columns:
        if col not in df.columns:
            continue

        vals = df[col].values.astype(float)

        if method == "median":
            scale = np.median(vals[vals > 0]) if np.any(vals > 0) else 1.0
        elif method == "mean":
            scale = np.mean(vals) if np.mean(vals) != 0 else 1.0
        elif method == "minmax":
            scale = vals.max() - vals.min()
            scale = scale if scale != 0 else 1.0
        else:
            raise ValueError(f"Méthode inconnue : {method!r}")

        scalers[col]   = float(scale)
        df[f"{col}_norm"] = vals / scale

    return df, scalers


def denormalize(
    values: np.ndarray,
    col: str,
    scalers: Dict[str, float],
) -> np.ndarray:
    """
    Dénormalise des valeurs à partir des scalers sauvegardés.

    Paramètres
    ----------
    values  : array normalisé
    col     : nom de la colonne originale
    scalers : dict retourné par normalize_features()
    """
    if col not in scalers:
        raise KeyError(f"Scaler introuvable pour '{col}'")
    return values * scalers[col]


# ── Pipeline complet ──────────────────────────────────────────────────────────

def full_feature_pipeline(
    df: pd.DataFrame,
    adstock_config: Optional[Dict] = None,
    saturation_config: Optional[Dict] = None,
    normalize: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Pipeline complet : adstock → saturation → normalisation.

    Paramètres
    ----------
    df                 : DataFrame brut validé
    adstock_config     : config adstock par canal
    saturation_config  : config saturation par canal
    normalize          : normalise les features finales

    Retourne
    --------
    (df_transformed, info) :
      - df_transformed : DataFrame avec toutes les transformations
      - info           : dict avec scalers et configs utilisées
    """
    # 1. Adstock
    df = apply_adstock_all_channels(df, config=adstock_config)

    # 2. Saturation
    df = apply_saturation_all_channels(df, config=saturation_config)

    # 3. Normalisation optionnelle
    scalers = {}
    if normalize:
        sat_cols = [f"{ch}_saturated" for ch in ["tv", "facebook", "search", "ooh", "print"]
                    if f"{ch}_saturated" in df.columns]
        df, scalers = normalize_features(df, columns=sat_cols + ["revenue"])

    info = {
        "adstock_config":    adstock_config or DEFAULT_ADSTOCK_CONFIG,
        "saturation_config": saturation_config or DEFAULT_SATURATION_CONFIG,
        "scalers":           scalers,
        "n_features":        len([c for c in df.columns if c.endswith("_saturated")]),
    }

    return df, info