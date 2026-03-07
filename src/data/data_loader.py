"""
data_loader.py
--------------
Chargement et préparation des données MMM.

TODO :
- load_raw_data() : charge le CSV brut
- load_market_data() : filtre sur un marché spécifique
- load_all_markets() : retourne un dict {market: df}
- split_train_test() : découpage temporel (80/20)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


DATA_PATH = Path(__file__).parent.parent.parent / "data" / "synthetic" / "mmm_multi_market.csv"


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Charge le dataset brut depuis le CSV."""
    # TODO : implémenter
    raise NotImplementedError


def load_market_data(market: str, path: Path = DATA_PATH) -> pd.DataFrame:
    """Charge et filtre les données d'un marché spécifique."""
    # TODO : implémenter
    raise NotImplementedError


def load_all_markets(path: Path = DATA_PATH) -> Dict[str, pd.DataFrame]:
    """Retourne un dictionnaire {market_code: dataframe}."""
    # TODO : implémenter
    raise NotImplementedError


def split_train_test(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporel : les dernières semaines = test.
    Pas de shuffle (ordre temporel obligatoire en time series).
    """
    # TODO : implémenter
    raise NotImplementedError
