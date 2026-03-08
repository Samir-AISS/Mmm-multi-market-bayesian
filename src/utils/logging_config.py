"""
logging_config.py
-----------------
Configuration centralisée du logging pour le projet MMM.

Usage:
    from src.utils.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Modèle entraîné avec succès pour le marché FR")
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: bool = False,
) -> logging.Logger:
    """
    Retourne un logger configuré.

    Paramètres
    ----------
    name     : nom du module (utiliser __name__)
    level    : niveau de log ("DEBUG", "INFO", "WARNING", "ERROR")
    log_file : si True, écrit aussi dans results/reports/mmm_YYYYMMDD.log
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # déjà configuré

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # ── Handler console ───────────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(console)

    # ── Handler fichier (optionnel) ───────────────────────────────────────────
    if log_file:
        log_dir  = Path(__file__).parent.parent.parent / "results" / "reports"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"mmm_{datetime.now().strftime('%Y%m%d')}.log"

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(file_handler)

    return logger


def set_global_level(level: str = "INFO") -> None:
    """
    Change le niveau de log de tous les loggers du projet MMM.

    Utile pour passer en mode DEBUG pendant le développement :
        set_global_level("DEBUG")
    """
    for name, lgr in logging.Logger.manager.loggerDict.items():
        if name.startswith("src.") and isinstance(lgr, logging.Logger):
            lgr.setLevel(getattr(logging, level.upper(), logging.INFO))
            