# -*- coding: utf-8 -*-
"""
main.py
Punto de entrada OCT_Static_Software V6.0.

Flags:
  --mock    Usar MockSpectrometer (sin hardware físico)
"""

import sys
import os
import logging

from constants import SOFTWARE_VERSION

# ── Configurar logging antes de importar nada más ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("oct.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtCore import QLocale
from PyQt5.QtWidgets import QApplication
from gui.main_gui import OCTGUI


def configure_decimal_locale() -> None:
    """Fijar el punto como separador decimal de la interfaz Qt."""
    QLocale.setDefault(QLocale(QLocale.C))

if __name__ == "__main__":
    use_mock = "--mock" in sys.argv
    logger.info("Iniciando %s (mock=%s)", SOFTWARE_VERSION, use_mock)
    configure_decimal_locale()
    app = QApplication(sys.argv)
    win = OCTGUI(use_mock=use_mock)
    win.show()
    sys.exit(app.exec_())
