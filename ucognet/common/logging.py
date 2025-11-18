"""
Sistema de logging personalizado para U-CogNet.
"""

import logging
import sys
from pathlib import Path

class UCogNetLogger:
    def __init__(self, name: str = "U-CogNet"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Crear directorio de logs si no existe
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Handler para archivo
        file_handler = logging.FileHandler(log_dir / "ucognet.log")
        file_handler.setLevel(logging.DEBUG)

        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Formato
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger

# Instancia global
logger = UCogNetLogger().get_logger()