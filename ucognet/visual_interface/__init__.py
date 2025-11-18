"""
Interfaz visual para U-CogNet.
Versión inicial: Simula renderizado para pruebas.
"""

import numpy as np
from typing import List
from ..common.types import Detection
from ..common.logging import logger

class VisualInterface:
    """
    Maneja la interfaz visual del sistema.
    Versión inicial: Simula renderizado sin display real.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.frame_count = 0
        logger.info(f"VisualInterface inicializado con resolución {width}x{height}")

    def render(self, frame: np.ndarray, detections: List[Detection],
               text: str, state: dict) -> None:
        """
        Renderiza el frame con detecciones y texto.
        Versión simulado: Solo registra la acción.
        """
        self.frame_count += 1

        # Simular renderizado
        logger.info(f"Frame {self.frame_count} renderizado: {len(detections)} detecciones, texto: {text[:50]}...")

        # En producción: aquí iría código para dibujar en pantalla
        # Por ejemplo, usando OpenCV o similar