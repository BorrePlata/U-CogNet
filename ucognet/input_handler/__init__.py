"""
Manejador de entrada para U-CogNet.
Versión real: Soporta imágenes, video y webcam.
"""

import numpy as np
import cv2
import time
from typing import Optional, Tuple, Union
from pathlib import Path
from ..common.types import Event
from ..common.logging import logger

class InputHandler:
    """
    Manejador de entrada del sistema.
    Soporta múltiples fuentes: imágenes, video files, webcam.
    """

    def __init__(self, source: Union[str, int] = 0, width: int = 640, height: int = 480):
        self.source = source
        self.width = width
        self.height = height
        self.frame_count = 0
        self.cap = None
        self.is_video = False
        self.is_image = False

        self._initialize_source()
        logger.info(f"InputHandler inicializado con fuente: {source}")

    def _initialize_source(self):
        """Inicializa la fuente de entrada."""
        if isinstance(self.source, str):
            if Path(self.source).exists():
                if self.source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.is_image = True
                    logger.info(f"Fuente: imagen estática - {self.source}")
                else:
                    self.cap = cv2.VideoCapture(self.source)
                    self.is_video = True
                    logger.info(f"Fuente: video file - {self.source}")
            else:
                logger.error(f"Archivo no encontrado: {self.source}")
                self._fallback_simulated()
        elif isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                logger.info(f"Fuente: webcam - dispositivo {self.source}")
            else:
                logger.warning(f"No se pudo abrir webcam {self.source}, usando simulación")
                self._fallback_simulated()
        else:
            self._fallback_simulated()

    def _fallback_simulated(self):
        """Fallback a modo simulado."""
        logger.info("Usando modo simulado para entrada")
        self.is_image = False
        self.is_video = False

    def get_frame(self) -> np.ndarray:
        """
        Obtiene un frame de la fuente configurada.
        """
        if self.is_image and isinstance(self.source, str):
            # Cargar imagen
            frame = cv2.imread(self.source)
            if frame is not None:
                frame = cv2.resize(frame, (self.width, self.height))
                self.frame_count += 1
                logger.debug(f"Imagen cargada: {self.source}")
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                logger.error(f"Error cargando imagen: {self.source}")

        elif self.cap and self.cap.isOpened():
            # Capturar de video/webcam
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (self.width, self.height))
                self.frame_count += 1
                logger.debug(f"Frame {self.frame_count} capturado")
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                logger.warning("No se pudo capturar frame, reiniciando video")
                if self.is_video:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if ret:
                        frame = cv2.resize(frame, (self.width, self.height))
                        self.frame_count += 1
                        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Fallback: simulación
        return self._get_simulated_frame()

    def _get_simulated_frame(self) -> np.ndarray:
        """Genera un frame simulado."""
        self.frame_count += 1

        # Simular un frame con ruido (placeholder)
        frame = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)

        # Añadir timestamp simulado
        timestamp = time.time()

        logger.debug(f"Frame simulado {self.frame_count} generado en {timestamp}")

        return frame

    def get_frame_with_metadata(self) -> Tuple[np.ndarray, dict]:
        """Retorna frame con metadata adicional."""
        frame = self.get_frame()
        metadata = {
            'frame_id': self.frame_count,
            'timestamp': time.time(),
            'source': str(self.source),
            'resolution': f"{self.width}x{self.height}",
            'is_simulated': not (self.cap or self.is_image)
        }
        return frame, metadata

    def release(self):
        """Libera recursos."""
        if self.cap:
            self.cap.release()
            logger.info("Recursos de captura liberados")

    def __del__(self):
        """Destructor para asegurar liberación de recursos."""
        self.release()

import numpy as np
import time
from typing import Optional
from ..common.types import Event
from ..common.logging import logger

class InputHandler:
    """
    Manejador de entrada del sistema.
    Proporciona frames simulados para desarrollo inicial.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.frame_count = 0
        logger.info(f"InputHandler inicializado con resolución {width}x{height}")

    def get_frame(self) -> np.ndarray:
        """
        Obtiene un frame simulado.
        En producción: conectaría a cámara o video stream.
        """
        self.frame_count += 1

        # Simular un frame con ruido (placeholder)
        frame = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)

        # Añadir timestamp simulado
        timestamp = time.time()

        logger.debug(f"Frame {self.frame_count} generado en {timestamp}")

        return frame

    def get_frame_with_metadata(self) -> tuple:
        """Retorna frame con metadata adicional."""
        frame = self.get_frame()
        metadata = {
            'frame_id': self.frame_count,
            'timestamp': time.time(),
            'source': 'simulated'
        }
        return frame, metadata