"""
Núcleo cognitivo para U-CogNet.
Versión inicial: Memoria a corto plazo con buffer circular.
"""

from collections import deque
import time
from typing import List, Dict, Any, Optional
from ..common.types import Event, Context
from ..common.logging import logger

class CognitiveCore:
    """
    Núcleo cognitivo del sistema.
    Maneja memoria a corto plazo y contexto temporal.
    """

    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self.event_buffer = deque(maxlen=buffer_size)
        self.context_window = 10  # Ventana para contexto
        logger.info(f"CognitiveCore inicializado con buffer de {buffer_size} eventos")

    def store(self, event: Event) -> None:
        """Almacena un evento en el buffer."""
        self.event_buffer.append(event)
        logger.debug(f"Evento almacenado. Buffer size: {len(self.event_buffer)}")

    def get_context(self) -> Context:
        """Obtiene el contexto temporal actual."""
        recent_events = list(self.event_buffer)[-self.context_window:]

        current_state = {
            'total_events': len(self.event_buffer),
            'recent_detections': sum(len(e.detections) for e in recent_events),
            'last_update': time.time()
        }

        context = Context(
            window_size=self.context_window,
            recent_events=recent_events,
            current_state=current_state
        )

        return context

    def get_recent_events(self, n: int = 5) -> List[Event]:
        """Obtiene los n eventos más recientes."""
        return list(self.event_buffer)[-n:]

    def clear_buffer(self) -> None:
        """Limpia el buffer de eventos."""
        self.event_buffer.clear()
        logger.info("Buffer de eventos limpiado")