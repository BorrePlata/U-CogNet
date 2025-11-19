"""
Núcleo cognitivo para U-CogNet.
Versión inicial: Memoria a corto plazo con buffer circular.
"""

from collections import deque
import time
from typing import List, Dict, Any, Optional
from ..common.types import Event, Context
from ..common.logging import logger

# Importar trazabilidad
try:
    from ucognet.core.tracing import get_event_bus, EventType
    TRACING_ENABLED = True
except ImportError:
    TRACING_ENABLED = False
    logger.warning("Módulo de trazabilidad no disponible")

class CognitiveCore:
    """
    Núcleo cognitivo del sistema.
    Maneja memoria a corto plazo y contexto temporal.
    """

    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self.event_buffer = deque(maxlen=buffer_size)
        self.context_window = 10  # Ventana para contexto

        # Inicializar trazabilidad
        self.event_bus = get_event_bus() if TRACING_ENABLED else None

        logger.info(f"CognitiveCore inicializado con buffer de {buffer_size} eventos")

        # Emitir evento de inicialización
        if self.event_bus:
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "CognitiveCore",
                outputs={"operation": "initialization", "buffer_size": buffer_size}
            )

    def store(self, event: Event) -> None:
        """Almacena un evento en el buffer."""
        start_time = time.time()

        self.event_buffer.append(event)
        buffer_size = len(self.event_buffer)

        processing_time = time.time() - start_time

        # Emitir evento de trazabilidad
        if self.event_bus:
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "CognitiveCore",
                inputs={"event_detections": len(event.detections)},
                outputs={"buffer_size": buffer_size, "operation": "store"},
                metrics={"processing_time_ms": processing_time * 1000},
                explanation=f"Stored event with {len(event.detections)} detections"
            )

        logger.debug(f"Evento almacenado. Buffer size: {buffer_size}")

    def get_context(self) -> Context:
        """Obtiene el contexto temporal actual."""
        start_time = time.time()

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

        processing_time = time.time() - start_time

        # Emitir evento de trazabilidad
        if self.event_bus:
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "CognitiveCore",
                outputs={
                    "operation": "get_context",
                    "context_events": len(recent_events),
                    "total_events": current_state['total_events']
                },
                metrics={"processing_time_ms": processing_time * 1000}
            )

        return context

    def get_recent_events(self, n: int = 5) -> List[Event]:
        """Obtiene los n eventos más recientes."""
        recent_events = list(self.event_buffer)[-n:]

        # Emitir evento de trazabilidad
        if self.event_bus:
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "CognitiveCore",
                outputs={"operation": "get_recent_events", "count": len(recent_events)}
            )

        return recent_events

    def clear_buffer(self) -> None:
        """Limpia el buffer de eventos."""
        old_size = len(self.event_buffer)
        self.event_buffer.clear()

        # Emitir evento de trazabilidad
        if self.event_bus:
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "CognitiveCore",
                outputs={"operation": "clear_buffer", "cleared_events": old_size}
            )

        logger.info("Buffer de eventos limpiado")