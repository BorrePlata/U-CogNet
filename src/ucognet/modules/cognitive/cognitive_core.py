from typing import List, Dict, Optional
from collections import deque
import time
from ...core.interfaces import CognitiveCore
from ...core.types import Event, Context

class CognitiveCoreImpl(CognitiveCore):
    """CognitiveCore con buffers para procesamiento en tiempo real."""

    def __init__(self, max_recent_events: int = 100, episodic_threshold: float = 0.8):
        self.recent_events = deque(maxlen=max_recent_events)
        self.episodic_memory: List[Dict] = []
        self.episodic_threshold = episodic_threshold

    def store(self, event: Event) -> None:
        """Almacena un evento en el buffer reciente y evalúa para memoria episódica."""
        self.recent_events.append(event)

        # Lógica básica para memoria episódica: eventos con alta confianza
        high_conf_detections = [d for d in event.detections if d.confidence > self.episodic_threshold]
        if high_conf_detections:
            episodic_entry = {
                "timestamp": event.timestamp,
                "detections": high_conf_detections,
                "frame_shape": event.frame.data.shape,
                "summary": f"{len(high_conf_detections)} high-confidence detections"
            }
            self.episodic_memory.append(episodic_entry)

            # Mantener memoria episódica limitada (últimos 1000 eventos importantes)
            if len(self.episodic_memory) > 1000:
                self.episodic_memory.pop(0)

    def get_context(self) -> Context:
        """Devuelve el contexto actual con eventos recientes y memoria episódica."""
        return Context(
            recent_events=list(self.recent_events),
            episodic_memory=self.episodic_memory.copy()
        )