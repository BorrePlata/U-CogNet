"""
Tipos de datos compartidos y estructuras comunes para U-CogNet.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Detection:
    """Representa una detección del sistema de visión."""
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    timestamp: datetime

@dataclass
class Event:
    """Evento cognitivo básico."""
    timestamp: datetime
    detections: List[Detection]
    frame: Optional[np.ndarray] = None
    context: Optional[Dict[str, Any]] = None

@dataclass
class Context:
    """Contexto temporal del sistema."""
    window_size: int
    recent_events: List[Event]
    current_state: Dict[str, Any]

@dataclass
class Metrics:
    """Métricas de evaluación del sistema."""
    precision: float
    recall: float
    f1_score: float
    mcc: float  # Matthews Correlation Coefficient
    latency_ms: float
    throughput_fps: float

@dataclass
class SystemState:
    """Estado global del sistema para TDA."""
    active_modules: List[str]
    resource_usage: Dict[str, float]
    performance_metrics: Metrics
    topology_config: Dict[str, Any]

@dataclass
class TopologyConfig:
    """Configuración de topología dinámica."""
    active_modules: List[str]
    connections: Dict[str, List[str]]
    resource_allocation: Dict[str, float]