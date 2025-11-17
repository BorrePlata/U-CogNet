# /mnt/c/Users/desar/Documents/Science/UCogNet/src/ucognet/modules/mycelium/types.py
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

@dataclass
class MycoContext:
    """Contexto de decisión para MycoNet"""
    task_id: str
    phase: str  # "exploration", "exploitation", "safety_check", "adaptation"
    metrics: Dict[str, float]  # precisión, recall, energía, riesgo, etc.
    timestamp: float
    extra: Optional[Dict[str, Any]] = None

@dataclass
class MycoSignal:
    """Señal de feromona o comunicación entre nodos"""
    source: str
    target: str
    strength: float
    signal_type: str  # "pheromone", "attention", "resource", "safety"
    context: MycoContext
    timestamp: float

@dataclass
class MycoPath:
    """Ruta de activación a través del grafo micelial"""
    nodes: List[str]
    edges: List[Tuple[str, str]]
    total_weight: float
    safety_score: float
    expected_reward: float
    context: MycoContext

@dataclass
class MycoMetrics:
    """Métricas de rendimiento del sistema micelial"""
    path_efficiency: float
    safety_compliance: float
    adaptation_rate: float
    resource_distribution: Dict[str, float]
    emergent_behaviors: List[str]
    pheromone_entropy: float
    convergence_speed: float