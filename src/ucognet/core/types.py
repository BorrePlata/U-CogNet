"""
Tipos de Datos para U-CogNet
Definiciones de tipos y estructuras de datos compartidas
"""

from typing import Dict, List, Any, Optional, Protocol, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class Frame:
    """Representa un frame de video o datos de entrada"""
    data: np.ndarray
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    frame_id: Optional[int] = None


@dataclass
class Detection:
    """Representa una detección en un frame"""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la detección a diccionario serializable"""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'features': self.features.tolist() if self.features is not None else None,
            'metadata': self.metadata
        }


@dataclass
class Event:
    """Representa un evento en el sistema"""
    frame: Frame
    detections: List[Detection]
    timestamp: datetime
    event_type: str = "detection"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el evento a diccionario serializable"""
        return {
            'detections': [d.to_dict() for d in self.detections],
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'metadata': self.metadata
        }


@dataclass
class Metrics:
    """Métricas de rendimiento del sistema"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    latency_ms: float = 0.0
    throughput_fps: float = 0.0
    memory_usage_mb: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TopologyConfig:
    """Configuración de topología de red"""
    nodes: List[str] = field(default_factory=list)
    edges: Dict[str, List[str]] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemState:
    """Estado completo del sistema"""
    metrics: Optional[Metrics] = None
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    load: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "active"


@dataclass
class TrainingData:
    """Datos de entrenamiento"""
    inputs: np.ndarray
    targets: np.ndarray
    weights: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuración de modelo"""
    architecture: str
    input_shape: List[int]
    output_shape: List[int]
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    optimizer_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Resultado de optimización"""
    parameters: Dict[str, Any]
    score: float
    iterations: int
    converged: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MycelialNode:
    """Nodo en la red micelial"""
    id: str
    position: np.ndarray
    connections: List[str] = field(default_factory=list)
    resources: Dict[str, float] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MycelialEdge:
    """Conexión en la red micelial"""
    source: str
    target: str
    weight: float = 1.0
    pheromone: float = 0.0
    distance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveTrace:
    """Traza cognitiva para MTC"""
    trace_id: str
    timestamp: datetime
    module: str
    operation: str
    input_data: Any
    output_data: Any
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Configuración de experimento"""
    name: str
    description: str
    modules: List[str]
    parameters: Dict[str, Any]
    metrics: List[str]
    duration_seconds: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExperimentResult:
    """Resultado de experimento"""
    config: ExperimentConfig
    metrics: Dict[str, float]
    traces: List[CognitiveTrace]
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None


# Protocolos para interfaces

class CognitiveModule(Protocol):
    """Protocolo para módulos cognitivos"""
    async def process(self, input_data: Any) -> Any:
        """Procesa datos de entrada"""
        ...

    async def get_metrics(self) -> Metrics:
        """Obtiene métricas del módulo"""
        ...

    def get_config(self) -> Dict[str, Any]:
        """Obtiene configuración del módulo"""
        ...


class Optimizer(Protocol):
    """Protocolo para optimizadores"""
    async def optimize(self, parameters: Dict[str, Any], data: Any) -> OptimizationResult:
        """Optimiza parámetros"""
        ...

    def get_best_parameters(self) -> Dict[str, Any]:
        """Obtiene mejores parámetros encontrados"""
        ...


class EvaluatorProtocol(Protocol):
    """Protocolo para evaluadores"""
    async def evaluate(self, predictions: Any, ground_truth: Any) -> Metrics:
        """Evalúa predicciones contra ground truth"""
        ...

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Obtiene configuración de evaluación"""
        ...


class TrainerProtocol(Protocol):
    """Protocolo para entrenadores"""
    async def train(self, data: TrainingData, config: ModelConfig) -> Any:
        """Entrena un modelo"""
        ...

    async def validate(self, data: TrainingData) -> Metrics:
        """Valida un modelo"""
        ...


# Tipos union para flexibilidad
DataType = Union[np.ndarray, List[Any], Dict[str, Any]]
ConfigType = Dict[str, Any]
ResultType = Union[Dict[str, Any], np.ndarray, List[Any]]