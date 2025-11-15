from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np

@dataclass
class Frame:
    data: np.ndarray
    timestamp: float
    metadata: Dict[str, Any]

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]

@dataclass
class Event:
    frame: Frame
    detections: List[Detection]
    timestamp: float

@dataclass
class Context:
    recent_events: List[Event]
    episodic_memory: List[Dict]

@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    mcc: float
    map: float

@dataclass
class SystemState:
    metrics: Optional[Metrics]
    topology: 'TopologyConfig'
    load: Dict[str, float]

@dataclass
class TopologyConfig:
    active_modules: List[str]
    connections: Dict[str, List[str]]
    resource_allocation: Dict[str, float]