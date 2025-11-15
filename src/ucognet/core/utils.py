from typing import Optional
from .types import Event, Frame, Detection, SystemState, Metrics, TopologyConfig

def build_event(frame: Frame, detections: list[Detection]) -> Event:
    return Event(frame=frame, detections=detections, timestamp=frame.timestamp)

def build_system_state(metrics: Optional[Metrics]) -> SystemState:
    return SystemState(metrics=metrics, topology=TopologyConfig([], {}, {}), load={})