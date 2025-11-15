import pytest
from ucognet.core.types import Frame, Detection, Event, Context, Metrics, SystemState, TopologyConfig
import numpy as np

def test_frame_creation():
    data = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = Frame(data=data, timestamp=0.0, metadata={"source": "test"})
    assert frame.data.shape == (480, 640, 3)
    assert frame.timestamp == 0.0
    assert frame.metadata["source"] == "test"

def test_detection_creation():
    detection = Detection(class_id=1, class_name="tank", confidence=0.95, bbox=[10, 20, 30, 40])
    assert detection.class_id == 1
    assert detection.confidence == 0.95

def test_event_creation():
    frame = Frame(data=np.zeros((100, 100, 3)), timestamp=1.0, metadata={})
    detections = [Detection(0, "test", 0.8, [0, 0, 10, 10])]
    event = Event(frame=frame, detections=detections, timestamp=1.0)
    assert len(event.detections) == 1
    assert event.timestamp == 1.0

def test_context_creation():
    events = [Event(Frame(np.zeros((100, 100, 3)), 0.0, {}), [], 0.0)]
    context = Context(recent_events=events, episodic_memory=[])
    assert len(context.recent_events) == 1

def test_metrics_creation():
    metrics = Metrics(precision=0.9, recall=0.8, f1=0.85, mcc=0.7, map=0.75)
    assert metrics.f1 == 0.85

def test_system_state_creation():
    topology = TopologyConfig(active_modules=["vision"], connections={}, resource_allocation={"cpu": 0.5})
    state = SystemState(metrics=None, topology=topology, load={"cpu": 0.3})
    assert state.topology.active_modules == ["vision"]

def test_topology_config_creation():
    config = TopologyConfig(active_modules=["input", "vision"], connections={"input": ["vision"]}, resource_allocation={"gpu": 1.0})
    assert "input" in config.active_modules
    assert config.connections["input"] == ["vision"]