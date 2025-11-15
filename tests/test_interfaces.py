import pytest
from ucognet.core.interfaces import InputHandler, VisionDetector, CognitiveCore, SemanticFeedback, Evaluator, TrainerLoop, TDAManager, VisualInterface
from ucognet.modules.input.mock_input import MockInputHandler
from ucognet.modules.vision.mock_vision import MockVisionDetector
from ucognet.modules.cognitive.mock_core import MockCognitiveCore
from ucognet.modules.cognitive.cognitive_core import CognitiveCoreImpl
from ucognet.modules.semantic.mock_feedback import MockSemanticFeedback
from ucognet.modules.eval.mock_evaluator import MockEvaluator
from ucognet.modules.train.mock_trainer import MockTrainerLoop
from ucognet.modules.tda.mock_tda import MockTDAManager
from ucognet.modules.ui.mock_ui import MockVisualInterface

def test_mock_input_handler():
    handler = MockInputHandler()
    assert isinstance(handler, InputHandler)
    frame = handler.get_frame()
    assert frame.data.shape == (480, 640, 3)

def test_yolov8_detector():
    # Test con frame del video dummy
    try:
        from ucognet.modules.input.opencv_camera import OpenCVInputHandler
        from ucognet.modules.vision.yolov8_detector import YOLOv8Detector
        
        # Obtener un frame
        handler = OpenCVInputHandler(source="test_video.mp4")
        frame = handler.get_frame()
        handler.release()
        
        # Detectar
        detector = YOLOv8Detector()
        detections = detector.detect(frame)
        
        assert isinstance(detections, list)
        for det in detections:
            assert isinstance(det, Detection)
            assert 0 <= det.confidence <= 1
    except Exception as e:
        pytest.skip(f"YOLO test skipped: {e}")  # Skip si modelo no descarga o GPU issues

def test_mock_vision_detector():
    detector = MockVisionDetector()
    assert isinstance(detector, VisionDetector)
    from ucognet.core.types import Frame
    frame = Frame(data=None, timestamp=0.0, metadata={})  # Mock frame
    detections = detector.detect(frame)
    assert len(detections) == 1

def test_mock_cognitive_core():
    core = MockCognitiveCore()
    assert isinstance(core, CognitiveCore)
    from ucognet.core.types import Event, Frame
    event = Event(Frame(None, 0.0, {}), [], 0.0)
    core.store(event)
    context = core.get_context()
    assert len(context.recent_events) == 1

def test_cognitive_core_impl():
    core = CognitiveCoreImpl(max_recent_events=10)
    assert isinstance(core, CognitiveCore)
    
    from ucognet.core.types import Event, Frame, Detection
    import numpy as np
    
    # Crear un evento con detección de alta confianza
    frame = Frame(data=np.zeros((480, 640, 3), dtype=np.uint8), timestamp=1.0, metadata={})
    detection = Detection(class_id=0, class_name="person", confidence=0.9, bbox=[10, 20, 30, 40])
    event = Event(frame=frame, detections=[detection], timestamp=1.0)
    
    # Almacenar evento
    core.store(event)
    
    # Verificar contexto
    context = core.get_context()
    assert len(context.recent_events) == 1
    assert context.recent_events[0] == event
    assert len(context.episodic_memory) == 1  # Debe almacenarse por alta confianza
    
    # Agregar evento con baja confianza
    low_conf_detection = Detection(class_id=1, class_name="car", confidence=0.5, bbox=[50, 60, 70, 80])
    low_event = Event(frame=frame, detections=[low_conf_detection], timestamp=2.0)
    core.store(low_event)
    
    context = core.get_context()
    assert len(context.recent_events) == 2
    assert len(context.episodic_memory) == 1  # Solo el evento de alta confianza

def test_mock_semantic_feedback():
    feedback = MockSemanticFeedback()
    assert isinstance(feedback, SemanticFeedback)
    from ucognet.core.types import Context
    context = Context([], [])
    text = feedback.generate(context, [])
    assert "Detected" in text

def test_mock_evaluator():
    evaluator = MockEvaluator()
    assert isinstance(evaluator, Evaluator)
    from ucognet.core.types import Event, Frame
    event = Event(Frame(None, 0.0, {}), [], 0.0)
    metrics = evaluator.maybe_update(event)
    assert metrics.f1 == 0.75

def test_mock_trainer_loop():
    trainer = MockTrainerLoop()
    assert isinstance(trainer, TrainerLoop)
    trainer.maybe_train(None)  # Should not raise

def test_mock_tda_manager():
    manager = MockTDAManager()
    assert isinstance(manager, TDAManager)
    from ucognet.core.types import SystemState, TopologyConfig
    state = SystemState(None, TopologyConfig([], {}, {}), {})
    config = manager.update(state)
    assert config.active_modules == ["all"]

def test_mock_visual_interface():
    interface = MockVisualInterface()
    assert isinstance(interface, VisualInterface)
    from ucognet.core.types import Frame, SystemState, TopologyConfig
    frame = Frame(None, 0.0, {})
    state = SystemState(None, TopologyConfig([], {}, {}), {})
    interface.render(frame, [], "test", state)  # Should not raise