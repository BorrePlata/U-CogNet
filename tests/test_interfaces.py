import pytest
from ucognet.core.interfaces import InputHandler, VisionDetector, CognitiveCore, SemanticFeedback, Evaluator, TrainerLoop, TDAManager, VisualInterface
from ucognet.modules.input.mock_input import MockInputHandler
from ucognet.modules.vision.mock_vision import MockVisionDetector
from ucognet.modules.cognitive.mock_core import MockCognitiveCore
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

def test_opencv_input_handler():
    # Test básico, asume que hay webcam o video
    try:
        from ucognet.modules.input.opencv_camera import OpenCVInputHandler
        handler = OpenCVInputHandler(source=0)  # Webcam
        assert isinstance(handler, InputHandler)
        frame = handler.get_frame()
        assert frame.data.ndim == 3  # Imagen
        handler.release()
    except (ValueError, RuntimeError):
        pytest.skip("No webcam disponible")

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