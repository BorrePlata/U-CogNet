import pytest
from ucognet.runtime.engine import Engine
from ucognet.modules.input.mock_input import MockInputHandler
from ucognet.modules.vision.mock_vision import MockVisionDetector
from ucognet.modules.cognitive.mock_core import MockCognitiveCore
from ucognet.modules.semantic.mock_feedback import MockSemanticFeedback
from ucognet.modules.eval.mock_evaluator import MockEvaluator
from ucognet.modules.train.mock_trainer import MockTrainerLoop
from ucognet.modules.tda.mock_tda import MockTDAManager
from ucognet.modules.ui.mock_ui import MockVisualInterface

def test_engine_creation():
    engine = Engine(
        input_handler=MockInputHandler(),
        vision_detector=MockVisionDetector(),
        cognitive_core=MockCognitiveCore(),
        semantic_feedback=MockSemanticFeedback(),
        evaluator=MockEvaluator(),
        trainer_loop=MockTrainerLoop(),
        tda_manager=MockTDAManager(),
        visual_interface=MockVisualInterface(),
    )
    assert engine.input_handler is not None

def test_engine_step(capsys):
    engine = Engine(
        input_handler=MockInputHandler(),
        vision_detector=MockVisionDetector(),
        cognitive_core=MockCognitiveCore(),
        semantic_feedback=MockSemanticFeedback(),
        evaluator=MockEvaluator(),
        trainer_loop=MockTrainerLoop(),
        tda_manager=MockTDAManager(),
        visual_interface=MockVisualInterface(),
    )
    engine.step()
    captured = capsys.readouterr()
    assert "Rendering: 1 detections" in captured.out