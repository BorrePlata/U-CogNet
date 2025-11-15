from ucognet.runtime.engine import Engine
from ucognet.modules.input.opencv_camera import OpenCVInputHandler
from ucognet.modules.vision.yolov8_detector import YOLOv8Detector
from ucognet.modules.cognitive.cognitive_core import CognitiveCoreImpl
from ucognet.modules.semantic.mock_feedback import MockSemanticFeedback
from ucognet.modules.eval.mock_evaluator import MockEvaluator
from ucognet.modules.train.mock_trainer import MockTrainerLoop
from ucognet.modules.tda.mock_tda import MockTDAManager
from ucognet.modules.ui.mock_ui import MockVisualInterface

def main() -> None:
    engine = Engine(
        input_handler=OpenCVInputHandler(source="videoplayback.webm"),
        vision_detector=YOLOv8Detector(),
        cognitive_core=CognitiveCoreImpl(),
        semantic_feedback=MockSemanticFeedback(),
        evaluator=MockEvaluator(),
        trainer_loop=MockTrainerLoop(),
        tda_manager=MockTDAManager(),
        visual_interface=MockVisualInterface(),
    )
    for _ in range(5):
        engine.step()

if __name__ == "__main__":
    main()