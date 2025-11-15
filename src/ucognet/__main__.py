from ucognet.runtime.engine import Engine
from ucognet.modules.input.opencv_camera import OpenCVInputHandler
from ucognet.modules.vision.yolov8_detector import YOLOv8Detector
from ucognet.modules.cognitive.cognitive_core import CognitiveCoreImpl
from ucognet.modules.semantic.mock_feedback import MockSemanticFeedback
from ucognet.modules.eval.mock_evaluator import MockEvaluator
from ucognet.modules.train.mock_trainer import MockTrainerLoop
from ucognet.modules.tda.mock_tda import MockTDAManager
from ucognet.modules.ui.opencv_ui import OpenCVVisualInterface

def main() -> None:
    engine = Engine(
        input_handler=OpenCVInputHandler(source="videoplayback.webm"),
        vision_detector=YOLOv8Detector(use_mediapipe=False),
        cognitive_core=CognitiveCoreImpl(),
        semantic_feedback=MockSemanticFeedback(),
        evaluator=MockEvaluator(),
        trainer_loop=MockTrainerLoop(),
        tda_manager=MockTDAManager(),
        visual_interface=OpenCVVisualInterface(),
    )
    
    try:
        while True:
            engine.step()
    except KeyboardInterrupt:
        print("Sistema detenido por el usuario")
    finally:
        # Cerrar recursos
        engine.input_handler.release()
        engine.visual_interface.close()

if __name__ == "__main__":
    main()