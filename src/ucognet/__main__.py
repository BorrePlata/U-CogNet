from ucognet.runtime.engine import Engine
from ucognet.modules.input.opencv_camera import OpenCVInputHandler
from ucognet.modules.vision.yolov8_detector import YOLOv8Detector
from ucognet.modules.cognitive.cognitive_core import CognitiveCoreImpl
from ucognet.modules.semantic.rule_based import RuleBasedSemanticFeedback
from ucognet.modules.eval.basic_evaluator import BasicEvaluator
from ucognet.modules.train.mock_trainer import MockTrainerLoop
from ucognet.modules.tda.basic_tda import BasicTDAManager
from ucognet.modules.ui.opencv_ui import OpenCVVisualInterface

def main() -> None:
    engine = Engine(
        input_handler=OpenCVInputHandler(source="videoplayback.webm"),
        vision_detector=YOLOv8Detector(use_mediapipe=False),  # Temporalmente desactivado para debugging
        cognitive_core=CognitiveCoreImpl(),
        semantic_feedback=RuleBasedSemanticFeedback(),
        evaluator=BasicEvaluator(),
        trainer_loop=MockTrainerLoop(),
        tda_manager=BasicTDAManager(),
        visual_interface=OpenCVVisualInterface(record_on_crowd=True, record_duration=30),
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