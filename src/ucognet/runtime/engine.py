from ucognet.core.interfaces import *
from ucognet.core.utils import build_event, build_system_state

class Engine:
    def __init__(
        self,
        input_handler: InputHandler,
        vision_detector: VisionDetector,
        cognitive_core: CognitiveCore,
        semantic_feedback: SemanticFeedback,
        evaluator: Evaluator,
        trainer_loop: TrainerLoop,
        tda_manager: TDAManager,
        visual_interface: VisualInterface,
    ):
        self.input_handler = input_handler
        self.vision_detector = vision_detector
        self.cognitive_core = cognitive_core
        self.semantic_feedback = semantic_feedback
        self.evaluator = evaluator
        self.trainer_loop = trainer_loop
        self.tda_manager = tda_manager
        self.visual_interface = visual_interface

    def step(self) -> None:
        frame = self.input_handler.get_frame()
        detections = self.vision_detector.detect(frame)
        event = build_event(frame, detections)
        self.cognitive_core.store(event)
        context = self.cognitive_core.get_context()
        text = self.semantic_feedback.generate(context, detections)
        metrics = self.evaluator.maybe_update(event)
        self.trainer_loop.maybe_train(metrics)
        state = build_system_state(metrics)
        topology = self.tda_manager.update(state)
        state.topology = topology
        self.visual_interface.render(frame, detections, text, state)