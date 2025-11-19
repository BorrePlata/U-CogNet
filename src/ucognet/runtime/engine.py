from ucognet.core.interfaces import *
from ucognet.core.utils import build_event, build_system_state
from ucognet.core.tracing import get_event_bus, EventType

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
        trace_manager: Optional[TraceManager] = None,
    ):
        self.input_handler = input_handler
        self.vision_detector = vision_detector
        self.cognitive_core = cognitive_core
        self.semantic_feedback = semantic_feedback
        self.evaluator = evaluator
        self.trainer_loop = trainer_loop
        self.tda_manager = tda_manager
        self.visual_interface = visual_interface
        self.trace_manager = trace_manager

        # Inicializar bus de eventos para trazabilidad
        self.event_bus = get_event_bus()

    def step(self) -> None:
        # Emitir evento de inicio de paso
        step_id = self.event_bus.emit(
            EventType.MODULE_INTERACTION,
            "Engine",
            outputs={"operation": "step", "phase": "start"}
        )

        try:
            frame = self.input_handler.get_frame()
            detections = self.vision_detector.detect(frame)
            event = build_event(frame, detections)

            # Trazar detección
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "VisionDetector",
                outputs={"detections_count": len(detections)},
                trace_id=step_id
            )

            self.cognitive_core.store(event)
            context = self.cognitive_core.get_context()

            # Trazar procesamiento cognitivo
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "CognitiveCore",
                outputs={"context_size": len(context.recent_events)},
                trace_id=step_id
            )

            text = self.semantic_feedback.generate(context, detections)
            metrics = self.evaluator.maybe_update(event)

            # Trazar evaluación
            if metrics:
                self.event_bus.emit(
                    EventType.EVALUATION_METRIC,
                    "Evaluator",
                    metrics=metrics.__dict__,
                    trace_id=step_id
                )

            self.trainer_loop.maybe_train(metrics)
            state = build_system_state(metrics)
            topology = self.tda_manager.update(state)
            state.topology = topology

            # Trazar cambios de topología
            if topology:
                self.event_bus.emit(
                    EventType.TOPOLOGY_CHANGE,
                    "TDAManager",
                    outputs={"topology": topology.__dict__},
                    trace_id=step_id
                )

            self.visual_interface.render(frame, detections, text, state)

            # Evento de paso completado exitosamente
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "Engine",
                outputs={"operation": "step", "phase": "completed", "success": True},
                trace_id=step_id
            )

        except Exception as e:
            # Trazar error
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "Engine",
                outputs={"operation": "step", "phase": "error", "error": str(e)},
                trace_id=step_id
            )
            raise