from typing import Protocol, runtime_checkable, List, Optional
from .types import Frame, Detection, Event, Context, Metrics, SystemState, TopologyConfig

@runtime_checkable
class InputHandler(Protocol):
    def get_frame(self) -> Frame: ...

@runtime_checkable
class VisionDetector(Protocol):
    def detect(self, frame: Frame) -> List[Detection]: ...

@runtime_checkable
class CognitiveCore(Protocol):
    def store(self, event: Event) -> None: ...
    def get_context(self) -> Context: ...

@runtime_checkable
class SemanticFeedback(Protocol):
    def generate(self, context: Context, detections: List[Detection]) -> str: ...

@runtime_checkable
class Evaluator(Protocol):
    def maybe_update(self, event: Event) -> Optional[Metrics]: ...

@runtime_checkable
class TrainerLoop(Protocol):
    def maybe_train(self, metrics: Optional[Metrics]) -> None: ...

@runtime_checkable
class TDAManager(Protocol):
    def update(self, state: SystemState) -> TopologyConfig: ...

@runtime_checkable
class VisualInterface(Protocol):
    def render(self, frame: Frame, detections: List[Detection], text: str, state: SystemState) -> None: ...