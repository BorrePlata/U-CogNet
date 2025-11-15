from ucognet.core.interfaces import VisualInterface
from ucognet.core.types import Frame, Detection, SystemState

class MockVisualInterface(VisualInterface):
    def render(self, frame: Frame, detections: list[Detection], text: str, state: SystemState) -> None:
        print(f"Rendering: {len(detections)} detections, text: {text}")