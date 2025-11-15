from ucognet.core.interfaces import VisionDetector
from ucognet.core.types import Frame, Detection

class MockVisionDetector(VisionDetector):
    def detect(self, frame: Frame) -> list[Detection]:
        return [Detection(class_id=0, class_name="mock", confidence=0.5, bbox=[10, 10, 50, 50])]