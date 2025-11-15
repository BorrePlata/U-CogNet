from ucognet.core.interfaces import SemanticFeedback
from ucognet.core.types import Context, Detection

class MockSemanticFeedback(SemanticFeedback):
    def generate(self, context: Context, detections: list[Detection]) -> str:
        return f"Detected {len(detections)} objects."