from typing import Optional
from ucognet.core.interfaces import TrainerLoop
from ucognet.core.types import Metrics

class MockTrainerLoop(TrainerLoop):
    def maybe_train(self, metrics: Optional[Metrics]) -> None:
        pass  # No training in mock