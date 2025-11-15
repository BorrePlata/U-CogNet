from typing import Optional
from ucognet.core.interfaces import Evaluator
from ucognet.core.types import Event, Metrics

class MockEvaluator(Evaluator):
    def maybe_update(self, event: Event) -> Optional[Metrics]:
        return Metrics(precision=0.8, recall=0.7, f1=0.75, mcc=0.6, map=0.65)