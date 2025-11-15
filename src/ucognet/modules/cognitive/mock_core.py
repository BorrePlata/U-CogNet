from ucognet.core.interfaces import CognitiveCore
from ucognet.core.types import Event, Context

class MockCognitiveCore(CognitiveCore):
    def __init__(self):
        self.events = []

    def store(self, event: Event) -> None:
        self.events.append(event)

    def get_context(self) -> Context:
        return Context(recent_events=self.events[-5:], episodic_memory=[])