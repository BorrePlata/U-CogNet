from ucognet.core.interfaces import TDAManager
from ucognet.core.types import SystemState, TopologyConfig

class MockTDAManager(TDAManager):
    def update(self, state: SystemState) -> TopologyConfig:
        return TopologyConfig(active_modules=["all"], connections={}, resource_allocation={"cpu": 0.5})