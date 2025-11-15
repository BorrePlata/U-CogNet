import numpy as np
from ucognet.core.interfaces import InputHandler
from ucognet.core.types import Frame

class MockInputHandler(InputHandler):
    def get_frame(self) -> Frame:
        return Frame(data=np.zeros((480, 640, 3), dtype=np.uint8), timestamp=0.0, metadata={})