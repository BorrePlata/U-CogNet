import cv2
import numpy as np
from ucognet.core.interfaces import InputHandler
from ucognet.core.types import Frame

class OpenCVInputHandler(InputHandler):
    def __init__(self, source=0):  # 0 para webcam, o path a video
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"No se pudo abrir la fuente de video: {source}")

    def get_frame(self) -> Frame:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("No se pudo leer el frame del video")
        # Convertir BGR a RGB si es necesario, pero para consistencia, mantener BGR
        timestamp = cv2.getTickCount() / cv2.getTickFrequency()
        return Frame(data=frame, timestamp=timestamp, metadata={"source": "opencv", "shape": frame.shape})

    def release(self):
        self.cap.release()