from typing import List
from .yolov8_detector import YOLOv8Detector
from .mediapipe_detector import MediaPipeDetector
from ...core.interfaces import VisionDetector
from ...core.types import Frame, Detection

class CombinedVisionDetector(VisionDetector):
    """Detector que combina YOLOv8 y MediaPipe para detección completa."""

    def __init__(self):
        self.yolo_detector = YOLOv8Detector()
        self.mediapipe_detector = MediaPipeDetector()

    def detect(self, frame: Frame) -> List[Detection]:
        """Combina detecciones de YOLOv8 y MediaPipe."""
        # Obtener detecciones de YOLOv8 (objetos)
        yolo_detections = self.yolo_detector.detect(frame)

        # Obtener detecciones de MediaPipe (poses, manos, rostro)
        mediapipe_detections = self.mediapipe_detector.detect(frame)

        # Combinar todas las detecciones
        all_detections = yolo_detections + mediapipe_detections

        return all_detections