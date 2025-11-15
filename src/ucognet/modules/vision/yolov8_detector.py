from ultralytics import YOLO
from ucognet.core.interfaces import VisionDetector
from ucognet.core.types import Frame, Detection
import numpy as np

class YOLOv8Detector(VisionDetector):
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5, classes=None):
        """
        Inicializar detector YOLOv8.
        - model_path: Path al modelo (descarga automática si no existe).
        - conf_threshold: Umbral de confianza.
        - classes: Lista de clases a detectar (None para todas).
        """
        self.model = YOLO(model_path)  # Descarga automática si no existe
        self.conf_threshold = conf_threshold
        self.classes = classes
        
        # Clases relacionadas con armas/peligro
        self.weapon_classes = {43: 'knife', 76: 'scissors', 34: 'baseball bat'}

    def detect(self, frame: Frame) -> list[Detection]:
        # Ejecutar inferencia
        results = self.model(frame.data, conf=self.conf_threshold, classes=self.classes, device='cuda' if self.model.device.type == 'cuda' else 'cpu')
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[cls]
                
                detection = Detection(
                    class_id=cls,
                    class_name=class_name,
                    confidence=float(conf),
                    bbox=[float(x1), float(y1), float(x2), float(y2)]
                )
                
                # Marcar si es un arma (agregamos atributo dinámicamente)
                if cls in self.weapon_classes:
                    detection.is_weapon = True
                else:
                    detection.is_weapon = False
                    
                detections.append(detection)
        
        return detections