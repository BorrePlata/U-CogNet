"""
Detector de visión para U-CogNet.
Versión real: Implementa YOLOv8 para detección de objetos.
"""

import numpy as np
import time
from typing import List, Optional
from ultralytics import YOLO
from ..common.types import Detection
from ..common.logging import logger

class VisionDetector:
    """
    Detector de visión computacional usando YOLOv8.
    Carga modelo pre-entrenado y realiza inferencia en tiempo real.
    """

    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_path = model_path
        self.classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

        # Añadir clases militares para SEMAR
        self.military_classes = ['tank', 'truck', 'car', 'person', 'airplane']
        self.classes.extend(self.military_classes)

        self._load_model()
        logger.info(f"VisionDetector YOLOv8 inicializado con threshold {confidence_threshold}")

    def _load_model(self):
        """Carga el modelo YOLOv8."""
        try:
            logger.info(f"Cargando modelo YOLOv8: {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("Modelo YOLOv8 cargado exitosamente")
        except Exception as e:
            logger.error(f"Error cargando modelo YOLOv8: {e}")
            # Fallback a simulación si no hay modelo
            logger.warning("Usando modo simulado por falta de modelo")
            self.model = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detecta objetos en el frame usando YOLOv8.
        Si no hay modelo, usa simulación.
        """
        if self.model is None:
            return self._simulate_detection(frame)

        try:
            # Realizar inferencia con YOLOv8
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extraer coordenadas y clase
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())

                        class_name = self.classes[cls] if cls < len(self.classes) else f"class_{cls}"

                        detection = Detection(
                            class_name=class_name,
                            confidence=float(conf),
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            timestamp=time.time()
                        )
                        detections.append(detection)

            logger.debug(f"YOLOv8 detectó {len(detections)} objetos")
            return detections

        except Exception as e:
            logger.error(f"Error en inferencia YOLOv8: {e}")
            return self._simulate_detection(frame)

    def _simulate_detection(self, frame: np.ndarray) -> List[Detection]:
        """
        Fallback: simula detecciones cuando no hay modelo real.
        """
        detections = []

        # Simular detecciones aleatorias (placeholder)
        num_detections = np.random.randint(0, 5)  # 0-4 detecciones

        for _ in range(num_detections):
            class_name = np.random.choice(self.classes[:10])  # Solo clases comunes
            confidence = np.random.uniform(0.3, 0.9)

            if confidence >= self.confidence_threshold:
                # Bbox aleatoria
                x1 = np.random.uniform(0, frame.shape[1] * 0.8)
                y1 = np.random.uniform(0, frame.shape[0] * 0.8)
                x2 = x1 + np.random.uniform(20, frame.shape[1] * 0.3)
                y2 = y1 + np.random.uniform(20, frame.shape[0] * 0.3)

                detection = Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    timestamp=time.time()
                )
                detections.append(detection)

        logger.debug(f"Simulación detectó {len(detections)} objetos")
        return detections

    def detect_military_targets(self, frame: np.ndarray) -> List[Detection]:
        """
        Detecta específicamente objetivos militares/tácticos.
        """
        all_detections = self.detect(frame)
        military_detections = [
            d for d in all_detections
            if d.class_name in self.military_classes or d.class_name in ['truck', 'car', 'person']
        ]
        return military_detections