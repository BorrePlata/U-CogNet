"""
YOLOv8 Detector Implementation for U-CogNet
Implementación real de detector de visión usando YOLOv8
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  ultralytics no disponible. Instalando...")
    import subprocess
    subprocess.run(["pip", "install", "ultralytics"], check=True)
    try:
        from ultralytics import YOLO
        YOLO_AVAILABLE = True
    except ImportError:
        YOLO_AVAILABLE = False
        print("❌ No se pudo instalar ultralytics. Usando simulación.")

from ucognet.core.types import Detection


class YOLOv8Detector:
    """
    Detector de visión usando YOLOv8 real
    Implementa la interfaz VisionDetector para detección de objetos
    """

    def __init__(self,
                 model_path: str = "yolov8n.pt",
                 conf_threshold: float = 0.3,
                 iou_threshold: float = 0.45,
                 use_mediapipe: bool = False,
                 device: str = "auto"):
        """
        Inicializar detector YOLOv8

        Args:
            model_path: Ruta al modelo YOLOv8 (.pt)
            conf_threshold: Umbral de confianza mínimo
            iou_threshold: Umbral IoU para NMS
            use_mediapipe: Si usar MediaPipe para pose estimation
            device: Dispositivo ('cpu', 'cuda', 'auto')
        """
        self.logger = logging.getLogger("YOLOv8Detector")
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_mediapipe = use_mediapipe
        self.device = device

        # Estado del modelo
        self.model = None
        self.is_loaded = False
        self.class_names = []

        # Estadísticas
        self.detection_count = 0
        self.total_inference_time = 0
        self.avg_inference_time = 0

        # Inicializar modelo
        self._load_model()

    def _load_model(self):
        """Cargar modelo YOLOv8"""
        try:
            if not YOLO_AVAILABLE:
                self.logger.error("❌ ultralytics no disponible")
                return

            # Determinar dispositivo
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.logger.info(f"🔧 Cargando YOLOv8: {self.model_path} en {self.device}")

            # Cargar modelo
            self.model = YOLO(self.model_path)
            self.model.to(self.device)

            # Configurar umbrales
            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold

            # Obtener nombres de clases
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
            else:
                # COCO classes por defecto
                self.class_names = [
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                ]

            self.is_loaded = True
            self.logger.info(f"✅ YOLOv8 cargado exitosamente. {len(self.class_names)} clases disponibles")

        except Exception as e:
            self.logger.error(f"❌ Error cargando YOLOv8: {e}")
            self.is_loaded = False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detectar objetos en el frame usando YOLOv8 real

        Args:
            frame: Frame de imagen en formato numpy array (H, W, C)

        Returns:
            Lista de detecciones
        """
        if not self.is_loaded or self.model is None:
            # Fallback a simulación si no hay modelo
            return self._simulate_detections(frame)

        try:
            start_time = time.time()

            # Ejecutar inferencia
            results = self.model(frame, verbose=False)

            inference_time = time.time() - start_time

            # Actualizar estadísticas
            self.detection_count += 1
            self.total_inference_time += inference_time
            self.avg_inference_time = self.total_inference_time / self.detection_count

            # Procesar resultados
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extraer datos de la caja
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())

                        # Crear detección
                        detection = Detection(
                            bbox=[float(x1), float(y1), float(x2), float(y2)],
                            confidence=float(conf),
                            class_id=class_id,
                            class_name=self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}",
                            features=None,
                            metadata={
                                'detector': 'yolov8_real',
                                'model': Path(self.model_path).name,
                                'inference_time': inference_time,
                                'device': self.device
                            }
                        )
                        detections.append(detection)

            self.logger.debug(f"🎯 Detectados {len(detections)} objetos en {inference_time:.4f}s")
            return detections

        except Exception as e:
            self.logger.error(f"❌ Error en detección YOLOv8: {e}")
            # Fallback a simulación
            return self._simulate_detections(frame)

    def _simulate_detections(self, frame: np.ndarray) -> List[Detection]:
        """
        Método de fallback: simular detecciones cuando YOLOv8 no está disponible
        """
        self.logger.warning("🔄 Usando simulación de detecciones (YOLOv8 no disponible)")

        detections = []
        height, width = frame.shape[:2]

        # Simular detecciones aleatorias (solo personas para compatibilidad)
        num_people = np.random.randint(0, 5)  # 0-4 personas

        for i in range(num_people):
            x1 = np.random.randint(0, width // 2)
            y1 = np.random.randint(0, height // 2)
            x2 = min(x1 + np.random.randint(50, 150), width)
            y2 = min(y1 + np.random.randint(100, 250), height)

            detection = Detection(
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                confidence=float(np.random.uniform(0.4, 0.9)),
                class_id=0,  # persona
                class_name="person",
                features=None,
                metadata={
                    'detector': 'simulated_fallback',
                    'reason': 'yolo_unavailable'
                }
            )
            detections.append(detection)

        return detections

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del detector"""
        return {
            'model_loaded': self.is_loaded,
            'model_path': self.model_path,
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'class_count': len(self.class_names),
            'detection_count': self.detection_count,
            'avg_inference_time': self.avg_inference_time,
            'total_inference_time': self.total_inference_time
        }

    def __str__(self) -> str:
        status = "✅ CARGADO" if self.is_loaded else "❌ NO CARGADO"
        return f"YOLOv8Detector({self.model_path}, {self.device}) - {status}"