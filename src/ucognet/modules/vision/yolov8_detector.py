from ultralytics import YOLO
from ucognet.core.interfaces import VisionDetector
from ucognet.core.types import Frame, Detection
import numpy as np

# MediaPipe imports (comentados para ahorrar recursos)
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

class YOLOv8Detector(VisionDetector):
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5, classes=None, use_mediapipe=False):
        """
        Inicializar detector YOLOv8 con opción modular para MediaPipe.
        - model_path: Path al modelo (descarga automática si no existe).
        - conf_threshold: Umbral de confianza.
        - classes: Lista de clases a detectar (None para todas).
        - use_mediapipe: Activar/desactivar MediaPipe para pose/rostro/manos.
        """
        self.model = YOLO(model_path)  # Descarga automática si no existe
        self.conf_threshold = conf_threshold
        self.classes = classes
        self.use_mediapipe = use_mediapipe
        
        # Clases relacionadas con armas/peligro
        self.weapon_classes = {43: 'knife', 76: 'scissors', 34: 'baseball bat'}
        
        # MediaPipe initialization (desactivado por defecto para ahorrar recursos)
        if self.use_mediapipe:
            self._init_mediapipe()
        else:
            self.mp_pose = None
            self.mp_face = None
            self.mp_hands = None

    def _init_mediapipe(self):
        """Inicializar MediaPipe (solo cuando se activa)."""
        # Descomentear cuando se quiera usar MediaPipe:
        # self.mp_pose = mp.solutions.pose.Pose(
        #     static_image_mode=False,
        #     model_complexity=1,
        #     enable_segmentation=False,
        #     min_detection_confidence=0.5
        # )
        # self.mp_face = mp.solutions.face_mesh.FaceMesh(
        #     static_image_mode=False,
        #     max_num_faces=1,
        #     min_detection_confidence=0.5
        # )
        # self.mp_hands = mp.solutions.hands.Hands(
        #     static_image_mode=False,
        #     max_num_hands=2,
        #     min_detection_confidence=0.5
        # )
        pass

    def detect(self, frame: Frame) -> list[Detection]:
        # Ejecutar inferencia YOLOv8
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
        
        # MediaPipe processing (desactivado por defecto)
        if self.use_mediapipe and self.mp_pose:
            detections.extend(self._detect_pose(frame))
        
        return detections

    def _detect_pose(self, frame: Frame) -> list[Detection]:
        """Detectar pose corporal con MediaPipe (desactivado)."""
        # Descomentear cuando se active MediaPipe:
        # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.data)
        # pose_result = self.mp_pose.process(mp_image)
        # 
        # pose_detections = []
        # if pose_result.pose_landmarks:
        #     # Convertir landmarks a detecciones
        #     detection = Detection(
        #         class_id=1000,  # ID especial para pose
        #         class_name="pose",
        #         confidence=0.9,
        #         bbox=[0, 0, frame.data.shape[1], frame.data.shape[0]]  # Toda la imagen
        #     )
        #     detection.pose_landmarks = pose_result.pose_landmarks
        #     pose_detections.append(detection)
        # 
        # return pose_detections
        return []