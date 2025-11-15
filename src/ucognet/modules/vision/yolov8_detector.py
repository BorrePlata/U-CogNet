from ultralytics import YOLO
from ucognet.core.interfaces import VisionDetector
from ucognet.core.types import Frame, Detection
import numpy as np

# MediaPipe imports (comentados para ahorrar recursos)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
        
        # Clases relacionadas con armas/peligro (solo armas reales del dataset COCO)
        self.weapon_classes = {
            # Armas reales
            34: 'baseball bat',    # Bate de béisbol
            43: 'knife',           # Cuchillo
            76: 'scissors',        # Tijeras
            
            # Objetos que pueden usarse como armas improvisadas
            39: 'bottle',          # Botella
            42: 'fork',            # Tenedor
        }
        
        # Configuración mejorada para detección de armas
        self.weapon_detection_config = {
            'min_confidence': 0.3,  # Umbral más bajo para armas (más sensibles)
            'proximity_threshold': 100,  # Distancia máxima en píxeles para considerar "armado"
            'size_ratio_threshold': 0.5,  # Relación de tamaño para validar proximidad
        }
        
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
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        pass

    def detect(self, frame: Frame) -> list[Detection]:
        # Ejecutar inferencia YOLOv8 con umbral adaptativo
        # Usar umbral más bajo para armas para mayor sensibilidad
        weapon_classes_list = list(self.weapon_classes.keys())
        
        # Primera pasada: detectar todo con umbral normal
        results = self.model(frame.data, conf=self.conf_threshold, classes=self.classes, device='cuda' if self.model.device.type == 'cuda' else 'cpu')
        
        detections = []
        weapon_detections = []
        
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
                
                # Marcar si es un arma con umbral más bajo
                if cls in self.weapon_classes and conf > self.weapon_detection_config['min_confidence']:
                    detection.is_weapon = True
                    weapon_detections.append(detection)
                else:
                    detection.is_weapon = False
                    
                detections.append(detection)
        
        # Segunda pasada específica para armas con umbral más bajo (si no se detectaron suficientes)
        if len(weapon_detections) == 0:
            weapon_results = self.model(frame.data, conf=self.weapon_detection_config['min_confidence'], 
                                      classes=weapon_classes_list, device='cuda' if self.model.device.type == 'cuda' else 'cpu')
            
            for result in weapon_results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[cls]
                    
                    # Verificar que no duplicamos detecciones
                    duplicate = any(d.class_id == cls and 
                                  abs(d.bbox[0] - float(x1)) < 10 and 
                                  abs(d.bbox[1] - float(y1)) < 10 
                                  for d in detections)
                    
                    if not duplicate:
                        detection = Detection(
                            class_id=cls,
                            class_name=class_name,
                            confidence=float(conf),
                            bbox=[float(x1), float(y1), float(x2), float(y2)]
                        )
                        detection.is_weapon = True
                        detections.append(detection)
                        weapon_detections.append(detection)
        
        # Detectar personas armadas con lógica mejorada
        self._mark_armed_persons(detections)
        
        # MediaPipe processing (desactivado por defecto)
        if self.use_mediapipe and self.mp_pose:
            detections.extend(self._detect_pose(frame))
        
        return detections

    def _mark_armed_persons(self, detections: list[Detection]):
        """Marcar personas que están cerca de armas como 'armadas' con lógica mejorada."""
        # Separar personas y armas
        persons = [d for d in detections if d.class_name == 'person']
        weapons = [d for d in detections if hasattr(d, 'is_weapon') and d.is_weapon and d.confidence > self.weapon_detection_config['min_confidence']]
        
        # Para cada persona, verificar si está cerca de un arma
        for person in persons:
            person.is_armed = False
            person.nearby_weapon = None
            person.weapon_distance = float('inf')
            
            person_center_x = (person.bbox[0] + person.bbox[2]) / 2
            person_center_y = (person.bbox[1] + person.bbox[3]) / 2
            person_width = person.bbox[2] - person.bbox[0]
            person_height = person.bbox[3] - person.bbox[1]
            
            for weapon in weapons:
                weapon_center_x = (weapon.bbox[0] + weapon.bbox[2]) / 2
                weapon_center_y = (weapon.bbox[1] + weapon.bbox[3]) / 2
                weapon_width = weapon.bbox[2] - weapon.bbox[0]
                weapon_height = weapon.bbox[3] - weapon.bbox[1]
                
                # Calcular distancia euclidiana entre centros
                distance = np.sqrt((person_center_x - weapon_center_x)**2 + (person_center_y - weapon_center_y)**2)
                
                # Calcular si el arma está en la zona de "mano" de la persona
                # Considerar que las manos están aproximadamente en los costados del torso
                hand_zone_left = person_center_x - person_width * 0.3
                hand_zone_right = person_center_x + person_width * 0.3
                hand_zone_top = person_center_y - person_height * 0.2
                hand_zone_bottom = person_center_y + person_height * 0.3
                
                weapon_in_hand_zone = (hand_zone_left <= weapon_center_x <= hand_zone_right and
                                     hand_zone_top <= weapon_center_y <= hand_zone_bottom)
                
                # Calcular relación de tamaño (arma no debe ser mucho más grande que la mano)
                size_ratio = max(weapon_width, weapon_height) / max(person_width, person_height)
                reasonable_size = size_ratio < self.weapon_detection_config['size_ratio_threshold']
                
                # Lógica mejorada: considerar distancia, zona de mano y tamaño razonable
                if (distance < self.weapon_detection_config['proximity_threshold'] and 
                    (weapon_in_hand_zone or distance < person_width * 0.5) and 
                    reasonable_size):
                    
                    # Si esta arma está más cerca que la anterior, actualizar
                    if distance < person.weapon_distance:
                        person.is_armed = True
                        person.nearby_weapon = weapon.class_name
                        person.weapon_distance = distance
                        person.weapon_confidence = weapon.confidence

    def _detect_pose(self, frame: Frame) -> list[Detection]:
        """Detectar pose corporal con MediaPipe (desactivado)."""
        # Descomentear cuando se active MediaPipe:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.data)
        pose_result = self.mp_pose.process(mp_image)

        pose_detections = []
        if pose_result.pose_landmarks:
            # Convertir landmarks a detecciones
            # Usar las dimensiones del frame original (NumPy array)
            height, width = frame.data.shape[:2]
            detection = Detection(
                class_id=1000,  # ID especial para pose
                class_name="pose",
                confidence=0.9,
                bbox=[0, 0, width, height]  # Toda la imagen
            )
            detection.pose_landmarks = pose_result.pose_landmarks
            pose_detections.append(detection)

        return pose_detections