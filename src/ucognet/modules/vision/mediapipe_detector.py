import mediapipe as mp
import cv2
import numpy as np
from ucognet.core.interfaces import VisionDetector
from ucognet.core.types import Frame, Detection

class MediaPipeDetector(VisionDetector):
    """Detector que combina MediaPipe para poses, manos y rostro."""

    def __init__(self):
        # Inicializar MediaPipe Holistic (pose + hands + face)
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def detect(self, frame: Frame) -> list[Detection]:
        """Detecta poses, manos y rostro usando MediaPipe."""
        detections = []

        # Convertir BGR a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame.data, cv2.COLOR_BGR2RGB)

        # Procesar con MediaPipe
        results = self.holistic.process(rgb_frame)

        # Procesar pose (cuerpo)
        if results.pose_landmarks:
            # Crear detección para pose completa
            pose_detection = Detection(
                class_id=1000,  # ID especial para pose
                class_name="body_pose",
                confidence=0.9,  # MediaPipe da alta confianza
                bbox=self._get_pose_bbox(results.pose_landmarks, frame.data.shape)
            )
            pose_detection.pose_landmarks = results.pose_landmarks
            detections.append(pose_detection)

        # Procesar manos izquierda
        if results.left_hand_landmarks:
            hand_detection = Detection(
                class_id=1001,
                class_name="left_hand",
                confidence=0.9,
                bbox=self._get_hand_bbox(results.left_hand_landmarks, frame.data.shape)
            )
            hand_detection.hand_landmarks = results.left_hand_landmarks
            detections.append(hand_detection)

        # Procesar manos derecha
        if results.right_hand_landmarks:
            hand_detection = Detection(
                class_id=1002,
                class_name="right_hand",
                confidence=0.9,
                bbox=self._get_hand_bbox(results.right_hand_landmarks, frame.data.shape)
            )
            hand_detection.hand_landmarks = results.right_hand_landmarks
            detections.append(hand_detection)

        # Procesar rostro
        if results.face_landmarks:
            face_detection = Detection(
                class_id=1003,
                class_name="face",
                confidence=0.9,
                bbox=self._get_face_bbox(results.face_landmarks, frame.data.shape)
            )
            face_detection.face_landmarks = results.face_landmarks
            detections.append(face_detection)

        return detections

    def _get_pose_bbox(self, landmarks, image_shape):
        """Calcula bounding box para pose completa."""
        h, w = image_shape[:2]
        x_coords = [lm.x * w for lm in landmarks.landmark]
        y_coords = [lm.y * h for lm in landmarks.landmark]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Agregar padding
        padding = 0.1
        x_min = max(0, x_min - (x_max - x_min) * padding)
        x_max = min(w, x_max + (x_max - x_min) * padding)
        y_min = max(0, y_min - (y_max - y_min) * padding)
        y_max = min(h, y_max + (y_max - y_min) * padding)

        return [x_min, y_min, x_max, y_max]

    def _get_hand_bbox(self, landmarks, image_shape):
        """Calcula bounding box para mano."""
        h, w = image_shape[:2]
        x_coords = [lm.x * w for lm in landmarks.landmark]
        y_coords = [lm.y * h for lm in landmarks.landmark]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Padding para mano
        padding = 0.2
        width = x_max - x_min
        height = y_max - y_min
        x_min = max(0, x_min - width * padding)
        x_max = min(w, x_max + width * padding)
        y_min = max(0, y_min - height * padding)
        y_max = min(h, y_max + height * padding)

        return [x_min, y_min, x_max, y_max]

    def _get_face_bbox(self, landmarks, image_shape):
        """Calcula bounding box para rostro."""
        h, w = image_shape[:2]
        x_coords = [lm.x * w for lm in landmarks.landmark]
        y_coords = [lm.y * h for lm in landmarks.landmark]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Padding para rostro
        padding = 0.1
        width = x_max - x_min
        height = y_max - y_min
        x_min = max(0, x_min - width * padding)
        x_max = min(w, x_max + width * padding)
        y_min = max(0, y_min - height * padding)
        y_max = min(h, y_max + height * padding)

        return [x_min, y_min, x_max, y_max]

    def draw_landmarks(self, frame, detections):
        """Dibuja landmarks de MediaPipe en el frame (para debugging)."""
        # Convertir de vuelta a RGB para MediaPipe drawing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Recrear results object para drawing
        class MockResults:
            def __init__(self, detections):
                self.pose_landmarks = None
                self.left_hand_landmarks = None
                self.right_hand_landmarks = None
                self.face_landmarks = None

                for det in detections:
                    if hasattr(det, 'pose_landmarks'):
                        self.pose_landmarks = det.pose_landmarks
                    elif hasattr(det, 'hand_landmarks') and det.class_name == "left_hand":
                        self.left_hand_landmarks = det.hand_landmarks
                    elif hasattr(det, 'hand_landmarks') and det.class_name == "right_hand":
                        self.right_hand_landmarks = det.hand_landmarks
                    elif hasattr(det, 'face_landmarks'):
                        self.face_landmarks = det.face_landmarks

        mock_results = MockResults(detections)

        # Dibujar landmarks
        self.mp_drawing.draw_landmarks(
            rgb_frame, mock_results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

        self.mp_drawing.draw_landmarks(
            rgb_frame, mock_results.left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS
        )

        self.mp_drawing.draw_landmarks(
            rgb_frame, mock_results.right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS
        )

        self.mp_drawing.draw_landmarks(
            rgb_frame, mock_results.face_landmarks,
            self.mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
        )

        # Convertir de vuelta a BGR
        return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)