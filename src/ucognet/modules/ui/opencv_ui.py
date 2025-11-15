import cv2
import numpy as np
from ucognet.core.interfaces import VisualInterface
from ucognet.core.types import Frame, Detection, SystemState

class OpenCVVisualInterface(VisualInterface):
    """VisualInterface que muestra video con detecciones usando OpenCV."""

    def __init__(self, window_name="U-CogNet Vision"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def render(self, frame: Frame, detections: list[Detection], text: str, state: SystemState) -> None:
        # Copiar el frame para no modificar el original
        display_frame = frame.data.copy()

        # Dibujar bounding boxes para cada detección
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            # Color: rojo para armas, verde para personas, azul para otros
            if hasattr(det, 'is_weapon') and det.is_weapon:
                color = (0, 0, 255)  # Rojo para armas
                thickness = 3  # Más grueso para armas
            elif det.class_name == "person":
                color = (0, 255, 0)  # Verde para personas
                thickness = 2
            else:
                color = (255, 0, 0)  # Azul para otros objetos
                thickness = 2
                
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

            # Etiqueta con clase y confianza
            label = f"{det.class_name} {det.confidence:.2f}"
            if hasattr(det, 'is_weapon') and det.is_weapon:
                label += " ⚠️ WEAPON"
            cv2.putText(display_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Agregar texto de feedback semántico
        cv2.putText(display_frame, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Agregar información del sistema
        if state.metrics:
            metrics_text = f"F1: {state.metrics.f1:.2f} | Precision: {state.metrics.precision:.2f}"
            cv2.putText(display_frame, metrics_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Mostrar el frame
        cv2.imshow(self.window_name, display_frame)

        # Esperar 1ms y verificar si se presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Usuario cerró la ventana")

    def close(self):
        """Cerrar la ventana."""
        cv2.destroyAllWindows()