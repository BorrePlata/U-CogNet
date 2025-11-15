import cv2
import numpy as np
from ucognet.core.interfaces import VisualInterface
from ucognet.core.types import Frame, Detection, SystemState

class OpenCVVisualInterface(VisualInterface):
    """VisualInterface que muestra video con detecciones usando OpenCV."""

    def __init__(self, window_name="U-CogNet Vision", record_on_crowd=True, record_duration=60):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Grabación automática
        self.record_on_crowd = record_on_crowd
        self.record_duration = record_duration  # segundos (aumentado a 60)
        self.recording = False
        self.record_start_time = None
        self.video_writer = None
        self.fps = 30  # asumimos 30 fps

    def render(self, frame: Frame, detections: list[Detection], text: str, state: SystemState) -> None:
        # Copiar el frame para no modificar el original
        display_frame = frame.data.copy()

        # Dibujar bounding boxes para cada detección
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            
            # Sistema de colores mejorado para armas y personas armadas
            if hasattr(det, 'is_weapon') and det.is_weapon:
                color = (0, 0, 255)  # Rojo brillante para armas
                thickness = 4  # Muy grueso para armas
                # Dibujar contorno adicional para mayor visibilidad
                cv2.rectangle(display_frame, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), 1)
            elif det.class_name == "person":
                # Verificar si la persona está armada
                if hasattr(det, 'is_armed') and det.is_armed:
                    color = (0, 0, 255)  # Rojo para personas armadas
                    thickness = 4  # Muy grueso para personas armadas
                    # Dibujar contorno adicional y patrón de alerta
                    cv2.rectangle(display_frame, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), 1)
                    # Patrón de alerta: líneas diagonales
                    for i in range(0, max(x2-x1, y2-y1), 10):
                        cv2.line(display_frame, (x1+i, y1), (x1, y1+i), (0, 0, 255), 1)
                        cv2.line(display_frame, (x2-i, y1), (x2, y1+i), (0, 0, 255), 1)
                else:
                    color = (0, 255, 0)  # Verde para personas normales
                    thickness = 2
            elif det.class_name in ["body_pose", "left_hand", "right_hand", "face"]:
                color = (255, 0, 255)  # Magenta para detecciones MediaPipe
                thickness = 2
            else:
                color = (255, 0, 0)  # Azul para otros objetos
                thickness = 2
                
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

            # Etiqueta mejorada con más información
            if hasattr(det, 'is_weapon') and det.is_weapon:
                label = f"⚠️ WEAPON: {det.class_name.upper()} ({det.confidence:.2f})"
                # Fondo negro para mejor legibilidad
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(display_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)
                cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            elif det.class_name == "person" and hasattr(det, 'is_armed') and det.is_armed:
                weapon_name = getattr(det, 'nearby_weapon', 'arma desconocida')
                distance = getattr(det, 'weapon_distance', 0)
                label = f"🚨 ARMADO: {weapon_name.upper()} (dist: {distance:.0f}px)"
                # Fondo negro para mejor legibilidad
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(display_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)
                cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                label = f"{det.class_name} {det.confidence:.2f}"
                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Dibujar landmarks de MediaPipe si hay detecciones
        mediapipe_detections = [d for d in detections if hasattr(d, 'pose_landmarks') or 
                               hasattr(d, 'hand_landmarks') or hasattr(d, 'face_landmarks')]
        if mediapipe_detections:
            # Crear instancia temporal de MediaPipe detector para dibujar
            from ..vision.mediapipe_detector import MediaPipeDetector
            temp_detector = MediaPipeDetector()
            display_frame = temp_detector.draw_landmarks(display_frame, mediapipe_detections)

        # Agregar texto de feedback semántico
        cv2.putText(display_frame, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Agregar información del sistema
        if state.metrics:
            metrics_text = f"F1: {state.metrics.f1:.2f} | Precision: {state.metrics.precision:.2f}"
            cv2.putText(display_frame, metrics_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Agregar estadísticas de detección de armas en esquina superior derecha
        weapon_count = sum(1 for det in detections if hasattr(det, 'is_weapon') and det.is_weapon)
        armed_person_count = sum(1 for det in detections if det.class_name == "person" and hasattr(det, 'is_armed') and det.is_armed)
        person_count = sum(1 for det in detections if det.class_name == "person")
        
        # Panel de estadísticas de seguridad
        stats_x = display_frame.shape[1] - 300
        stats_y = 30
        
        # Fondo semi-transparente para el panel
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (stats_x - 10, stats_y - 25), (display_frame.shape[1] - 10, stats_y + 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # Título del panel
        cv2.putText(display_frame, "SECURITY STATUS", (stats_x, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Estadísticas
        cv2.putText(display_frame, f"Personas: {person_count}", (stats_x, stats_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Armas detectadas: {weapon_count}", (stats_x, stats_y + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(display_frame, f"Personas armadas: {armed_person_count}", (stats_x, stats_y + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Alerta visual si hay armas o personas armadas
        if weapon_count > 0 or armed_person_count > 0:
            # Marco rojo intermitente en los bordes
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 0, 255), 3)
            cv2.putText(display_frame, "🚨 ALERTA DE SEGURIDAD 🚨", (stats_x, stats_y + 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Contar personas para grabación automática
        person_count = sum(1 for det in detections if det.class_name == "person")
        
        # Iniciar grabación si hay muchas personas y no estamos grabando
        if self.record_on_crowd and person_count >= 1 and not self.recording:
            self._start_recording(frame)
        
        # Detener grabación solo si no hay personas por más de 10 segundos
        if self.recording:
            time_since_last_person = cv2.getTickCount() / cv2.getTickFrequency() - self.record_start_time
            # Si han pasado más de 10 segundos sin personas, detener grabación
            if person_count == 0 and time_since_last_person > 10:
                self._stop_recording()
            # Si han pasado más de 120 segundos totales, detener grabación (máximo 2 minutos)
            elif time_since_last_person > 120:
                self._stop_recording()
        
        # Agregar contador de personas y estado de grabación
        crowd_text = f"Personas: {person_count}"
        if self.recording:
            elapsed = cv2.getTickCount() / cv2.getTickFrequency() - self.record_start_time
            crowd_text += f" | REC {elapsed:.1f}s (auto-stop: 2min o 10s sin personas)"
            # Marco rojo cuando graba
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 0, 255), 5)
        
        cv2.putText(display_frame, crowd_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Grabar frame si está activo
        if self.recording and self.video_writer:
            self.video_writer.write(display_frame)

        # Mostrar el frame
        cv2.imshow(self.window_name, display_frame)

        # Esperar 1ms y verificar si se presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self._stop_recording()  # Asegurar que se guarde si estaba grabando
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Usuario cerró la ventana")

    def _start_recording(self, frame: Frame):
        """Iniciar grabación de video."""
        import time
        timestamp = int(time.time())
        filename = f"demo_clip_{timestamp}.avi"
        
        # Probar diferentes codecs en orden de preferencia
        codecs_to_try = ['XVID', 'DIVX', 'MJPG', 'mp4v']
        fourcc = None
        
        for codec in codecs_to_try:
            try:
                test_fourcc = cv2.VideoWriter_fourcc(*codec)
                # Intentar crear un VideoWriter temporal para verificar si el codec funciona
                temp_writer = cv2.VideoWriter(f"test_{codec}.avi", test_fourcc, self.fps, (640, 480))
                if temp_writer.isOpened():
                    temp_writer.release()
                    import os
                    os.remove(f"test_{codec}.avi")  # Limpiar archivo de prueba
                    fourcc = test_fourcc
                    print(f"✅ Codec {codec} disponible")
                    break
                else:
                    temp_writer.release()
                    try:
                        os.remove(f"test_{codec}.avi")
                    except:
                        pass
            except:
                continue
        
        if fourcc is None:
            print("⚠️ No se encontró codec compatible, usando MJPG por defecto")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        
        height, width = frame.data.shape[:2]
        self.video_writer = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))
        
        if not self.video_writer.isOpened():
            print("❌ Error: No se pudo inicializar el VideoWriter")
            return
        
        self.recording = True
        self.record_start_time = cv2.getTickCount() / cv2.getTickFrequency()
        print(f"🎬 Iniciando grabación automática: {filename} (hasta 2 min o hasta que no haya personas por 10s)")

    def _stop_recording(self):
        """Detener grabación de video."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        if self.recording:
            self.recording = False
            print("✅ Grabación completada y guardada")

    def close(self):
        """Cerrar la ventana y liberar recursos."""
        self._stop_recording()
        cv2.destroyAllWindows()