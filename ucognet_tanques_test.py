#!/usr/bin/env python3
"""
U-CogNet Video Test: Tanques
============================

Prueba completa del sistema U-CogNet con video de tanques usando YOLOv8 únicamente
(MediaPipe desactivado para optimización).
"""

import sys
import os
import cv2
import time
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ucognet.modules.vision.yolov8_detector import YOLOv8Detector
from ucognet.modules.semantic.advanced_feedback import AdvancedSemanticFeedback
from ucognet.core.types import Context, Detection, Frame

class UCogNetVideoTester:
    """Sistema de prueba para U-CogNet con video de tanques."""

    # Clases relevantes para video de tanques (filtrar falsos positivos)
    RELEVANT_CLASSES = {
        'person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle',
        'tank', 'armored_vehicle', 'military_vehicle', 'vehicle'
    }

    # Clases a excluir explícitamente (falsos positivos comunes)
    EXCLUDED_CLASSES = {
        'airplane', 'train', 'boat', 'ship', 'cup', 'bottle', 'chair',
        'dining table', 'potted plant', 'tv', 'laptop', 'mouse', 'keyboard',
        'cell phone', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'couch', 'bed', 'toilet', 'sink', 'refrigerator', 'oven', 'microwave'
    }

    def __init__(self, video_path: str, output_dir: str = "tanques_test_results"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components (MediaPipe disabled)
        print("🚀 Inicializando U-CogNet (YOLOv8 only)...")
        self.vision_detector = YOLOv8Detector(use_mediapipe=False)  # MediaPipe desactivado
        self.semantic_feedback = AdvancedSemanticFeedback()

        # Mock context for semantic analysis
        self.context = Context(recent_events=[], episodic_memory=[])

        # Video processing variables
        self.cap = None
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_stats = []

        # Results tracking
        self.detection_stats = {
            'total_frames': 0,
            'frames_with_detections': 0,
            'total_detections': 0,
            'threat_levels': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0},
            'object_counts': {}
        }

    def initialize_video(self):
        """Initialize video capture."""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video no encontrado: {self.video_path}")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {self.video_path}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"📹 Video cargado: {self.width}x{self.height} @ {self.fps:.1f} FPS")
        print(f"📊 Total frames: {self.total_frames}")

    def create_frame(self, cv_frame) -> Frame:
        """Convert OpenCV frame to U-CogNet Frame."""
        # Convert BGR to RGB for consistency
        rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)

        return Frame(
            data=rgb_frame,
            timestamp=time.time(),
            metadata={
                'source': 'video_tanques.mp4',
                'frame_number': self.frame_count,
                'resolution': f"{self.width}x{self.height}"
            }
        )

    def draw_detections(self, frame, detections: list[Detection]):
        """Draw detections on frame for visualization."""
        display_frame = frame.copy()

        for detection in detections:
            # Extract bbox coordinates
            if isinstance(detection.bbox, list) and len(detection.bbox) == 4:
                x1, y1, x2, y2 = detection.bbox
            else:
                continue

            # Choose color based on class
            if 'tank' in detection.class_name.lower() or 'vehicle' in detection.class_name.lower():
                color = (0, 0, 255)  # Red for military vehicles
            elif detection.class_name in ['person', 'car', 'truck']:
                color = (255, 0, 0)  # Blue for other vehicles/people
            else:
                color = (0, 255, 0)  # Green for other objects

            # Draw bounding box
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(display_frame, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return display_frame

    def filter_relevant_detections(self, detections: list[Detection]) -> list[Detection]:
        """Filter detections to only include relevant classes for tank video analysis."""
        filtered = []
        for detection in detections:
            class_lower = detection.class_name.lower()

            # Excluir explícitamente clases no relevantes
            if detection.class_name in self.EXCLUDED_CLASSES:
                continue

            # Incluir solo clases relevantes o que podrían ser vehículos militares
            if (detection.class_name in self.RELEVANT_CLASSES or
                'tank' in class_lower or
                'vehicle' in class_lower or
                'military' in class_lower or
                'armored' in class_lower):
                filtered.append(detection)
            # Special case: sometimes tanks might be classified as trucks or cars
            elif (detection.class_name in ['truck', 'car'] and
                  detection.confidence > 0.1):  # Lower confidence threshold to catch potential tanks
                # Rename to potential military vehicle for analysis
                detection.class_name = f"potential_vehicle"
                filtered.append(detection)

        return filtered

    def update_stats(self, detections: list[Detection], threat_level: str):
        """Update detection statistics."""
        self.detection_stats['total_frames'] += 1

        if detections:
            self.detection_stats['frames_with_detections'] += 1
            self.detection_stats['total_detections'] += len(detections)

            # Count objects by class
            for detection in detections:
                class_name = detection.class_name
                if class_name not in self.detection_stats['object_counts']:
                    self.detection_stats['object_counts'][class_name] = 0
                self.detection_stats['object_counts'][class_name] += 1

        # Update threat level stats
        if threat_level in self.detection_stats['threat_levels']:
            self.detection_stats['threat_levels'][threat_level] += 1

    def print_progress(self, frame_num: int, fps: float, analysis: str):
        """Print progress information."""
        progress = (frame_num / self.total_frames) * 100
        threat_indicator = ""
        if "CRITICAL" in analysis:
            threat_indicator = "🚨 CRÍTICO"
        elif "HIGH" in analysis:
            threat_indicator = "⚠️  ALTO"
        elif "MEDIUM" in analysis:
            threat_indicator = "⚡ MEDIO"

        print("2d"
              "5.1f"
              "30s")

    def save_results(self):
        """Save test results to file."""
        results_file = self.output_dir / "tanques_test_results.txt"

        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("U-CogNet Video Test Results: Tanques\n")
            f.write("=" * 40 + "\n\n")

            f.write("📊 Estadísticas Generales:\n")
            f.write(f"  Total frames procesados: {self.detection_stats['total_frames']}\n")
            f.write(f"  Frames con detecciones: {self.detection_stats['frames_with_detections']}\n")
            f.write(f"  Total detecciones: {self.detection_stats['total_detections']}\n")
            f.write(f"  Tiempo total: {time.time() - self.start_time:.1f}s\n")

            # Coverage
            coverage = (self.detection_stats['frames_with_detections'] / max(1, self.detection_stats['total_frames'])) * 100
            f.write(f"  Cobertura de detección: {coverage:.1f}%\n")

            f.write("\n🎯 Niveles de Amenaza:\n")
            for level, count in self.detection_stats['threat_levels'].items():
                percentage = (count / max(1, self.detection_stats['total_frames'])) * 100
                f.write(f"  {level:8s}: {count:3d} frames ({percentage:5.1f}%)\n")
            f.write("\n🔍 Objetos Detectados:\n")
            for obj_class, count in sorted(self.detection_stats['object_counts'].items(),
                                         key=lambda x: x[1], reverse=True):
                f.write(f"  {obj_class:15s}: {count:4d}\n")
            f.write("\n✅ Test completado exitosamente!\n")

        print(f"\n💾 Resultados guardados en: {results_file}")

    def run_test(self, max_frames: int = None, display: bool = True):
        """Run the complete video test."""
        try:
            self.initialize_video()

            print("\n🎬 Iniciando procesamiento de video de tanques...")
            print("   (Presiona 'q' para salir, 'p' para pausar)\n")

            paused = False

            while self.cap.isOpened():
                if not paused:
                    ret, cv_frame = self.cap.read()
                    if not ret:
                        break

                    self.frame_count += 1

                    # Limit frames if specified
                    if max_frames and self.frame_count > max_frames:
                        break

                    # Create U-CogNet frame
                    frame = self.create_frame(cv_frame)

                    # Vision detection (YOLOv8 only)
                    detections_start = time.time()
                    raw_detections = self.vision_detector.detect(frame)
                    # Filter relevant detections for tank video
                    detections = self.filter_relevant_detections(raw_detections)
                    detection_time = time.time() - detections_start

                    # Semantic analysis
                    semantic_start = time.time()
                    analysis = self.semantic_feedback.generate(self.context, detections)
                    semantic_time = time.time() - semantic_start

                    # Calculate FPS
                    total_time = detection_time + semantic_time
                    current_fps = 1.0 / total_time if total_time > 0 else 0
                    self.fps_stats.append(current_fps)

                    # Extract threat level from analysis
                    threat_level = "LOW"
                    if "CRITICAL" in analysis:
                        threat_level = "CRITICAL"
                    elif "HIGH" in analysis:
                        threat_level = "HIGH"
                    elif "MEDIUM" in analysis:
                        threat_level = "MEDIUM"

                    # Update statistics
                    self.update_stats(detections, threat_level)

                    # Print progress
                    self.print_progress(self.frame_count, current_fps, analysis)

                    # Display frame with detections
                    if display:
                        display_frame = self.draw_detections(cv_frame, detections)

                        # Add analysis text overlay
                        y_offset = 30
                        for line in analysis.split(' | '):
                            cv2.putText(display_frame, line, (10, y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(display_frame, line, (10, y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                            y_offset += 25

                        # Add performance info
                        perf_text = f"FPS: {current_fps:.1f} | Frame: {self.frame_count}/{self.total_frames}"
                        cv2.putText(display_frame, perf_text, (10, self.height - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        cv2.imshow('U-CogNet: Video Tanques Test', display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("⏸️  Pausado..." if paused else "▶️  Reanudado...")

            # Cleanup
            self.cap.release()
            if display:
                cv2.destroyAllWindows()

            # Save results
            self.save_results()

            # Print final statistics
            self.print_final_stats()

        except Exception as e:
            print(f"❌ Error durante la prueba: {str(e)}")
            if self.cap:
                self.cap.release()
            if display:
                cv2.destroyAllWindows()
            raise

    def print_final_stats(self):
        """Print final test statistics."""
        total_time = time.time() - self.start_time
        avg_fps = np.mean(self.fps_stats) if self.fps_stats else 0

        print("\n" + "="*60)
        print("🎯 RESULTADOS FINALES - U-CogNet Video Test: Tanques")
        print("="*60)

        print("⏱️  Performance:")
        print(f"  Tiempo total: {total_time:.1f}s")
        print(f"  Frames procesados: {self.detection_stats['total_frames']}")
        print(f"  FPS promedio: {avg_fps:.1f}")

        print("\n📊 Cobertura:")
        coverage = (self.detection_stats['frames_with_detections'] /
                  max(1, self.detection_stats['total_frames'])) * 100
        print(f"  Frames con detecciones: {coverage:.1f}%")

        print("\n🎯 Amenazas Detectadas:")
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            count = self.detection_stats['threat_levels'][level]
            percentage = (count / max(1, self.detection_stats['total_frames'])) * 100
            print(f"  {level:8s}: {count:3d} frames ({percentage:5.1f}%)")

        print("\n🔍 Top Objetos Detectados:")
        sorted_objects = sorted(self.detection_stats['object_counts'].items(),
                             key=lambda x: x[1], reverse=True)[:10]
        for obj_class, count in sorted_objects:
            percentage = (count / max(1, self.detection_stats['total_detections'])) * 100
            print(f"  {obj_class:15s}: {count:4d} ({percentage:5.1f}%)")
        print("\n✅ Test completado exitosamente!")
        print("   U-CogNet demostró capacidades avanzadas de análisis táctico.")


def main():
    """Main function to run the tanques video test."""
    video_path = "video_tanques.mp4"

    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Video no encontrado: {video_path}")
        return

    # Create tester
    tester = UCogNetVideoTester(video_path)

    # Run test (process entire video, set display=True for visualization)
    print("🧪 Iniciando prueba de U-CogNet con video de tanques...")
    print("   - YOLOv8 activado (MediaPipe desactivado)")
    print("   - Análisis semántico avanzado activado")
    print("   - Procesamiento de TODO el video (sin límite de frames)")
    print("   - Filtro activado: solo detecciones relevantes para escenario táctico")

    try:
        tester.run_test(display=True)  # Procesar todo el video sin límite
    except KeyboardInterrupt:
        print("\n⏹️  Test interrumpido por usuario")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
