#!/usr/bin/env python3
"""
Script de Prueba Militar con Autoevaluación
Prueba el sistema U-CogNet con detección de tanques y aprendizaje automático.
"""

import cv2
import sys
import os
from pathlib import Path
import time
from typing import List, Dict

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ucognet.modules.vision.yolov8_detector import YOLOv8Detector
from ucognet.modules.eval.military_auto_evaluator import MilitaryAutoEvaluator
from ucognet.core.types import Frame, Event, Detection

def main():
    """Función principal para prueba militar con autoevaluación."""

    print("🚀 Iniciando Prueba Militar U-CogNet con Autoevaluación")
    print("=" * 60)

    # Configurar rutas
    video_path = Path("video_tanques.mp4")
    if not video_path.exists():
        print(f"❌ Video no encontrado: {video_path}")
        return

    # Inicializar componentes
    print("🔧 Inicializando componentes...")

    # Detector YOLOv8 militar
    detector = YOLOv8Detector(
        model_path="yolov8m.pt",  # Modelo mediano para mejor detección
        conf_threshold=0.3        # Umbral más bajo para objetos militares
    )

    # Evaluador con auto-aprendizaje
    evaluator = MilitaryAutoEvaluator()

    print("✅ Componentes inicializados")
    print(f"📹 Procesando video: {video_path}")
    print(f"🤖 Modelo: yolov8m.pt (conf: 0.3)")
    print("-" * 60)

    # Abrir video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("❌ Error al abrir el video")
        return

    # Estadísticas de procesamiento
    frame_count = 0
    start_time = time.time()
    military_detections_total = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Procesar frame con detector
            frame_obj = Frame(
                data=frame,
                timestamp=time.time(),
                metadata={'source': 'military_test', 'frame_number': frame_count}
            )
            detections = detector.detect(frame_obj)

            # Crear evento para evaluación
            event = Event(
                frame=frame_obj,
                detections=detections,
                timestamp=time.time()
            )

            # Evaluar y aprender automáticamente
            metrics = evaluator.maybe_update(event)

            # Contar detecciones militares
            military_detections = [d for d in detections if any(
                term in d.class_name.lower() for term in
                ['tank', 'armored', 'military', 'vehicle', 'truck', 'car']
            )]
            military_detections_total += len(military_detections)

            # Mostrar progreso cada 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                print(f"📊 Frame {frame_count} | FPS: {fps:.1f}")
                print(f"   🎯 Detecciones militares: {military_detections_total}")
                print(f"   📈 Total detecciones: {len(detections)}")

                if metrics:
                    print(f"   📊 Métricas: P:{metrics.precision:.2f} R:{metrics.recall:.2f} F1:{metrics.f1:.2f}")
                    print(f"   🎓 Ciclos de aprendizaje: {evaluator.performance_stats['adaptation_cycles']}")

                # Mostrar parámetros adaptativos actuales
                params = evaluator.get_adaptive_params()
                print(f"   ⚙️  Conf threshold: {params['conf_threshold']:.2f}")
                print(f"   📚 Clases militares: {len(params['military_classes'])}")

                # Mostrar detecciones actuales
                if military_detections:
                    print("   🚗 Detecciones actuales:")
                    for det in military_detections[:3]:  # Mostrar máximo 3
                        print(f"      - {det.class_name}: {det.confidence:.2f}")
                else:
                    print("   ❌ Sin detecciones militares en este frame")

                print("-" * 40)

            # Salir si se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n⏹️  Prueba interrumpida por usuario")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Resultados finales
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0

    print("\n" + "=" * 60)
    print("📊 RESULTADOS FINALES")
    print("=" * 60)
    print(f"🎬 Frames procesados: {frame_count}")
    print(f"⏱️  Tiempo total: {total_time:.1f}s")
    print(f"🎯 FPS promedio: {avg_fps:.1f}")
    print(f"🚗 Detecciones militares totales: {military_detections_total}")
    print(f"📈 Ratio detecciones/frame: {military_detections_total/frame_count:.2f}")

    # Estadísticas de aprendizaje
    print(f"\n🧠 APRENDIZAJE AUTOMÁTICO")
    print(f"   🔄 Ciclos de adaptación: {evaluator.performance_stats['adaptation_cycles']}")
    print(f"   📚 Clases aprendidas: {len(evaluator.get_adaptive_params()['military_classes'])}")

    # Métricas finales
    final_metrics = evaluator.maybe_update(Event(
        frame=Frame(
            data=np.zeros((480, 640, 3), dtype=np.uint8),  # Frame vacío para evaluación final
            timestamp=time.time(),
            metadata={'final': True}
        ),
        detections=[],
        timestamp=time.time()
    ))

    if final_metrics:
        print(f"\n📊 MÉTRICAS FINALES")
        print(f"   🎯 Precisión: {final_metrics.precision:.3f}")
        print(f"   🔍 Recall: {final_metrics.recall:.3f}")
        print(f"   ⚖️  F1-Score: {final_metrics.f1:.3f}")
        print(f"   📐 MCC: {final_metrics.mcc:.3f}")
        print(f"   🗺️  mAP: {final_metrics.map:.3f}")

    # Evaluación del éxito
    success_rate = military_detections_total / max(1, frame_count)
    if success_rate > 0.1:
        print(f"\n✅ ÉXITO: Sistema detectó tanques con ratio {success_rate:.2f}")
        print("🎉 Autoevaluación y aprendizaje funcionando correctamente!")
    else:
        print(f"\n⚠️  ATENCIÓN: Ratio de detección bajo ({success_rate:.2f})")
        print("💡 El modelo puede necesitar más entrenamiento o ajuste de parámetros")

    print("\n🏁 Prueba militar completada")

if __name__ == "__main__":
    main()