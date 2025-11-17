#!/usr/bin/env python3
"""
Sistema Completo U-CogNet con Aprendizaje de Tanques
Demostración completa del sistema de autoevaluación militar con aprendizaje incremental.
"""

import cv2
import sys
import os
from pathlib import Path
import numpy as np
import time

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ucognet.modules.vision.yolov8_detector import YOLOv8Detector
from ucognet.modules.eval.military_auto_evaluator import MilitaryAutoEvaluator
from ucognet.core.types import Frame, Event

def main():
    """Demostración completa del sistema U-CogNet con aprendizaje de tanques."""

    print("🚀 U-CogNet - Sistema Completo de Detección Militar con Autoevaluación")
    print("=" * 80)

    # Configurar rutas
    video_path = Path("video_tanques.mp4")
    if not video_path.exists():
        print("❌ Video no encontrado")
        return

    # Inicializar componentes del sistema
    print("🔧 Inicializando componentes del sistema...")

    # 1. Detector YOLOv8 especializado en objetos militares
    detector = YOLOv8Detector(
        model_path="yolov8m.pt",
        conf_threshold=0.3
    )

    # 2. Evaluador con aprendizaje automático de tanques
    evaluator = MilitaryAutoEvaluator()

    print("✅ Sistema inicializado")
    print(f"📹 Procesando video: {video_path}")
    print(f"🤖 Detector: YOLOv8m (conf: 0.3)")
    print(f"🧠 Evaluador: Autoevaluación militar con aprendizaje incremental")
    print("-" * 80)

    # Abrir video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("❌ Error al abrir el video")
        return

    # Estadísticas de procesamiento
    frame_count = 0
    start_time = time.time()
    total_tank_detections = 0
    learning_cycles = 0

    print("🎬 Iniciando procesamiento en tiempo real...")
    print("El sistema aprenderá automáticamente a detectar tanques")
    print("-" * 80)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Crear objeto Frame
            frame_obj = Frame(
                data=frame,
                timestamp=time.time(),
                metadata={'frame_number': frame_count, 'source': 'tank_detection_demo'}
            )

            # 1. Detectar objetos con YOLOv8
            detections = detector.detect(frame_obj)

            # 2. Crear evento para evaluación y aprendizaje
            event = Event(
                frame=frame_obj,
                detections=detections,
                timestamp=time.time()
            )

            # 3. Evaluar y aprender automáticamente
            metrics = evaluator.maybe_update(event)

            # Contar detecciones de tanques aprendidas
            tank_detections = [d for d in detections if d.class_name == 'tank']
            total_tank_detections += len(tank_detections)

            # Mostrar progreso cada 50 frames
            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                print(f"📊 Frame {frame_count} | FPS: {fps:.1f} | Tanques: {total_tank_detections}")

                # Mostrar métricas si están disponibles
                if metrics:
                    print(f"   📈 Métricas: P:{metrics.precision:.2f} R:{metrics.recall:.2f} F1:{metrics.f1:.2f}")

                # Mostrar estadísticas de aprendizaje
                learning_stats = evaluator.get_tank_learning_stats()
                if learning_stats['tanks_confirmed'] > 0:
                    print(f"   🧠 Tanques aprendidos: {learning_stats['tanks_confirmed']}")
                    print(f"   📚 Firmas visuales: {learning_stats['learned_signatures']}")

                # Mostrar detecciones actuales
                if tank_detections:
                    print("   🎯 TANQUES DETECTADOS:")
                    for det in tank_detections[:2]:  # Mostrar máximo 2
                        print(".2f")
                elif detections:
                    # Mostrar otras detecciones que podrían ser tanques
                    military_like = [d for d in detections if d.class_name in ['train', 'truck', 'car']]
                    if military_like:
                        print("   🚛 Candidatos militares:")
                        for det in military_like[:2]:
                            print(f"      - {det.class_name}: {det.confidence:.2f}")

                print("-" * 60)

            # Salir después de 1000 frames para demo
            if frame_count >= 1000:
                break

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrumpida por usuario")

    finally:
        cap.release()

    # Resultados finales
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0

    print("\n" + "=" * 80)
    print("📊 RESULTADOS FINALES - U-CogNet Sistema Completo")
    print("=" * 80)
    print(f"🎬 Frames procesados: {frame_count}")
    print(f"⏱️  Tiempo total: {total_time:.1f}s")
    print(f"🎯 FPS promedio: {avg_fps:.1f}")
    print(f"🛡️ Tanques detectados: {total_tank_detections}")
    print(f"📈 Ratio detección: {total_tank_detections/frame_count:.2f} tanques/frame")
    # Estadísticas de aprendizaje
    learning_stats = evaluator.get_tank_learning_stats()
    print(f"\n🧠 APRENDIZAJE AUTOMÁTICO")
    print(f"   🔄 Ciclos de adaptación: {evaluator.performance_stats['adaptation_cycles']}")
    print(f"   🛡️ Tanques aprendidos: {learning_stats['tanks_confirmed']}")
    print(f"   ❌ Falsos positivos rechazados: {learning_stats['false_positives_rejected']}")
    print(f"   📚 Firmas visuales aprendidas: {learning_stats['learned_signatures']}")
    print(f"   🎯 Candidatos analizados: {learning_stats['total_candidates_analyzed']}")

    # Métricas finales
    print(f"\n📊 MÉTRICAS FINALES")
    print("   (Evaluadas en el último frame procesado)")

    # Obtener métricas finales haciendo una evaluación dummy
    final_event = Event(
        frame=Frame(
            data=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=time.time(),
            metadata={'final': True}
        ),
        detections=[],
        timestamp=time.time()
    )
    final_metrics = evaluator.maybe_update(final_event)

    if final_metrics:
        print(f"   🎯 Precisión: {final_metrics.precision:.3f}")
        print(f"   🔍 Recall: {final_metrics.recall:.3f}")
        print(f"   ⚖️  F1-Score: {final_metrics.f1:.3f}")
        print(f"   📐 MCC: {final_metrics.mcc:.3f}")
        print(f"   🗺️  mAP: {final_metrics.map:.3f}")

    # Evaluación del éxito
    success_score = total_tank_detections / max(1, frame_count)
    learning_score = learning_stats['tanks_confirmed'] / max(1, learning_stats['total_candidates_analyzed'])

    print(f"\n🏆 EVALUACIÓN DEL SISTEMA")
    print(f"   📈 Ratio de detección de tanques: {success_score:.3f}")
    print(f"   🧠 Ratio de aprendizaje: {learning_score:.3f}")

    if success_score > 0.01 and learning_score > 0.5:
        print("   ✅ ÉXITO TOTAL: Sistema aprendió y detectó tanques exitosamente!")
        print("   🎉 U-CogNet demostró capacidad de aprendizaje incremental")
    elif success_score > 0.005:
        print("   ⚠️ ÉXITO PARCIAL: Detectó algunos tanques pero puede mejorar")
    else:
        print("   ❌ LIMITACIÓN: No detectó tanques suficientes")
        print("   💡 El video podría necesitar más frames con tanques visibles")

    print(f"\n🏁 Demo completada - Sistema U-CogNet operativo")

if __name__ == "__main__":
    main()