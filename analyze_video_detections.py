#!/usr/bin/env python3
"""
Analizador de Video para Detección de Tanques
Verifica qué está detectando realmente el modelo en el video
"""

import cv2
import sys
import os
from pathlib import Path
import numpy as np
from collections import Counter

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

def analyze_video_detections(video_path, model_path='yolov8m.pt', conf_threshold=0.1):
    """Analiza todas las detecciones en el video"""

    print("🔍 Analizando detecciones en el video...")
    print(f"📹 Video: {video_path}")
    print(f"🤖 Modelo: {model_path}")
    print(f"🎯 Confianza mínima: {conf_threshold}")
    print("-" * 50)

    # Cargar modelo
    model = YOLO(model_path)

    # Abrir video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("❌ Error al abrir el video")
        return

    # Estadísticas
    frame_count = 0
    total_detections = 0
    detections_by_class = Counter()
    confidence_scores = []
    tank_related_detections = []

    # Clases que podrían ser tanques
    potential_tank_classes = {'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'}

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Detectar con umbral muy bajo
            results = model(frame, conf=conf_threshold, verbose=False)

            frame_detections = 0

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = model.names[cls]

                    total_detections += 1
                    frame_detections += 1
                    detections_by_class[class_name] += 1
                    confidence_scores.append(conf)

                    # Verificar si podría ser un tanque
                    if class_name in potential_tank_classes:
                        # Calcular tamaño relativo
                        frame_area = frame.shape[0] * frame.shape[1]
                        bbox_area = (x2 - x1) * (y2 - y1)
                        relative_size = bbox_area / frame_area

                        tank_related_detections.append({
                            'frame': frame_count,
                            'class': class_name,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2],
                            'relative_size': relative_size
                        })

            # Mostrar progreso cada 100 frames
            if frame_count % 100 == 0:
                print(f"📊 Frame {frame_count}: {frame_detections} detecciones")

        # Mostrar resumen
        print("\n" + "=" * 60)
        print("📊 ANÁLISIS COMPLETO")
        print("=" * 60)
        print(f"🎬 Frames procesados: {frame_count}")
        print(f"🎯 Total detecciones: {total_detections}")
        print(f"📊 Confianza promedio: {np.mean(confidence_scores):.2f}")
        print(f"📊 Confianza máxima: {np.max(confidence_scores):.2f}")
        print(f"📊 Confianza mínima: {np.min(confidence_scores):.2f}")
        print(f"📈 Detecciones por frame: {total_detections/frame_count:.2f}")

        print(f"\n🔍 Clases detectadas (top 10):")
        for class_name, count in detections_by_class.most_common(10):
            percentage = (count / total_detections) * 100
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        # Análisis específico de tanques
        print(f"\n🚗 POSIBLES DETECCIONES DE TANQUES:")
        print(f"   Candidatos encontrados: {len(tank_related_detections)}")

        if tank_related_detections:
            # Ordenar por confianza
            tank_related_detections.sort(key=lambda x: x['confidence'], reverse=True)

            print("   Top 5 candidatos:")
            for i, det in enumerate(tank_related_detections[:5]):
                print(f"      {i+1}. Frame {det['frame']}: {det['class']} "
                      f"(conf: {det['confidence']:.2f}, size: {det['relative_size']:.3f})")

            # Análisis de tamaños
            sizes = [d['relative_size'] for d in tank_related_detections]
            print(f"   Tamaño promedio: {np.mean(sizes):.4f}")
            print(f"   Tamaño máximo: {np.max(sizes):.4f}")
            print(f"   Tamaño mínimo: {np.min(sizes):.4f}")
            # Verificar si hay objetos grandes que podrían ser tanques
            large_objects = [d for d in tank_related_detections if d['relative_size'] > 0.05]
            print(f"   Objetos grandes (>5% del frame): {len(large_objects)}")

            if large_objects:
                print("   Objetos grandes detectados:")
                for obj in large_objects[:3]:
                    print(f"      - {obj['class']} en frame {obj['frame']}: "
                          ".3f")
        else:
            print("   ❌ No se encontraron candidatos que podrían ser tanques")
            print("   💡 El modelo YOLOv8 estándar no tiene clase 'tank' entrenada")

    finally:
        cap.release()

    return {
        'total_frames': frame_count,
        'total_detections': total_detections,
        'detections_by_class': dict(detections_by_class),
        'tank_candidates': tank_related_detections,
        'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0
    }

def main():
    """Función principal"""
    video_path = Path("video_tanques.mp4")

    if not video_path.exists():
        print(f"❌ Video no encontrado: {video_path}")
        return

    # Análisis con diferentes umbrales
    thresholds = [0.1, 0.2, 0.3, 0.5]

    print("🔬 ANÁLISIS DETALLADO DEL VIDEO")
    print("Comparando detecciones con diferentes umbrales de confianza")
    print("=" * 80)

    results = {}
    for threshold in thresholds:
        print(f"\n🎯 Probando con conf_threshold = {threshold}")
        result = analyze_video_detections(video_path, conf_threshold=threshold)
        results[threshold] = result

        # Resumen rápido
        tank_candidates = len(result['tank_candidates'])
        large_objects = len([d for d in result['tank_candidates'] if d['relative_size'] > 0.05])
        print(f"   📊 Resumen: {result['total_detections']} dets, "
              f"{tank_candidates} candidatos tanque, {large_objects} objetos grandes")

    # Análisis comparativo
    print(f"\n" + "=" * 80)
    print("📈 ANÁLISIS COMPARATIVO")
    print("=" * 80)

    print("Umbral | Total Dets | Candidatos Tanque | Objetos Grandes | Conf Promedio")
    print("-------|------------|-------------------|-----------------|---------------")
    for threshold in thresholds:
        result = results[threshold]
        tank_candidates = len(result['tank_candidates'])
        large_objects = len([d for d in result['tank_candidates'] if d['relative_size'] > 0.05])
        avg_conf = result['avg_confidence']
        print("5.1f")

    # Conclusiones
    best_threshold = max(results.keys(), key=lambda t: len(results[t]['tank_candidates']))
    best_result = results[best_threshold]

    print(f"\n🎯 CONCLUSIÓN:")
    print(f"   Mejor umbral: {best_threshold}")
    print(f"   Candidatos tanque encontrados: {len(best_result['tank_candidates'])}")

    if best_result['tank_candidates']:
        large_tanks = [d for d in best_result['tank_candidates'] if d['relative_size'] > 0.05]
        print(f"   Tanques potenciales grandes: {len(large_tanks)}")

        if large_tanks:
            print("   ✅ ¡SÍ HAY TANQUES en el video!")
            print("   💡 Pero el modelo YOLOv8 estándar no los reconoce como 'tank'")
            print("   🔧 Necesitamos entrenar el modelo para detectar tanques")
        else:
            print("   ⚠️  Los candidatos son muy pequeños para ser tanques")
    else:
        print("   ❌ No se encontraron candidatos que podrían ser tanques")
        print("   🤔 El video podría no tener tanques visibles o estar muy comprimido")

if __name__ == "__main__":
    main()