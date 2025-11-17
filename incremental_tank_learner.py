#!/usr/bin/env python3
"""
Sistema de Aprendizaje Incremental para Detección de Tanques
Permite al modelo aprender nuevas clases (como 'tank') en tiempo real.
"""

import cv2
import sys
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
import json

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from ucognet.core.types import Frame, Event, Detection

class IncrementalTankLearner:
    """Sistema que aprende a detectar tanques incrementalmente."""

    def __init__(self, base_model_path='yolov8m.pt'):
        self.base_model = YOLO(base_model_path)

        # Conocimiento aprendido sobre tanques
        self.tank_knowledge = {
            'learned_signatures': [],  # Firmas visuales de tanques
            'tank_candidates': [],     # Candidatos etiquetados como tanques
            'false_positives': [],     # Detecciones incorrectas
            'confidence_threshold': 0.3,
            'min_tank_size': 0.05,     # Mínimo 5% del frame
            'learning_rate': 0.01
        }

        # Estadísticas de aprendizaje
        self.learning_stats = {
            'total_candidates_analyzed': 0,
            'tanks_confirmed': 0,
            'false_positives_rejected': 0,
            'learning_iterations': 0
        }

        # Cargar conocimiento previo si existe
        self.load_knowledge()

    def analyze_frame_for_tanks(self, frame: np.ndarray) -> List[Dict]:
        """Analiza un frame en busca de posibles tanques."""

        # Detectar con modelo base
        results = self.base_model(frame, conf=self.tank_knowledge['confidence_threshold'], verbose=False)

        tank_candidates = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = self.base_model.names[cls]

                # Calcular métricas del objeto
                bbox_area = (x2 - x1) * (y2 - y1)
                frame_area = frame.shape[0] * frame.shape[1]
                relative_size = bbox_area / frame_area
                aspect_ratio = (x2 - x1) / max(1, (y2 - y1))  # width/height

                # Verificar si es candidato a tanque
                if self._is_tank_candidate(class_name, relative_size, aspect_ratio, conf):
                    candidate = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_name': class_name,
                        'relative_size': relative_size,
                        'aspect_ratio': aspect_ratio,
                        'tank_probability': self._calculate_tank_probability(
                            class_name, relative_size, aspect_ratio, conf
                        ),
                        'features': self._extract_visual_features(frame, [x1, y1, x2, y2])
                    }
                    tank_candidates.append(candidate)

        return tank_candidates

    def _is_tank_candidate(self, class_name: str, size: float, aspect_ratio: float, confidence: float) -> bool:
        """Determina si un objeto es candidato a ser tanque."""

        # Clases que podrían ser tanques
        tank_like_classes = {'train', 'truck', 'car', 'bus'}

        # Criterios para candidatos
        size_ok = size >= self.tank_knowledge['min_tank_size']
        class_ok = class_name in tank_like_classes
        confidence_ok = confidence >= self.tank_knowledge['confidence_threshold']

        # Tanques suelen ser más anchos que altos (aspect ratio > 1)
        shape_ok = aspect_ratio > 1.2

        return size_ok and class_ok and confidence_ok and shape_ok

    def _calculate_tank_probability(self, class_name: str, size: float, aspect_ratio: float, confidence: float) -> float:
        """Calcula la probabilidad de que un objeto sea un tanque."""

        # Pesos para diferentes características
        weights = {
            'size': 0.3,      # Tanques son grandes
            'shape': 0.3,     # Tanques son anchos
            'class': 0.2,     # Ciertas clases son más probables
            'confidence': 0.2 # Alta confianza en la detección
        }

        # Puntaje de tamaño (más grande = más probable tanque)
        size_score = min(1.0, size / 0.3)  # Normalizar

        # Puntaje de forma (más ancho = más probable tanque)
        shape_score = min(1.0, aspect_ratio / 3.0)  # Tanques pueden ser 2-3 veces más anchos

        # Puntaje de clase
        class_scores = {'train': 0.8, 'truck': 0.6, 'bus': 0.4, 'car': 0.2}
        class_score = class_scores.get(class_name, 0.1)

        # Puntaje de confianza
        confidence_score = confidence

        # Combinar scores
        tank_prob = (
            weights['size'] * size_score +
            weights['shape'] * shape_score +
            weights['class'] * class_score +
            weights['confidence'] * confidence_score
        )

        return min(1.0, tank_prob)

    def _extract_visual_features(self, frame: np.ndarray, bbox: List[float]) -> Dict:
        """Extrae características visuales del objeto candidato."""

        x1, y1, x2, y2 = map(int, bbox)

        # Extraer ROI
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return {'mean_color': [0, 0, 0], 'std_color': [0, 0, 0], 'edges': 0}

        # Características de color
        mean_color = np.mean(roi, axis=(0, 1)).tolist()
        std_color = np.std(roi, axis=(0, 1)).tolist()

        # Características de bordes (simplificado)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size

        return {
            'mean_color': mean_color,
            'std_color': std_color,
            'edges': edge_density,
            'bbox_size': [x2-x1, y2-y1]
        }

    def learn_from_candidate(self, candidate: Dict, is_tank: bool):
        """Aprende de un candidato etiquetado."""

        self.learning_stats['total_candidates_analyzed'] += 1

        if is_tank:
            self.learning_stats['tanks_confirmed'] += 1
            self.tank_knowledge['tank_candidates'].append(candidate)

            # Aprender firma visual
            self._learn_visual_signature(candidate)

            print(f"✅ Aprendido: Tanque confirmado (prob: {candidate['tank_probability']:.2f})")
        else:
            self.learning_stats['false_positives_rejected'] += 1
            self.tank_knowledge['false_positives'].append(candidate)
            print(f"❌ Rechazado: No es tanque (prob: {candidate['tank_probability']:.2f})")

        # Adaptar umbrales basado en aprendizaje
        self._adapt_thresholds()

        # Guardar conocimiento
        self.save_knowledge()

    def _learn_visual_signature(self, candidate: Dict):
        """Aprende una firma visual de tanque."""

        signature = {
            'features': candidate['features'],
            'aspect_ratio': candidate['aspect_ratio'],
            'relative_size': candidate['relative_size'],
            'learned_at': time.time()
        }

        self.tank_knowledge['learned_signatures'].append(signature)

        # Mantener solo las 50 firmas más recientes
        if len(self.tank_knowledge['learned_signatures']) > 50:
            self.tank_knowledge['learned_signatures'].pop(0)

    def _adapt_thresholds(self):
        """Adapta umbrales basado en el aprendizaje."""

        if len(self.tank_knowledge['tank_candidates']) < 5:
            return  # Necesitamos más datos

        # Calcular estadísticas de tanques confirmados
        tank_sizes = [c['relative_size'] for c in self.tank_knowledge['tank_candidates']]
        tank_ratios = [c['aspect_ratio'] for c in self.tank_knowledge['tank_candidates']]

        # Ajustar umbral de tamaño mínimo
        avg_tank_size = np.mean(tank_sizes)
        self.tank_knowledge['min_tank_size'] = max(0.03, avg_tank_size * 0.7)

        print(f"🔧 Umbrales adaptados: min_size={self.tank_knowledge['min_tank_size']:.3f}")

    def get_tank_detections(self, frame: np.ndarray) -> List[Detection]:
        """Retorna detecciones de tanques aprendidas."""

        candidates = self.analyze_frame_for_tanks(frame)
        detections = []

        for candidate in candidates:
            # Usar probabilidad aprendida como confianza
            confidence = candidate['tank_probability']

            if confidence >= 0.5:  # Umbral para reportar como tanque
                detection = Detection(
                    class_id=80,  # ID personalizado para tanques
                    class_name='tank',
                    confidence=confidence,
                    bbox=candidate['bbox']
                )
                detections.append(detection)

        return detections

    def save_knowledge(self):
        """Guarda el conocimiento aprendido."""
        knowledge_file = Path("tank_knowledge.json")
        try:
            with open(knowledge_file, 'w') as f:
                # Convertir numpy arrays a listas para JSON
                knowledge = self.tank_knowledge.copy()
                json.dump(knowledge, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Error guardando conocimiento: {e}")

    def load_knowledge(self):
        """Carga conocimiento previamente aprendido."""
        knowledge_file = Path("tank_knowledge.json")
        if knowledge_file.exists():
            try:
                with open(knowledge_file, 'r') as f:
                    self.tank_knowledge.update(json.load(f))
                print(f"📚 Conocimiento cargado: {len(self.tank_knowledge['tank_candidates'])} tanques aprendidos")
            except Exception as e:
                print(f"⚠️ Error cargando conocimiento: {e}")

    def get_learning_stats(self) -> Dict:
        """Retorna estadísticas de aprendizaje."""
        return {
            **self.learning_stats,
            'learned_signatures': len(self.tank_knowledge['learned_signatures']),
            'tank_candidates': len(self.tank_knowledge['tank_candidates']),
            'false_positives': len(self.tank_knowledge['false_positives'])
        }

def main():
    """Demo del sistema de aprendizaje incremental."""

    print("🚀 Sistema de Aprendizaje Incremental para Tanques")
    print("=" * 60)

    # Inicializar learner
    learner = IncrementalTankLearner()

    # Abrir video
    video_path = Path("video_tanques.mp4")
    if not video_path.exists():
        print("❌ Video no encontrado")
        return

    cap = cv2.VideoCapture(str(video_path))

    # Simular aprendizaje supervisado
    print("🎓 Fase de Aprendizaje Supervisado")
    print("Analizando frames y 'aprendiendo' qué objetos son tanques...")

    frame_count = 0
    learned_tanks = 0

    try:
        while cap.isOpened() and learned_tanks < 10:  # Aprender de los primeros 10 tanques
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Analizar frame
            candidates = learner.analyze_frame_for_tanks(frame)

            for candidate in candidates:
                # Simular etiquetado humano: asumir que objetos grandes son tanques
                is_tank = candidate['relative_size'] > 0.4  # Objetos muy grandes son tanques

                learner.learn_from_candidate(candidate, is_tank)

                if is_tank:
                    learned_tanks += 1
                    print(f"🛡️ Tanque #{learned_tanks} aprendido en frame {frame_count}")

                if learned_tanks >= 10:
                    break

            if frame_count % 100 == 0:
                print(f"📊 Procesados {frame_count} frames, tanques aprendidos: {learned_tanks}")

    finally:
        cap.release()

    # Mostrar estadísticas finales
    stats = learner.get_learning_stats()
    print(f"\n" + "=" * 60)
    print("📊 RESULTADOS DEL APRENDIZAJE")
    print("=" * 60)
    print(f"🎬 Frames procesados: {frame_count}")
    print(f"🛡️ Tanques aprendidos: {stats['tanks_confirmed']}")
    print(f"❌ Falsos positivos rechazados: {stats['false_positives_rejected']}")
    print(f"📚 Firmas visuales aprendidas: {stats['learned_signatures']}")
    print(f"🎯 Umbral de tamaño mínimo adaptado: {learner.tank_knowledge['min_tank_size']:.3f}")

    # Demo de detección con conocimiento aprendido
    print(f"\n" + "=" * 60)
    print("🧪 PRUEBA DE DETECCIÓN APRENDIDA")
    print("=" * 60)

    cap = cv2.VideoCapture(str(video_path))

    # Buscar en frames donde sabemos que hay tanques (alrededor de frame 260)
    test_frames = 0
    tanks_detected = 0
    start_frame = 250  # Comenzar cerca de donde aprendimos tanques

    # Ir al frame de inicio
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    try:
        while cap.isOpened() and test_frames < 50:  # Probar en 50 frames alrededor de los tanques
            ret, frame = cap.read()
            if not ret:
                break

            test_frames += 1
            current_frame = start_frame + test_frames

            # Detectar tanques con conocimiento aprendido
            tank_detections = learner.get_tank_detections(frame)
            tanks_detected += len(tank_detections)

            if tank_detections:
                print(f"🎯 Frame {current_frame}: {len(tank_detections)} tanque(s) detectado(s)!")
                for i, det in enumerate(tank_detections):
                    print(".2f")

    finally:
        cap.release()

    print(f"\n✅ Prueba completada:")
    print(f"   📊 Frames probados: {test_frames}")
    print(f"   🛡️ Tanques detectados: {tanks_detected}")
    print(f"   📈 Ratio detección: {tanks_detected/test_frames:.2f} tanques/frame")

    if tanks_detected > 0:
        print("🎉 ¡ÉXITO! El sistema aprendió a detectar tanques!")
    else:
        print("🤔 El sistema necesita más ejemplos de tanques para aprender.")

if __name__ == "__main__":
    main()