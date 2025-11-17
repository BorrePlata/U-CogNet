"""
Sistema de Autoevaluación Militar para U-CogNet
Implementa aprendizaje automático adaptativo para detección de objetos militares.
"""

from typing import List, Dict, Optional, Tuple
from ucognet.core.interfaces import Evaluator
from ucognet.core.types import Event, Metrics, Detection
import numpy as np
from collections import defaultdict
import time
from incremental_tank_learner import IncrementalTankLearner

class MilitaryAutoEvaluator(Evaluator):
    """Evaluador inteligente que aprende y adapta parámetros para detección militar."""

    def __init__(self):
        # Estado de aprendizaje
        self.learning_history: List[Dict] = []
        self.max_history = 500  # Mantener más historial para aprendizaje

        # Sistema de aprendizaje incremental para tanques
        self.tank_learner = IncrementalTankLearner()
        self.tank_learning_enabled = True

        # Parámetros adaptativos
        self.adaptive_params = {
            'conf_threshold': 0.3,  # Umbral inicial más bajo para objetos militares
            'iou_threshold': 0.45,
            'max_detections': 100,
            'military_classes': {'tank', 'armored_vehicle', 'military_vehicle', 'truck', 'car'},
            'learning_rate': 0.01
        }

        # Estadísticas de aprendizaje
        self.performance_stats = {
            'frames_processed': 0,
            'military_detections': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'adaptation_cycles': 0
        }

        # Memoria de objetos detectados
        self.detection_memory: Dict[str, List] = defaultdict(list)

        # Sistema de feedback para aprendizaje
        self.feedback_system = {
            'last_adaptation': time.time(),
            'adaptation_interval': 100,  # Frames entre adaptaciones
            'min_samples_for_learning': 50
        }

    def maybe_update(self, event: Event) -> Optional[Metrics]:
        """Evalúa rendimiento y adapta parámetros automáticamente."""

        self.performance_stats['frames_processed'] += 1
        current_detections = event.detections

        # Aprendizaje incremental de tanques si está habilitado
        if self.tank_learning_enabled and hasattr(event, 'frame') and event.frame is not None:
            self._learn_from_frame(event.frame.data, current_detections)

        # Almacenar en memoria de aprendizaje
        self._store_detection_history(current_detections)

        # Calcular métricas militares específicas
        military_metrics = self._calculate_military_metrics(current_detections)

        # Sistema de auto-aprendizaje
        if self._should_adapt():
            self._adapt_parameters(current_detections)

        # Actualizar estadísticas
        self._update_performance_stats(current_detections)

        return Metrics(
            precision=round(military_metrics['precision'], 3),
            recall=round(military_metrics['recall'], 3),
            f1=round(military_metrics['f1'], 3),
            mcc=round(military_metrics['mcc'], 3),
            map=round(military_metrics['map'], 3)
        )

    def _calculate_military_metrics(self, detections: List[Detection]) -> Dict[str, float]:
        """Calcula métricas específicas para escenarios militares."""

        # Identificar detecciones militares
        military_detections = []
        civilian_detections = []

        for detection in detections:
            class_lower = detection.class_name.lower()
            if any(military_term in class_lower for military_term in
                   ['tank', 'armored', 'military', 'vehicle', 'truck', 'car']):
                military_detections.append(detection)
            else:
                civilian_detections.append(detection)

        # Calcular precisión militar (foco en detecciones relevantes)
        if military_detections:
            avg_confidence = np.mean([d.confidence for d in military_detections])
            consistency_score = self._calculate_temporal_consistency(military_detections)
            precision = min(0.95, avg_confidence * 0.6 + consistency_score * 0.4)
        else:
            precision = 0.5  # Valor neutral cuando no hay detecciones militares

        # Calcular recall (capacidad de detectar amenazas)
        total_detections = len(detections)
        if total_detections > 0:
            military_ratio = len(military_detections) / total_detections
            recall = military_ratio * 0.7 + min(1.0, total_detections / 10.0) * 0.3
        else:
            recall = 0.0

        # F1 y MCC
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        mcc = self._calculate_military_mcc(military_detections, civilian_detections)

        # mAP estimado
        map_score = self._estimate_military_map(detections)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc,
            'map': map_score
        }

    def _calculate_temporal_consistency(self, detections: List[Detection]) -> float:
        """Calcula consistencia temporal de detecciones."""
        if len(self.learning_history) < 3:
            return 0.5

        # Comparar con frames anteriores
        recent_detections = []
        for i in range(min(5, len(self.learning_history))):
            recent_detections.extend(self.learning_history[-(i+1)].get('detections', []))

        if not recent_detections:
            return 0.5

        # Calcular similitud
        current_classes = set(d.class_name for d in detections)
        recent_classes = set(d.class_name for d in recent_detections)

        intersection = len(current_classes & recent_classes)
        union = len(current_classes | recent_classes)

        return intersection / union if union > 0 else 0.5

    def _calculate_military_mcc(self, military: List[Detection], civilian: List[Detection]) -> float:
        """Calcula MCC específico para contexto militar."""
        # En contexto militar: military = TP, civilian = FP
        tp = len(military)  # Verdaderos positivos (detecciones militares)
        fp = len(civilian) * 0.3  # Falsos positivos (penalizados)
        fn = max(0, 5 - tp)  # Falsos negativos (esperamos al menos algunas detecciones militares)
        tn = max(0, 10 - fp)  # Verdaderos negativos

        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return (tp * tn - fp * fn) / denominator if denominator > 0 else 0

    def _estimate_military_map(self, detections: List[Detection]) -> float:
        """Estima mAP para contexto militar."""
        if not detections:
            return 0.0

        # mAP militar: favorecer detecciones de vehículos y objetos relevantes
        military_score = 0
        civilian_penalty = 0

        for detection in detections:
            class_lower = detection.class_name.lower()
            if any(term in class_lower for term in ['tank', 'vehicle', 'truck', 'car']):
                military_score += detection.confidence
            else:
                civilian_penalty += 0.1  # Penalización por detecciones irrelevantes

        total_score = military_score - civilian_penalty
        return max(0, min(1.0, total_score / max(1, len(detections))))

    def _store_detection_history(self, detections: List[Detection]):
        """Almacena historial de detecciones para aprendizaje."""
        detection_data = {
            'timestamp': time.time(),
            'detections': detections.copy(),
            'military_count': len([d for d in detections if any(term in d.class_name.lower()
                              for term in ['tank', 'military', 'vehicle'])]),
            'total_count': len(detections)
        }

        self.learning_history.append(detection_data)
        if len(self.learning_history) > self.max_history:
            self.learning_history.pop(0)

    def _should_adapt(self) -> bool:
        """Determina si es momento de adaptar parámetros."""
        frames_since_adaptation = self.performance_stats['frames_processed'] - self.performance_stats['adaptation_cycles']

        return (frames_since_adaptation >= self.feedback_system['adaptation_interval'] and
                len(self.learning_history) >= self.feedback_system['min_samples_for_learning'])

    def _adapt_parameters(self, current_detections: List[Detection]):
        """Adapta parámetros basado en rendimiento histórico."""
        print("🔄 Iniciando adaptación de parámetros militares...")

        # Analizar rendimiento reciente
        recent_history = self.learning_history[-50:]  # Últimos 50 frames

        military_detections_recent = sum(h['military_count'] for h in recent_history)
        total_detections_recent = sum(h['total_count'] for h in recent_history)

        # Calcular ratios
        military_ratio = military_detections_recent / max(1, total_detections_recent)

        # Adaptar umbral de confianza
        if military_ratio < 0.1:  # Muy pocas detecciones militares
            self.adaptive_params['conf_threshold'] = max(0.1,
                self.adaptive_params['conf_threshold'] - self.adaptive_params['learning_rate'])
            print(f"⚡ Bajando umbral de confianza a {self.adaptive_params['conf_threshold']:.2f}")
        elif military_ratio > 0.5:  # Demasiadas detecciones (posible ruido)
            self.adaptive_params['conf_threshold'] = min(0.8,
                self.adaptive_params['conf_threshold'] + self.adaptive_params['learning_rate'])
            print(f"🎯 Subiendo umbral de confianza a {self.adaptive_params['conf_threshold']:.2f}")

        # Adaptar clases militares basadas en lo que se detecta
        detected_classes = set()
        for detection in current_detections:
            detected_classes.add(detection.class_name.lower())

        # Aprender nuevas clases potencialmente militares
        for class_name in detected_classes:
            if any(term in class_name for term in ['vehicle', 'truck', 'car', 'tank']):
                if class_name not in self.adaptive_params['military_classes']:
                    self.adaptive_params['military_classes'].add(class_name)
                    print(f"📚 Aprendiendo nueva clase militar: {class_name}")

        self.performance_stats['adaptation_cycles'] += 1
        self.feedback_system['last_adaptation'] = time.time()

        print(f"✅ Adaptación completada. Ciclo: {self.performance_stats['adaptation_cycles']}")

    def _learn_from_frame(self, frame_data: np.ndarray, detections: List[Detection]):
        """Aprende de tanques en el frame actual."""

        # Analizar frame en busca de candidatos a tanques
        tank_candidates = self.tank_learner.analyze_frame_for_tanks(frame_data)

        # Aprender de candidatos grandes (asumir que son tanques)
        for candidate in tank_candidates:
            if candidate['relative_size'] > 0.4:  # Objetos muy grandes son tanques
                self.tank_learner.learn_from_candidate(candidate, is_tank=True)

                # Agregar detección de tanque aprendida a las detecciones actuales
                learned_tank_detections = self.tank_learner.get_tank_detections(frame_data)
                detections.extend(learned_tank_detections)

    def get_tank_learning_stats(self) -> Dict:
        """Retorna estadísticas del aprendizaje de tanques."""
        return self.tank_learner.get_learning_stats()

    def _update_performance_stats(self, detections: List[Detection]):
        """Actualiza estadísticas de rendimiento."""
        military_count = len([d for d in detections if any(term in d.class_name.lower()
                          for term in ['tank', 'military', 'vehicle', 'truck', 'car'])])

        self.performance_stats['military_detections'] += military_count

    def get_adaptive_params(self) -> Dict:
        """Retorna parámetros adaptativos actuales."""
        return self.adaptive_params.copy()

    def reset_learning(self):
        """Reinicia el estado de aprendizaje."""
        self.learning_history.clear()
        self.performance_stats = {k: 0 for k in self.performance_stats.keys()}
        print("🔄 Estado de aprendizaje reiniciado")