from typing import Optional, List
from ucognet.core.interfaces import Evaluator
from ucognet.core.types import Event, Metrics, Detection
import numpy as np
from collections import defaultdict

class BasicEvaluator(Evaluator):
    """Evaluator que calcula métricas básicas de rendimiento usando heurísticas."""

    def __init__(self):
        # Estado para tracking de métricas a lo largo del tiempo
        self.detection_history: List[List[Detection]] = []
        self.max_history = 100  # Mantener últimas 100 frames

        # Métricas acumuladas
        self.total_frames = 0
        self.total_detections = 0
        self.consistent_detections = 0

        # Thresholds para evaluación
        self.consistency_threshold = 0.8  # Confianza mínima para considerar consistente
        self.temporal_consistency_window = 5  # Frames para verificar consistencia temporal

    def maybe_update(self, event: Event) -> Optional[Metrics]:
        """Calcula métricas basadas en el evento actual y el historial."""

        self.total_frames += 1
        current_detections = event.detections
        self.detection_history.append(current_detections)

        # Mantener historial limitado
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)

        # Calcular métricas básicas
        precision = self._calculate_precision(current_detections)
        recall = self._calculate_recall(current_detections)
        f1 = self._calculate_f1(precision, recall)
        mcc = self._calculate_mcc(current_detections)
        map_score = self._calculate_map(current_detections)

        # Actualizar métricas de consistencia
        self._update_consistency_metrics(current_detections)

        return Metrics(
            precision=round(precision, 3),
            recall=round(recall, 3),
            f1=round(f1, 3),
            mcc=round(mcc, 3),
            map=round(map_score, 3)
        )

    def _calculate_precision(self, detections: List[Detection]) -> float:
        """Calcula precisión basada en confianza promedio y consistencia."""
        if not detections:
            return 1.0  # No hay falsos positivos si no hay detecciones

        # Precisión = TP / (TP + FP)
        # Usamos heurística: alta confianza + consistencia temporal = alta precisión
        high_conf_detections = [d for d in detections if d.confidence > self.consistency_threshold]
        consistent_detections = self._count_temporally_consistent(detections)

        if len(high_conf_detections) == 0:
            return 0.0

        # Precisión estimada basada en confianza y consistencia
        confidence_factor = np.mean([d.confidence for d in high_conf_detections])
        consistency_factor = consistent_detections / len(detections) if detections else 0

        return min(0.95, confidence_factor * 0.7 + consistency_factor * 0.3)

    def _calculate_recall(self, detections: List[Detection]) -> float:
        """Calcula recall basado en cobertura de escena y tipos de objetos."""
        if not detections:
            return 0.0

        # Recall = TP / (TP + FN)
        # Heurística: más detecciones + variedad de clases = mejor recall
        unique_classes = len(set(d.class_name for d in detections))
        high_conf_count = len([d for d in detections if d.confidence > 0.5])

        # Factor de cobertura: más clases detectadas = mejor recall
        coverage_factor = min(1.0, unique_classes / 10.0)  # Asumiendo ~10 clases importantes

        # Factor de confianza: más detecciones de alta confianza = mejor recall
        confidence_factor = high_conf_count / len(detections) if detections else 0

        return coverage_factor * 0.6 + confidence_factor * 0.4

    def _calculate_f1(self, precision: float, recall: float) -> float:
        """Calcula F1-score como media armónica de precision y recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _calculate_mcc(self, detections: List[Detection]) -> float:
        """Calcula MCC (Matthews Correlation Coefficient) usando heurísticas."""
        if not detections:
            return 0.0

        # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        # Heurística simplificada basada en consistencia y confianza
        consistency_score = self._calculate_temporal_consistency()
        confidence_score = np.mean([d.confidence for d in detections]) if detections else 0

        # MCC aproximado: valores entre -1 y 1
        mcc_approx = (consistency_score + confidence_score - 1) * 0.5
        return max(-1.0, min(1.0, mcc_approx))

    def _calculate_map(self, detections: List[Detection]) -> float:
        """Calcula mAP (mean Average Precision) usando heurísticas."""
        if not detections:
            return 0.0

        # mAP aproximado basado en confianza promedio y consistencia por clase
        class_confidences = defaultdict(list)

        for det in detections:
            class_confidences[det.class_name].append(det.confidence)

        # Calcular AP promedio por clase
        ap_scores = []
        for class_name, confidences in class_confidences.items():
            if confidences:
                # AP simplificado: promedio de confianzas ordenadas
                sorted_conf = sorted(confidences, reverse=True)
                ap = np.mean(sorted_conf)
                ap_scores.append(ap)

        return np.mean(ap_scores) if ap_scores else 0.0

    def _count_temporally_consistent(self, detections: List[Detection]) -> int:
        """Cuenta detecciones que son consistentes con frames anteriores."""
        if len(self.detection_history) < 2:
            return len(detections)

        consistent_count = 0
        current_classes = set(d.class_name for d in detections)

        # Verificar consistencia con últimas N frames
        for prev_detections in self.detection_history[-self.temporal_consistency_window:]:
            prev_classes = set(d.class_name for d in prev_detections)

            # Intersección de clases detectadas
            common_classes = current_classes.intersection(prev_classes)
            if common_classes:
                consistent_count += len(common_classes)

        return consistent_count

    def _calculate_temporal_consistency(self) -> float:
        """Calcula consistencia temporal global."""
        if len(self.detection_history) < 2:
            return 0.5  # Valor neutral

        total_consistent = 0
        total_possible = 0

        # Comparar cada frame con el siguiente
        for i in range(len(self.detection_history) - 1):
            current = self.detection_history[i]
            next_frame = self.detection_history[i + 1]

            current_classes = set(d.class_name for d in current)
            next_classes = set(d.class_name for d in next_frame)

            # Calcular Jaccard similarity
            intersection = len(current_classes.intersection(next_classes))
            union = len(current_classes.union(next_classes))

            if union > 0:
                similarity = intersection / union
                total_consistent += similarity
                total_possible += 1

        return total_consistent / total_possible if total_possible > 0 else 0.5

    def _update_consistency_metrics(self, detections: List[Detection]):
        """Actualiza métricas de consistencia global."""
        self.total_detections += len(detections)

        # Contar detecciones consistentes (alta confianza + estables temporalmente)
        consistent = len([d for d in detections if d.confidence > self.consistency_threshold])
        consistent += self._count_temporally_consistent(detections)

        self.consistent_detections += consistent