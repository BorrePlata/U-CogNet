"""
Evaluador del sistema U-CogNet.
Versión inicial: Métricas básicas de simulación.
"""

import time
from typing import Optional
from ..common.types import Metrics, Event
from ..common.utils import calculate_metrics
from ..common.logging import logger

class Evaluator:
    """
    Evaluador del rendimiento del sistema.
    Calcula métricas de precisión, recall, etc.
    """

    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.latencies = []
        self.start_time = time.time()
        logger.info("Evaluator inicializado")

    def update(self, y_true: Optional[int] = None, y_pred: Optional[int] = None, latency: Optional[float] = None) -> Metrics:
        """
        Actualiza métricas con nueva predicción.
        Para simulación: usa valores aleatorios si no se proporcionan.
        """
        if y_true is None or y_pred is None:
            # Simulación: asumir algunos TP/FP aleatorios
            if len(self.latencies) % 10 == 0:  # Cada 10 evaluaciones, simular error
                self.false_positives += 1
            else:
                self.true_positives += 1

        if latency:
            self.latencies.append(latency)

        # Calcular métricas actuales
        metrics_dict = calculate_metrics(
            self.true_positives, self.false_positives,
            self.true_negatives, self.false_negatives
        )

        # Añadir métricas de rendimiento
        total_time = time.time() - self.start_time
        throughput = len(self.latencies) / total_time if total_time > 0 else 0
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0

        metrics = Metrics(
            precision=metrics_dict['precision'],
            recall=metrics_dict['recall'],
            f1_score=metrics_dict['f1_score'],
            mcc=metrics_dict['mcc'],
            latency_ms=avg_latency * 1000,
            throughput_fps=throughput
        )

        logger.debug(f"Métricas actualizadas: F1={metrics.f1_score:.3f}, Latency={metrics.latency_ms:.2f}ms")

        return metrics

    def reset(self) -> None:
        """Reinicia las métricas."""
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.latencies = []
        self.start_time = time.time()
        logger.info("Métricas reiniciadas")