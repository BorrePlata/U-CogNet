"""
Evaluator - Sistema de Evaluación y Métricas para U-CogNet
Calcula métricas de rendimiento, confianza y coherencia del sistema
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .interfaces import EvaluatorInterface
from .tracing import get_event_bus, EventType


class MetricType(Enum):
    """Tipos de métricas disponibles"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    MCC = "mcc"  # Matthews Correlation Coefficient
    MAP = "map"  # Mean Average Precision
    CONFIDENCE = "confidence"
    LATENCY = "latency"
    RESOURCE_USAGE = "resource_usage"
    ERROR_RATE = "error_rate"


@dataclass
class EvaluationResult:
    """Resultado de una evaluación"""
    metric_type: MetricType
    value: float
    confidence: float
    timestamp: float
    context: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceReport:
    """Reporte completo de rendimiento"""
    overall_score: float
    metrics: Dict[str, EvaluationResult]
    recommendations: List[str]
    alerts: List[str]
    timestamp: float


class Evaluator(EvaluatorInterface):
    """
    Sistema de evaluación completo para U-CogNet
    Calcula métricas de rendimiento, coherencia y confianza
    """

    def __init__(self):
        self.event_bus = get_event_bus()

        # Almacenamiento de métricas
        self.metrics_history: Dict[str, List[EvaluationResult]] = {}
        self.performance_baseline: Dict[str, float] = {}
        self.evaluation_context: Dict[str, Any] = {}

        # Configuración de evaluación
        self.evaluation_window = 100  # número de mediciones para análisis
        self.confidence_threshold = 0.7
        self.performance_threshold = 0.6

        # Estado del evaluador
        self.is_evaluating = False
        self.last_evaluation = 0
        self.evaluation_frequency = 30  # segundos

        # Métricas críticas para monitoreo
        self.critical_metrics = {
            "system_confidence": 0.0,
            "error_rate": 0.0,
            "resource_efficiency": 1.0,
            "coherence_score": 0.0
        }

        print("📊 Evaluator inicializado")

    async def initialize(self) -> bool:
        """Inicializa el sistema de evaluación"""
        try:
            # Establecer baselines iniciales
            await self._establish_baselines()

            # Iniciar evaluación continua
            asyncio.create_task(self._continuous_evaluation())

            # Emitir evento de inicialización
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "Evaluator",
                outputs={"initialized": True},
                explanation="Inicialización del Evaluator"
            )

            print("✅ Evaluator inicializado correctamente")
            return True

        except Exception as e:
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "Evaluator",
                outputs={"initialized": False, "error": str(e)},
                log_level=2
            )
            print(f"❌ Error inicializando Evaluator: {e}")
            return False

    async def _establish_baselines(self):
        """Establece líneas base para métricas críticas"""
        # Baselines iniciales conservadores
        self.performance_baseline = {
            "accuracy": 0.7,
            "precision": 0.7,
            "recall": 0.7,
            "f1_score": 0.7,
            "mcc": 0.4,
            "confidence": 0.6,
            "latency": 1.0,  # segundos
            "resource_usage": 0.5,
            "error_rate": 0.05
        }

        print("📈 Baselines de rendimiento establecidos")

    async def _continuous_evaluation(self):
        """Evaluación continua del sistema"""
        while True:
            try:
                current_time = time.time()
                if current_time - self.last_evaluation >= self.evaluation_frequency:
                    await self.evaluate_performance()
                    self.last_evaluation = current_time

                await asyncio.sleep(10)  # verificar cada 10 segundos

            except Exception as e:
                print(f"⚠️ Error en evaluación continua: {e}")
                await asyncio.sleep(30)

    async def calculate_metrics(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, EvaluationResult]:
        """
        Calcula métricas de rendimiento para datos dados

        Args:
            data: Datos para evaluar
            context: Contexto adicional de evaluación

        Returns:
            Diccionario de métricas calculadas
        """
        self.is_evaluating = True
        context = context or {}

        try:
            metrics = {}

            # Métricas de clasificación si hay predicciones y etiquetas
            if "predictions" in data and "ground_truth" in data:
                classification_metrics = await self._calculate_classification_metrics(
                    data["predictions"], data["ground_truth"]
                )
                metrics.update(classification_metrics)

            # Métricas de confianza
            if "confidences" in data:
                confidence_metrics = await self._calculate_confidence_metrics(data["confidences"])
                metrics.update(confidence_metrics)

            # Métricas de rendimiento del sistema
            if "latencies" in data:
                performance_metrics = await self._calculate_performance_metrics(data)
                metrics.update(performance_metrics)

            # Almacenar métricas en historial
            for metric_name, result in metrics.items():
                if metric_name not in self.metrics_history:
                    self.metrics_history[metric_name] = []
                self.metrics_history[metric_name].append(result)

                # Mantener historial limitado
                if len(self.metrics_history[metric_name]) > self.evaluation_window:
                    self.metrics_history[metric_name] = self.metrics_history[metric_name][-self.evaluation_window:]

            # Emitir evento de métricas calculadas
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "Evaluator",
                outputs={"metrics_calculated": len(metrics)},
                context={"metric_types": list(metrics.keys())},
                explanation="Métricas calculadas exitosamente"
            )

            return metrics

        except Exception as e:
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "Evaluator",
                outputs={"metrics_error": str(e)},
                log_level=2
            )
            raise
        finally:
            self.is_evaluating = False

    async def _calculate_classification_metrics(self, predictions: List[Any], ground_truth: List[Any]) -> Dict[str, EvaluationResult]:
        """Calcula métricas de clasificación"""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predicciones y etiquetas deben tener la misma longitud")

        if not predictions:
            return {}

        # Convertir a arrays numpy para cálculos eficientes
        y_pred = np.array(predictions)
        y_true = np.array(ground_truth)

        # Calcular métricas básicas
        accuracy = np.mean(y_pred == y_true)

        # Para métricas más complejas, asumir clasificación binaria si es necesario
        unique_labels = np.unique(np.concatenate([y_pred, y_true]))

        if len(unique_labels) == 2:
            # Clasificación binaria
            tp = np.sum((y_pred == 1) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # MCC (Matthews Correlation Coefficient)
            mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0
        else:
            # Multiclass - aproximaciones
            precision = accuracy  # simplificación
            recall = accuracy
            f1 = accuracy
            mcc = 0.0  # no calculable fácilmente para multiclass

        # mAP (Mean Average Precision) - simplificado
        map_score = accuracy  # aproximación

        timestamp = time.time()

        metrics = {
            "accuracy": EvaluationResult(
                MetricType.ACCURACY, accuracy, 0.9, timestamp,
                {"num_samples": len(predictions)}
            ),
            "precision": EvaluationResult(
                MetricType.PRECISION, precision, 0.8, timestamp,
                {"num_classes": len(unique_labels)}
            ),
            "recall": EvaluationResult(
                MetricType.RECALL, recall, 0.8, timestamp,
                {"num_classes": len(unique_labels)}
            ),
            "f1_score": EvaluationResult(
                MetricType.F1_SCORE, f1, 0.8, timestamp,
                {"num_classes": len(unique_labels)}
            ),
            "mcc": EvaluationResult(
                MetricType.MCC, mcc, 0.7, timestamp,
                {"num_classes": len(unique_labels)}
            ),
            "map": EvaluationResult(
                MetricType.MAP, map_score, 0.6, timestamp,
                {"num_classes": len(unique_labels)}
            )
        }

        return metrics

    async def _calculate_confidence_metrics(self, confidences: List[float]) -> Dict[str, EvaluationResult]:
        """Calcula métricas de confianza"""
        if not confidences:
            return {}

        conf_array = np.array(confidences)
        timestamp = time.time()

        # Estadísticas de confianza
        mean_confidence = np.mean(conf_array)
        std_confidence = np.std(conf_array)
        min_confidence = np.min(conf_array)
        max_confidence = np.max(conf_array)

        # Confianza en las predicciones de alta confianza
        high_conf_threshold = 0.8
        high_conf_predictions = conf_array > high_conf_threshold
        high_conf_ratio = np.mean(high_conf_predictions)

        # Estabilidad de confianza
        confidence_stability = 1.0 - std_confidence  # mayor estabilidad = menor desviación

        metrics = {
            "confidence_mean": EvaluationResult(
                MetricType.CONFIDENCE, mean_confidence, 0.9, timestamp,
                {"std": std_confidence, "min": min_confidence, "max": max_confidence}
            ),
            "confidence_stability": EvaluationResult(
                MetricType.CONFIDENCE, confidence_stability, 0.8, timestamp,
                {"std": std_confidence}
            ),
            "high_confidence_ratio": EvaluationResult(
                MetricType.CONFIDENCE, high_conf_ratio, 0.7, timestamp,
                {"threshold": high_conf_threshold}
            )
        }

        return metrics

    async def _calculate_performance_metrics(self, data: Dict[str, Any]) -> Dict[str, EvaluationResult]:
        """Calcula métricas de rendimiento del sistema"""
        timestamp = time.time()
        metrics = {}

        # Latencia
        if "latencies" in data:
            latencies = np.array(data["latencies"])
            mean_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)

            metrics["latency"] = EvaluationResult(
                MetricType.LATENCY, mean_latency, 0.9, timestamp,
                {"p95": p95_latency, "std": np.std(latencies)}
            )

        # Uso de recursos
        if "resource_usage" in data:
            resource_usage = data["resource_usage"]
            if isinstance(resource_usage, (list, np.ndarray)):
                resource_usage = np.mean(resource_usage)

            metrics["resource_usage"] = EvaluationResult(
                MetricType.RESOURCE_USAGE, resource_usage, 0.8, timestamp,
                {"type": "average"}
            )

        # Tasa de error
        if "errors" in data:
            error_count = data["errors"]
            total_operations = data.get("total_operations", 1)
            error_rate = error_count / total_operations if total_operations > 0 else 0

            metrics["error_rate"] = EvaluationResult(
                MetricType.ERROR_RATE, error_rate, 0.9, timestamp,
                {"errors": error_count, "total": total_operations}
            )

        return metrics

    async def evaluate_performance(self, context: Optional[Dict[str, Any]] = None) -> PerformanceReport:
        """
        Evalúa el rendimiento general del sistema

        Args:
            context: Contexto adicional para la evaluación

        Returns:
            Reporte completo de rendimiento
        """
        context = context or {}
        timestamp = time.time()

        try:
            # Recopilar todas las métricas recientes
            all_metrics = {}
            recommendations = []
            alerts = []

            # Analizar cada tipo de métrica
            for metric_name, history in self.metrics_history.items():
                if not history:
                    continue

                recent_values = [m.value for m in history[-10:]]  # últimas 10 mediciones
                recent_confidences = [m.confidence for m in history[-10:]]

                if not recent_values:
                    continue

                # Calcular estadísticas
                current_value = recent_values[-1]
                trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0] if len(recent_values) > 1 else 0
                avg_confidence = np.mean(recent_confidences)

                # Crear resultado de evaluación
                result = EvaluationResult(
                    MetricType(metric_name) if metric_name in [m.value for m in MetricType] else MetricType.CONFIDENCE,
                    current_value, avg_confidence, timestamp, context
                )

                all_metrics[metric_name] = result

                # Generar recomendaciones y alertas
                baseline = self.performance_baseline.get(metric_name, 0.5)

                if current_value < baseline * 0.8:
                    alerts.append(f"🔴 {metric_name} significativamente por debajo del baseline ({current_value:.3f} < {baseline:.3f})")
                elif current_value < baseline:
                    recommendations.append(f"⚠️ {metric_name} por debajo del baseline, considerar optimización")

                if trend < -0.01:  # tendencia negativa
                    recommendations.append(f"📉 {metric_name} muestra tendencia negativa, monitorear de cerca")

            # Calcular puntuación general
            if all_metrics:
                # Puntuación ponderada basada en importancia de métricas
                weights = {
                    "accuracy": 0.3,
                    "f1_score": 0.25,
                    "confidence_mean": 0.2,
                    "latency": -0.15,  # negativo porque menor es mejor
                    "error_rate": -0.1   # negativo
                }

                weighted_sum = 0
                total_weight = 0

                for metric_name, result in all_metrics.items():
                    if metric_name in weights:
                        weight = weights[metric_name]
                        # Normalizar latencia y error rate (menor es mejor)
                        if "latency" in metric_name or "error_rate" in metric_name:
                            normalized_value = 1.0 - min(result.value, 1.0)  # invertir escala
                        else:
                            normalized_value = result.value

                        weighted_sum += weight * normalized_value
                        total_weight += abs(weight)

                overall_score = weighted_sum / total_weight if total_weight > 0 else 0.5
                overall_score = np.clip(overall_score, 0, 1)
            else:
                overall_score = 0.5

            # Actualizar métricas críticas
            self.critical_metrics["system_confidence"] = overall_score
            if "error_rate" in all_metrics:
                self.critical_metrics["error_rate"] = all_metrics["error_rate"].value
            if "resource_usage" in all_metrics:
                self.critical_metrics["resource_efficiency"] = 1.0 - all_metrics["resource_usage"].value

            # Calcular coherencia (simplificada)
            coherence_score = self._calculate_coherence_score(all_metrics)
            self.critical_metrics["coherence_score"] = coherence_score

            report = PerformanceReport(
                overall_score=overall_score,
                metrics=all_metrics,
                recommendations=recommendations,
                alerts=alerts,
                timestamp=timestamp
            )

            # Emitir evento de evaluación
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "Evaluator",
                outputs={
                    "overall_score": overall_score,
                    "recommendations_count": len(recommendations),
                    "alerts_count": len(alerts)
                },
                context={"metrics_count": len(all_metrics)},
                explanation="Evaluación de rendimiento completada"
            )

            return report

        except Exception as e:
            # Reporte de error
            error_report = PerformanceReport(
                overall_score=0.0,
                metrics={},
                recommendations=["Investigar error en evaluación"],
                alerts=[f"Error en evaluación: {str(e)}"],
                timestamp=timestamp
            )

            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "Evaluator",
                outputs={"evaluation_error": str(e)},
                log_level=2
            )

            return error_report

    def _calculate_coherence_score(self, metrics: Dict[str, EvaluationResult]) -> float:
        """Calcula puntuación de coherencia entre métricas"""
        if len(metrics) < 2:
            return 0.5

        # Calcular correlación entre tendencias de métricas
        values_lists = []
        for result in metrics.values():
            if len(self.metrics_history.get(result.metric_type.value, [])) >= 5:
                recent = [m.value for m in self.metrics_history[result.metric_type.value][-5:]]
                values_lists.append(recent)

        if len(values_lists) < 2:
            return 0.5

        # Calcular coherencia basada en correlaciones
        correlations = []
        for i in range(len(values_lists)):
            for j in range(i + 1, len(values_lists)):
                if len(values_lists[i]) == len(values_lists[j]):
                    corr = np.corrcoef(values_lists[i], values_lists[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))  # usar valor absoluto

        if correlations:
            avg_correlation = np.mean(correlations)
            # Coherencia: alta correlación = alta coherencia
            coherence = 0.5 + 0.5 * avg_correlation
        else:
            coherence = 0.5

        return coherence

    def get_confidence_score(self) -> float:
        """Obtiene la puntuación de confianza general del sistema"""
        return self.critical_metrics["system_confidence"]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de rendimiento"""
        return {
            "overall_score": self.critical_metrics["system_confidence"],
            "error_rate": self.critical_metrics["error_rate"],
            "resource_efficiency": self.critical_metrics["resource_efficiency"],
            "coherence_score": self.critical_metrics["coherence_score"],
            "metrics_tracked": len(self.metrics_history),
            "total_evaluations": sum(len(history) for history in self.metrics_history.values()),
            "last_evaluation": self.last_evaluation
        }

    def get_detailed_metrics(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """Obtiene métricas detalladas"""
        if metric_name:
            if metric_name in self.metrics_history:
                history = self.metrics_history[metric_name]
                values = [m.value for m in history]
                return {
                    "metric": metric_name,
                    "current_value": values[-1] if values else None,
                    "mean": np.mean(values) if values else None,
                    "std": np.std(values) if values else None,
                    "trend": np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0,
                    "history_length": len(history)
                }
            else:
                return {"error": f"Métrica '{metric_name}' no encontrada"}
        else:
            # Resumen de todas las métricas
            summary = {}
            for name, history in self.metrics_history.items():
                if history:
                    values = [m.value for m in history]
                    summary[name] = {
                        "current": values[-1],
                        "mean": np.mean(values),
                        "count": len(history)
                    }
            return summary