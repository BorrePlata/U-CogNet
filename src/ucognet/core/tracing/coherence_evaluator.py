"""
Coherence & Ethics Evaluator
Evalúa la coherencia interna y alineación ética de las trazas cognitivas
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import statistics
from datetime import datetime, timedelta

from .cognitive_event import CognitiveEvent, EventType

class CoherenceMetrics:
    """Métricas de coherencia del sistema"""

    def __init__(self):
        self.temporal_coherence = 0.0  # Consistencia temporal de decisiones
        self.inter_module_coherence = 0.0  # Coherencia entre módulos
        self.narrative_coherence = 0.0  # Coherencia narrativa del episodio
        self.stability_score = 0.0  # Estabilidad del sistema
        self.adaptability_index = 0.0  # Capacidad de adaptación

class EthicsEvaluation:
    """Evaluación ética del comportamiento"""

    def __init__(self):
        self.alignment_score = 0.0  # Alineación con objetivos del sistema
        self.safety_compliance = 0.0  # Cumplimiento de reglas de seguridad
        self.resource_efficiency = 0.0  # Eficiencia en uso de recursos
        self.transparency_level = 0.0  # Nivel de transparencia
        self.bias_detection = 0.0  # Detección de sesgos

class CoherenceEthicsEvaluator:
    """
    Evaluador de coherencia cognitiva y ética.
    Analiza trazas para detectar inconsistencias, alineación ética y estabilidad.
    """

    def __init__(self):
        # Reglas de coherencia
        self.coherence_rules = {
            'temporal_consistency': self._check_temporal_consistency,
            'decision_stability': self._check_decision_stability,
            'module_alignment': self._check_module_alignment,
            'resource_efficiency': self._check_resource_efficiency,
            'safety_compliance': self._check_safety_compliance
        }

        # Reglas éticas
        self.ethics_rules = {
            'objective_alignment': self._check_objective_alignment,
            'safety_boundaries': self._check_safety_boundaries,
            'transparency_requirement': self._check_transparency,
            'bias_detection': self._check_bias_detection,
            'resource_stewardship': self._check_resource_stewardship
        }

        # Umbrales de evaluación
        self.thresholds = {
            'high_coherence': 0.8,
            'medium_coherence': 0.6,
            'low_coherence': 0.4,
            'critical_coherence': 0.2
        }

    def evaluate_episode(self, events: List[CognitiveEvent]) -> Tuple[CoherenceMetrics, EthicsEvaluation]:
        """
        Evalúa un episodio completo de eventos.

        Returns:
            Tuple de métricas de coherencia y evaluación ética
        """
        if not events:
            return CoherenceMetrics(), EthicsEvaluation()

        # Ordenar eventos por tiempo
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        # Evaluar coherencia
        coherence = self._evaluate_coherence(sorted_events)

        # Evaluar ética
        ethics = self._evaluate_ethics(sorted_events)

        return coherence, ethics

    def _evaluate_coherence(self, events: List[CognitiveEvent]) -> CoherenceMetrics:
        """Evalúa métricas de coherencia"""
        metrics = CoherenceMetrics()

        # Calcular cada aspecto de coherencia
        metrics.temporal_coherence = self._calculate_temporal_coherence(events)
        metrics.inter_module_coherence = self._calculate_inter_module_coherence(events)
        metrics.narrative_coherence = self._calculate_narrative_coherence(events)
        metrics.stability_score = self._calculate_stability_score(events)
        metrics.adaptability_index = self._calculate_adaptability_index(events)

        return metrics

    def _evaluate_ethics(self, events: List[CognitiveEvent]) -> EthicsEvaluation:
        """Evalúa aspectos éticos"""
        evaluation = EthicsEvaluation()

        # Calcular cada aspecto ético
        evaluation.alignment_score = self._calculate_alignment_score(events)
        evaluation.safety_compliance = self._calculate_safety_compliance(events)
        evaluation.resource_efficiency = self._calculate_resource_efficiency(events)
        evaluation.transparency_level = self._calculate_transparency_level(events)
        evaluation.bias_detection = self._calculate_bias_detection(events)

        return evaluation

    def _calculate_temporal_coherence(self, events: List[CognitiveEvent]) -> float:
        """Calcula coherencia temporal: consistencia de decisiones en el tiempo"""
        decisions = [e for e in events if e.event_type == EventType.DECISION]

        if len(decisions) < 2:
            return 1.0  # Perfecta coherencia si hay pocas decisiones

        # Analizar consistencia de decisiones similares
        decision_patterns = defaultdict(list)

        for decision in decisions:
            key = f"{decision.source_module}_{decision.outputs.get('decision', 'unknown')}"
            decision_patterns[key].append(decision.timestamp)

        # Calcular variabilidad temporal
        consistency_scores = []

        for pattern, timestamps in decision_patterns.items():
            if len(timestamps) > 1:
                # Calcular intervalos entre decisiones similares
                intervals = []
                for i in range(1, len(timestamps)):
                    interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                    intervals.append(interval)

                if intervals:
                    # Menor variabilidad = mayor coherencia
                    if len(intervals) > 1:
                        mean_interval = statistics.mean(intervals)
                        std_interval = statistics.stdev(intervals)
                        cv = std_interval / mean_interval if mean_interval > 0 else 0
                        consistency = max(0, 1 - cv)  # Coeficiente de variación invertido
                        consistency_scores.append(consistency)
                    else:
                        consistency_scores.append(1.0)

        return statistics.mean(consistency_scores) if consistency_scores else 0.8

    def _calculate_inter_module_coherence(self, events: List[CognitiveEvent]) -> float:
        """Calcula coherencia entre módulos"""
        # Agrupar eventos por módulo
        module_events = defaultdict(list)
        for event in events:
            module_events[event.source_module].append(event)

        if len(module_events) < 2:
            return 1.0

        # Analizar interacciones entre módulos
        interactions = [e for e in events if e.event_type == EventType.MODULE_INTERACTION]

        coherence_scores = []

        for interaction in interactions:
            target_module = interaction.outputs.get('target_module')
            interaction_type = interaction.outputs.get('interaction_type')

            # Verificar si el módulo objetivo existe y responde coherentemente
            target_events = module_events.get(target_module, [])
            relevant_responses = [
                e for e in target_events
                if e.timestamp > interaction.timestamp and
                (e.timestamp - interaction.timestamp).total_seconds() < 5  # 5 segundos de ventana
            ]

            if relevant_responses:
                # Hay respuesta coherente
                coherence_scores.append(1.0)
            else:
                # No hay respuesta o es incoherente
                coherence_scores.append(0.3)

        return statistics.mean(coherence_scores) if coherence_scores else 0.7

    def _calculate_narrative_coherence(self, events: List[CognitiveEvent]) -> float:
        """Calcula coherencia narrativa del episodio"""
        # Buscar patrón causa-efecto
        rewards = [e for e in events if e.event_type == EventType.REWARD]
        decisions = [e for e in events if e.event_type == EventType.DECISION]

        if not rewards or not decisions:
            return 0.5

        # Para cada reward, verificar si hay decisiones previas razonables
        narrative_scores = []

        for reward in rewards:
            reward_time = reward.timestamp
            reward_value = reward.outputs.get('reward', 0)

            # Buscar decisiones en los 10 segundos previos
            relevant_decisions = [
                d for d in decisions
                if 0 < (reward_time - d.timestamp).total_seconds() < 10
            ]

            if relevant_decisions:
                # Verificar si las decisiones son consistentes con el reward
                decision_quality = statistics.mean([
                    d.metrics.get('decision_quality', 0.5) for d in relevant_decisions
                ])

                # Reward positivo con buenas decisiones = coherencia alta
                if reward_value > 0 and decision_quality > 0.6:
                    narrative_scores.append(0.9)
                elif reward_value < 0 and decision_quality < 0.4:
                    narrative_scores.append(0.8)
                else:
                    narrative_scores.append(0.6)
            else:
                # Reward sin decisiones previas = incoherente
                narrative_scores.append(0.3)

        return statistics.mean(narrative_scores) if narrative_scores else 0.5

    def _calculate_stability_score(self, events: List[CognitiveEvent]) -> float:
        """Calcula estabilidad del sistema"""
        # Analizar cambios de configuración y su impacto
        config_changes = [e for e in events if e.event_type in [EventType.GATING_CHANGE, EventType.TOPOLOGY_CHANGE]]

        if not config_changes:
            return 1.0  # Estable si no hay cambios

        # Evaluar impacto de cambios
        stability_scores = []

        for change in config_changes:
            # Buscar métricas de evaluación después del cambio
            post_change_events = [
                e for e in events
                if e.timestamp > change.timestamp and
                e.event_type == EventType.EVALUATION_METRIC and
                (e.timestamp - change.timestamp).total_seconds() < 30  # 30 segundos después
            ]

            if post_change_events:
                # Verificar si las métricas se mantienen estables
                pre_metrics = change.context.get('pre_change_metrics', {})
                post_metrics = post_change_events[0].metrics

                # Comparar estabilidad (simplificado)
                stability = 0.8  # Asumir razonablemente estable
                for key in ['precision', 'recall', 'f1_score']:
                    pre_val = pre_metrics.get(key, 0.5)
                    post_val = post_metrics.get(key, 0.5)
                    diff = abs(post_val - pre_val)
                    if diff > 0.2:  # Cambio significativo
                        stability -= 0.2

                stability_scores.append(max(0, stability))
            else:
                stability_scores.append(0.5)  # No hay datos post-cambio

        return statistics.mean(stability_scores) if stability_scores else 0.8

    def _calculate_adaptability_index(self, events: List[CognitiveEvent]) -> float:
        """Calcula índice de adaptabilidad"""
        # Medir capacidad de respuesta a cambios
        learning_events = [e for e in events if e.event_type == EventType.LEARNING_STEP]
        performance_events = [e for e in events if e.event_type == EventType.EVALUATION_METRIC]

        if not learning_events or not performance_events:
            return 0.5

        # Verificar mejora en rendimiento después de aprendizaje
        adaptability_scores = []

        for learn_event in learning_events:
            # Buscar mejoras en rendimiento posteriores
            post_performance = [
                p for p in performance_events
                if p.timestamp > learn_event.timestamp and
                (p.timestamp - learn_event.timestamp).total_seconds() < 60  # 1 minuto después
            ]

            if post_performance:
                # Comparar con rendimiento anterior
                pre_performance = [
                    p for p in performance_events
                    if p.timestamp < learn_event.timestamp
                ]

                if pre_performance:
                    pre_avg = statistics.mean([p.metrics.get('f1_score', 0.5) for p in pre_performance[-3:]])
                    post_avg = statistics.mean([p.metrics.get('f1_score', 0.5) for p in post_performance[:3]])

                    if post_avg > pre_avg:
                        adaptability_scores.append(0.9)  # Mejora = buena adaptabilidad
                    else:
                        adaptability_scores.append(0.6)  # No mejora pero intentó
                else:
                    adaptability_scores.append(0.7)  # Primeros eventos de aprendizaje

        return statistics.mean(adaptability_scores) if adaptability_scores else 0.6

    # Métodos de evaluación ética

    def _calculate_alignment_score(self, events: List[CognitiveEvent]) -> float:
        """Calcula alineación con objetivos del sistema"""
        # Verificar que las acciones sirvan a los objetivos declarados
        decisions = [e for e in events if e.event_type == EventType.DECISION]

        alignment_scores = []

        for decision in decisions:
            context = decision.context
            decision_quality = decision.metrics.get('decision_quality', 0.5)

            # Verificar coherencia con contexto
            if context.get('objective_alignment', True):
                alignment_scores.append(decision_quality)
            else:
                alignment_scores.append(0.3)  # Desalineado

        return statistics.mean(alignment_scores) if alignment_scores else 0.7

    def _calculate_safety_compliance(self, events: List[CognitiveEvent]) -> float:
        """Calcula cumplimiento de reglas de seguridad"""
        security_events = [e for e in events if e.event_type == EventType.SECURITY_CHECK]

        if not security_events:
            return 0.8  # Asumir cumplimiento si no hay checks

        compliance_scores = []

        for security_event in security_events:
            result = security_event.outputs.get('result', True)
            check_type = security_event.outputs.get('check_type', 'unknown')

            if result:
                compliance_scores.append(1.0)
            else:
                # Penalizar más checks críticos
                if 'critical' in check_type.lower():
                    compliance_scores.append(0.1)
                else:
                    compliance_scores.append(0.5)

        return statistics.mean(compliance_scores)

    def _calculate_resource_efficiency(self, events: List[CognitiveEvent]) -> float:
        """Calcula eficiencia en uso de recursos"""
        # Analizar métricas de rendimiento vs recursos usados
        eval_events = [e for e in events if e.event_type == EventType.EVALUATION_METRIC]

        efficiency_scores = []

        for eval_event in eval_events:
            metrics = eval_event.metrics
            performance = metrics.get('f1_score', 0.5)
            latency = metrics.get('latency_ms', 100)
            throughput = metrics.get('throughput_fps', 10)

            # Eficiencia = rendimiento / (latencia + 1/throughput)
            efficiency = performance / (latency/1000 + 1/max(throughput, 0.1))
            efficiency_scores.append(min(1.0, efficiency * 100))  # Normalizar

        return statistics.mean(efficiency_scores) if efficiency_scores else 0.7

    def _calculate_transparency_level(self, events: List[CognitiveEvent]) -> float:
        """Calcula nivel de transparencia"""
        # Medir completitud de explicaciones y trazas
        total_events = len(events)
        explained_events = sum(1 for e in events if e.explanation)

        if total_events == 0:
            return 0.0

        transparency_ratio = explained_events / total_events

        # Bonus por calidad de explicaciones
        detailed_explanations = sum(1 for e in events if e.explanation and len(e.explanation) > 20)

        return min(1.0, transparency_ratio + (detailed_explanations / total_events) * 0.2)

    def _calculate_bias_detection(self, events: List[CognitiveEvent]) -> float:
        """Calcula capacidad de detección de sesgos"""
        # Buscar patrones de sesgo en decisiones
        decisions = [e for e in events if e.event_type == EventType.DECISION]

        if len(decisions) < 10:
            return 0.5  # Necesita más datos

        # Analizar distribución de decisiones
        decision_values = [d.outputs.get('decision', 0) for d in decisions if isinstance(d.outputs.get('decision'), (int, float))]

        if len(decision_values) < 5:
            return 0.5

        # Verificar sesgos estadísticos
        mean_decision = statistics.mean(decision_values)
        std_decision = statistics.stdev(decision_values) if len(decision_values) > 1 else 0

        # Sesgo bajo = buena detección de sesgos (ironía)
        bias_score = min(1.0, std_decision / abs(mean_decision + 0.1))

        return 1.0 - bias_score  # Invertir: baja variabilidad = alto sesgo detectado

    def get_overall_coherence_score(self, coherence: CoherenceMetrics) -> float:
        """Calcula score general de coherencia"""
        weights = {
            'temporal': 0.25,
            'inter_module': 0.25,
            'narrative': 0.2,
            'stability': 0.15,
            'adaptability': 0.15
        }

        return (
            coherence.temporal_coherence * weights['temporal'] +
            coherence.inter_module_coherence * weights['inter_module'] +
            coherence.narrative_coherence * weights['narrative'] +
            coherence.stability_score * weights['stability'] +
            coherence.adaptability_index * weights['adaptability']
        )

    def get_overall_ethics_score(self, ethics: EthicsEvaluation) -> float:
        """Calcula score general de ética"""
        weights = {
            'alignment': 0.3,
            'safety': 0.3,
            'efficiency': 0.15,
            'transparency': 0.15,
            'bias_detection': 0.1
        }

        return (
            ethics.alignment_score * weights['alignment'] +
            ethics.safety_compliance * weights['safety'] +
            ethics.resource_efficiency * weights['efficiency'] +
            ethics.transparency_level * weights['transparency'] +
            ethics.bias_detection * weights['bias_detection']
        )

    def get_coherence_level(self, score: float) -> str:
        """Clasifica nivel de coherencia"""
        if score >= self.thresholds['high_coherence']:
            return "HIGH"
        elif score >= self.thresholds['medium_coherence']:
            return "MEDIUM"
        elif score >= self.thresholds['low_coherence']:
            return "LOW"
        else:
            return "CRITICAL"

    def get_recommendations(self, coherence: CoherenceMetrics, ethics: EthicsEvaluation) -> List[str]:
        """Genera recomendaciones basadas en evaluación"""
        recommendations = []

        # Recomendaciones de coherencia
        if coherence.temporal_coherence < 0.6:
            recommendations.append("Mejorar consistencia temporal de decisiones - reducir oscilaciones")

        if coherence.inter_module_coherence < 0.6:
            recommendations.append("Aumentar comunicación y alineación entre módulos")

        if coherence.stability_score < 0.6:
            recommendations.append("Revisar estabilidad del sistema - cambios de configuración muy frecuentes")

        # Recomendaciones éticas
        if ethics.safety_compliance < 0.8:
            recommendations.append("Mejorar cumplimiento de reglas de seguridad")

        if ethics.transparency_level < 0.7:
            recommendations.append("Aumentar transparencia - agregar más explicaciones a decisiones")

        if ethics.bias_detection < 0.6:
            recommendations.append("Implementar mejor detección y corrección de sesgos")

        return recommendations

    # Métodos stub para reglas de coherencia
    def _check_temporal_consistency(self, events: List[CognitiveEvent]) -> float:
        """Verifica consistencia temporal de decisiones"""
        return 0.8  # Stub implementation

    def _check_decision_stability(self, events: List[CognitiveEvent]) -> float:
        """Verifica estabilidad de decisiones"""
        return 0.7  # Stub implementation

    def _check_module_alignment(self, events: List[CognitiveEvent]) -> float:
        """Verifica alineación entre módulos"""
        return 0.9  # Stub implementation

    def _check_resource_efficiency(self, events: List[CognitiveEvent]) -> float:
        """Verifica eficiencia en uso de recursos"""
        return 0.6  # Stub implementation

    def _check_safety_compliance(self, events: List[CognitiveEvent]) -> float:
        """Verifica cumplimiento de reglas de seguridad"""
        return 0.8  # Stub implementation

    # Métodos stub para reglas éticas
    def _check_objective_alignment(self, events: List[CognitiveEvent]) -> float:
        """Verifica alineación con objetivos"""
        return 0.9  # Stub implementation

    def _check_safety_boundaries(self, events: List[CognitiveEvent]) -> float:
        """Verifica límites de seguridad"""
        return 0.8  # Stub implementation

    def _check_transparency(self, events: List[CognitiveEvent]) -> float:
        """Verifica transparencia"""
        return 0.7  # Stub implementation

    def _check_bias_detection(self, events: List[CognitiveEvent]) -> float:
        """Verifica detección de sesgos"""
        return 0.6  # Stub implementation

    def _check_resource_stewardship(self, events: List[CognitiveEvent]) -> float:
        """Verifica administración de recursos"""
        return 0.8  # Stub implementation