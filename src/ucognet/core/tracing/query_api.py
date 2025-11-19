"""
Trace Query API
API para consultar y analizar trazas cognitivas
"""

from typing import Dict, List, Any, Optional, Callable, Iterator, Tuple
import json
from datetime import datetime, timedelta
from collections import defaultdict
import re

from .cognitive_event import CognitiveEvent, EventType, LogLevel
from .trace_core import CognitiveTraceCore
from .causal_builder import CausalGraphBuilder
from .coherence_evaluator import CoherenceEthicsEvaluator, CoherenceMetrics, EthicsEvaluation

class QueryFilter:
    """Filtro para consultas de eventos"""

    def __init__(self,
                 event_types: Optional[List[EventType]] = None,
                 source_modules: Optional[List[str]] = None,
                 episode_ids: Optional[List[str]] = None,
                 time_range: Optional[Tuple[datetime, datetime]] = None,
                 log_levels: Optional[List[LogLevel]] = None,
                 custom_filter: Optional[Callable[[CognitiveEvent], bool]] = None):
        self.event_types = event_types or []
        self.source_modules = source_modules or []
        self.episode_ids = episode_ids or []
        self.time_range = time_range
        self.log_levels = log_levels or []
        self.custom_filter = custom_filter

    def matches(self, event: CognitiveEvent) -> bool:
        """Verifica si un evento cumple con el filtro"""
        # Tipo de evento
        if self.event_types and event.event_type not in self.event_types:
            return False

        # Módulo fuente
        if self.source_modules and event.source_module not in self.source_modules:
            return False

        # Episode ID
        if self.episode_ids and event.episode_id not in self.episode_ids:
            return False

        # Rango temporal
        if self.time_range:
            start_time, end_time = self.time_range
            if not (start_time <= event.timestamp <= end_time):
                return False

        # Nivel de log
        if self.log_levels and event.log_level not in self.log_levels:
            return False

        # Filtro personalizado
        if self.custom_filter and not self.custom_filter(event):
            return False

        return True

class TraceQueryAPI:
    """
    API para consultar y analizar trazas cognitivas.
    Proporciona interfaz unificada para explorar el comportamiento del sistema.
    """

    def __init__(self, trace_core: CognitiveTraceCore,
                 causal_builder: Optional[CausalGraphBuilder] = None,
                 coherence_evaluator: Optional[CoherenceEthicsEvaluator] = None):
        self.trace_core = trace_core
        self.causal_builder = causal_builder or CausalGraphBuilder()
        self.coherence_evaluator = coherence_evaluator or CoherenceEthicsEvaluator()

    def get_episode_events(self, episode_id: str, filter: Optional[QueryFilter] = None) -> List[CognitiveEvent]:
        """Obtiene todos los eventos de un episodio"""
        events = self.trace_core.get_episode_events(episode_id)

        if filter:
            events = [e for e in events if filter.matches(e)]

        return sorted(events, key=lambda e: e.timestamp)

    def find_episodes(self, filter: QueryFilter, limit: int = 50) -> List[str]:
        """Encuentra IDs de episodios que cumplen con el filtro"""
        episode_events = defaultdict(list)

        # Obtener eventos que cumplen el filtro
        for event in self._iterate_events(filter):
            if event.episode_id:
                episode_events[event.episode_id].append(event)

        # Ordenar por cantidad de eventos (más activos primero)
        sorted_episodes = sorted(
            episode_events.keys(),
            key=lambda eid: len(episode_events[eid]),
            reverse=True
        )

        return sorted_episodes[:limit]

    def get_event_timeline(self, episode_id: str, event_types: Optional[List[EventType]] = None) -> List[Dict[str, Any]]:
        """Obtiene línea de tiempo de eventos para un episodio"""
        filter = QueryFilter(episode_ids=[episode_id], event_types=event_types)
        events = self.get_episode_events(episode_id, filter)

        timeline = []
        for event in events:
            timeline.append({
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type.value,
                'source_module': event.source_module,
                'description': event.explanation or f"{event.event_type.value} from {event.source_module}",
                'metrics': event.metrics,
                'outputs': event.outputs
            })

        return timeline

    def get_episode_summary(self, episode_id: str) -> Dict[str, Any]:
        """Obtiene resumen estadístico de un episodio"""
        events = self.get_episode_events(episode_id)

        if not events:
            return {'episode_id': episode_id, 'error': 'Episode not found'}

        start_time = min(e.timestamp for e in events)
        end_time = max(e.timestamp for e in events)
        duration = (end_time - start_time).total_seconds()

        # Estadísticas por tipo de evento
        event_counts = defaultdict(int)
        module_counts = defaultdict(int)
        total_rewards = 0
        total_decisions = 0

        for event in events:
            event_counts[event.event_type.value] += 1
            module_counts[event.source_module] += 1

            if event.event_type == EventType.REWARD:
                total_rewards += event.outputs.get('reward', 0)
            elif event.event_type == EventType.DECISION:
                total_decisions += 1

        # Evaluar coherencia y ética
        coherence, ethics = self.coherence_evaluator.evaluate_episode(events)

        return {
            'episode_id': episode_id,
            'duration_seconds': duration,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_events': len(events),
            'event_counts': dict(event_counts),
            'module_counts': dict(module_counts),
            'total_rewards': total_rewards,
            'total_decisions': total_decisions,
            'coherence_score': self.coherence_evaluator.get_overall_coherence_score(coherence),
            'ethics_score': self.coherence_evaluator.get_overall_ethics_score(ethics),
            'coherence_level': self.coherence_evaluator.get_coherence_level(
                self.coherence_evaluator.get_overall_coherence_score(coherence)
            ),
            'recommendations': self.coherence_evaluator.get_recommendations(coherence, ethics)
        }

    def get_causal_analysis(self, episode_id: str) -> Dict[str, Any]:
        """Obtiene análisis causal de un episodio"""
        events = self.get_episode_events(episode_id)

        if not events or not self.causal_builder:
            return {'episode_id': episode_id, 'error': 'No causal analysis available'}

        # Agregar eventos al constructor causal
        temp_builder = CausalGraphBuilder()
        temp_builder.add_events_batch(events)

        # Análisis del episodio
        analysis = temp_builder.analyze_episode(episode_id)

        # Encontrar causas raíz de eventos importantes
        important_events = [
            e for e in events
            if e.event_type in [EventType.REWARD, EventType.EVALUATION_METRIC]
        ]

        root_causes = []
        for event in important_events[:5]:  # Analizar top 5 eventos importantes
            causes = temp_builder.get_root_causes(event.event_id)
            root_causes.extend([{
                'target_event': event.event_id,
                'target_type': event.event_type.value,
                'root_cause': cause.event_id,
                'cause_type': cause.event_type.value,
                'cause_module': cause.source_module,
                'explanation': cause.explanation
            } for cause in causes])

        return {
            'episode_id': episode_id,
            **analysis,
            'root_causes': root_causes,
            'graph_summary': temp_builder.get_graph_summary()
        }

    def search_events(self, query: str, filter: Optional[QueryFilter] = None, limit: int = 100) -> List[CognitiveEvent]:
        """Busca eventos usando consulta de texto"""
        # Implementar búsqueda simple por texto
        def text_matches(event: CognitiveEvent) -> bool:
            search_text = f"{event.explanation or ''} {event.source_module} {event.event_type.value}"
            return query.lower() in search_text.lower()

        # Combinar con filtro existente
        combined_filter = filter or QueryFilter()
        original_custom = combined_filter.custom_filter

        def combined_custom(event: CognitiveEvent) -> bool:
            return text_matches(event) and (original_custom is None or original_custom(event))

        combined_filter.custom_filter = combined_custom

        # Buscar eventos
        matching_events = list(self._iterate_events(combined_filter))

        # Ordenar por relevancia (simplificado: por tiempo reciente)
        matching_events.sort(key=lambda e: e.timestamp, reverse=True)

        return matching_events[:limit]

    def get_system_health_report(self, hours: int = 24) -> Dict[str, Any]:
        """Genera reporte de salud del sistema basado en trazas"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Obtener eventos recientes
        recent_filter = QueryFilter(time_range=(cutoff_time, datetime.now()))
        recent_events = list(self._iterate_events(recent_filter))

        if not recent_events:
            return {'error': 'No recent events found'}

        # Análisis de salud
        health_report = {
            'time_period_hours': hours,
            'total_events': len(recent_events),
            'event_rate_per_hour': len(recent_events) / hours,
            'modules_active': len(set(e.source_module for e in recent_events)),
            'episodes_active': len(set(e.episode_id for e in recent_events if e.episode_id)),
        }

        # Análisis por tipo de evento
        event_types = defaultdict(int)
        error_events = []
        performance_metrics = []

        for event in recent_events:
            event_types[event.event_type.value] += 1

            # Detectar errores
            if event.outputs.get('success') is False or 'error' in event.outputs:
                error_events.append(event)

            # Recopilar métricas de rendimiento
            if event.metrics:
                performance_metrics.append(event.metrics)

        health_report['event_types'] = dict(event_types)
        health_report['error_count'] = len(error_events)
        health_report['error_rate'] = len(error_events) / len(recent_events) if recent_events else 0

        # Métricas de rendimiento promedio
        if performance_metrics:
            avg_metrics = {}
            for key in ['execution_time_ms', 'latency_ms', 'throughput_fps']:
                values = [m.get(key) for m in performance_metrics if key in m]
                if values:
                    avg_metrics[f'avg_{key}'] = sum(values) / len(values)

            health_report['average_performance'] = avg_metrics

        # Evaluar coherencia general
        if len(recent_events) > 10:
            coherence, ethics = self.coherence_evaluator.evaluate_episode(recent_events)
            health_report['overall_coherence'] = self.coherence_evaluator.get_overall_coherence_score(coherence)
            health_report['overall_ethics'] = self.coherence_evaluator.get_overall_ethics_score(ethics)
            health_report['coherence_level'] = self.coherence_evaluator.get_coherence_level(
                health_report['overall_coherence']
            )

        return health_report

    def export_episode_data(self, episode_id: str, filepath: str, format: str = 'json') -> None:
        """Exporta datos completos de un episodio"""
        events = self.get_episode_events(episode_id)
        summary = self.get_episode_summary(episode_id)
        causal_analysis = self.get_causal_analysis(episode_id)

        export_data = {
            'episode_id': episode_id,
            'export_timestamp': datetime.now().isoformat(),
            'summary': summary,
            'causal_analysis': causal_analysis,
            'events': [event.to_dict() for event in events]
        }

        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        else:
            # Formato simplificado para otros usos
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Episode {episode_id} Export\n")
                f.write(f"Generated: {datetime.now()}\n\n")
                f.write(f"Summary: {summary}\n\n")
                f.write("Events:\n")
                for event in events:
                    f.write(f"- {event.timestamp}: {event.event_type.value} from {event.source_module}\n")

    def _iterate_events(self, filter: Optional[QueryFilter] = None) -> Iterator[CognitiveEvent]:
        """Itera sobre eventos aplicando filtro"""
        # Obtener eventos recientes (últimos 1000 para eficiencia)
        recent_events = self.trace_core.get_recent_events(1000)

        for event in recent_events:
            if filter is None or filter.matches(event):
                yield event

    def get_query_templates(self) -> Dict[str, QueryFilter]:
        """Devuelve plantillas de consultas comunes"""
        return {
            'errors_only': QueryFilter(
                custom_filter=lambda e: e.outputs.get('success') is False or 'error' in e.outputs
            ),
            'decisions_only': QueryFilter(event_types=[EventType.DECISION]),
            'rewards_only': QueryFilter(event_types=[EventType.REWARD]),
            'security_events': QueryFilter(event_types=[EventType.SECURITY_CHECK]),
            'learning_events': QueryFilter(event_types=[EventType.LEARNING_STEP]),
            'high_importance': QueryFilter(log_levels=[LogLevel.INFO, LogLevel.SUMMARY])
        }