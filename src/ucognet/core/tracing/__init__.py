"""
Módulo de Trazabilidad Cognitiva para U-CogNet
Implementa el Meta-módulo de Trazabilidad Cognitiva (MTC) completo
"""

from .cognitive_event import CognitiveEvent, CognitiveEventSchema, EventType, LogLevel
from .trace_core import CognitiveTraceCore
from .event_bus import CognitiveEventBus, emit_event, get_event_bus
from .causal_builder import CausalGraphBuilder, CausalLink
from .coherence_evaluator import CoherenceEthicsEvaluator, CoherenceMetrics, EthicsEvaluation
from .query_api import TraceQueryAPI, QueryFilter
from .storage_manager import TraceStorageManager, StorageConfig

__all__ = [
    # Eventos
    'CognitiveEvent',
    'CognitiveEventSchema',
    'EventType',
    'LogLevel',

    # Núcleo
    'CognitiveTraceCore',

    # Bus de eventos
    'CognitiveEventBus',
    'emit_event',
    'get_event_bus',

    # Análisis causal
    'CausalGraphBuilder',
    'CausalLink',

    # Evaluación
    'CoherenceEthicsEvaluator',
    'CoherenceMetrics',
    'EthicsEvaluation',

    # Consultas
    'TraceQueryAPI',
    'QueryFilter',

    # Almacenamiento
    'TraceStorageManager',
    'StorageConfig'
]