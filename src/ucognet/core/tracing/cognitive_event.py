"""
Esquema de Evento Cognitivo para U-CogNet
Define la estructura unificada de eventos que todos los módulos deben usar
"""

import uuid
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
from enum import Enum

class EventType(Enum):
    """Tipos de eventos cognitivos permitidos"""
    DECISION = "decision"
    UPDATE = "update"
    REWARD = "reward"
    GATING_CHANGE = "gating_change"
    SECURITY_CHECK = "security_check"
    TOPOLOGY_CHANGE = "topology_change"
    LEARNING_STEP = "learning_step"
    EVALUATION_METRIC = "evaluation_metric"
    MODULE_INTERACTION = "module_interaction"
    SYSTEM_STATE = "system_state"

class LogLevel(Enum):
    """Niveles de detalle para trazabilidad"""
    TRACE = "trace"      # Todo, incluyendo estados internos
    DEBUG = "debug"      # Información de desarrollo
    INFO = "info"        # Eventos importantes
    SUMMARY = "summary"  # Solo resúmenes agregados

@dataclass
class CognitiveEventSchema:
    """
    Esquema unificado para eventos cognitivos en U-CogNet.
    Todos los módulos deben usar este esquema para emitir eventos.
    """

    # Origen y tipo (requeridos)
    source_module: str  # "CognitiveCore", "Evaluator", "TDAManager", etc.
    event_type: EventType

    # Identificación única
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: Optional[str] = None

    # Temporal
    timestamp: datetime = field(default_factory=datetime.now)

    # Contexto RL/Control
    episode_id: Optional[str] = None
    step_id: Optional[int] = None

    # Datos del evento
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Explicación humana
    explanation: Optional[str] = None

    # Metadata adicional
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Nivel de detalle
    log_level: LogLevel = LogLevel.INFO

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el evento a diccionario para serialización"""
        def safe_serialize(obj: Any) -> Any:
            """Serializa un objeto de manera segura, convirtiendo objetos complejos a representaciones serializables"""
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [safe_serialize(item) for item in obj]
            elif isinstance(obj, dict):
                return {str(k): safe_serialize(v) for k, v in obj.items()}
            elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
                try:
                    return obj.to_dict()
                except:
                    return str(obj)
            elif hasattr(obj, '__dict__'):
                # Para objetos con atributos, serializar el __dict__
                try:
                    result = {}
                    for k, v in obj.__dict__.items():
                        if not k.startswith('_'):  # Evitar atributos privados
                            result[str(k)] = safe_serialize(v)
                    return result
                except:
                    return str(obj)
            else:
                # Para otros objetos, usar representación string
                return str(obj)

        data = {
            'event_id': self.event_id,
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'timestamp': self.timestamp.isoformat(),
            'source_module': self.source_module,
            'event_type': self.event_type.value,
            'episode_id': self.episode_id,
            'step_id': self.step_id,
            'inputs': safe_serialize(self.inputs),
            'outputs': safe_serialize(self.outputs),
            'context': safe_serialize(self.context),
            'metrics': safe_serialize(self.metrics),
            'explanation': self.explanation,
            'metadata': safe_serialize(self.metadata),
            'log_level': self.log_level.value
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CognitiveEventSchema':
        """Crea un evento desde diccionario"""
        # Convertir strings a enums
        data['event_type'] = EventType(data['event_type'])
        data['log_level'] = LogLevel(data['log_level'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

    def to_json(self) -> str:
        """Serializa a JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'CognitiveEventSchema':
        """Deserializa desde JSON"""
        return cls.from_dict(json.loads(json_str))

# Alias para compatibilidad
CognitiveEvent = CognitiveEventSchema