"""
Cognitive Event Bus
Capa de instrumentación que permite a todos los módulos emitir eventos sin acoplamiento
"""

import functools
import inspect
from typing import Dict, Any, Optional, Callable, TypeVar
from contextlib import contextmanager
import time

from .cognitive_event import CognitiveEvent, EventType, LogLevel
from .trace_core import CognitiveTraceCore

# Singleton global para el bus de eventos
_global_event_bus = None

def get_event_bus() -> 'CognitiveEventBus':
    """Obtiene la instancia global del event bus"""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = CognitiveEventBus()
    return _global_event_bus

class CognitiveEventBus:
    """
    Bus de eventos cognitivos.
    Proporciona API unificada para que todos los módulos emitan eventos.
    """

    def __init__(self, trace_core: Optional[CognitiveTraceCore] = None):
        self.trace_core = trace_core or CognitiveTraceCore()
        self.decorators_active = True
        print("📡 CognitiveEventBus inicializado")

    def emit(self,
             event_type: EventType,
             source_module: str,
             inputs: Optional[Dict[str, Any]] = None,
             outputs: Optional[Dict[str, Any]] = None,
             context: Optional[Dict[str, Any]] = None,
             metrics: Optional[Dict[str, Any]] = None,
             explanation: Optional[str] = None,
             episode_id: Optional[str] = None,
             step_id: Optional[int] = None,
             trace_id: Optional[str] = None,
             span_id: Optional[str] = None,
             log_level: LogLevel = LogLevel.INFO,
             metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Emite un evento al sistema de trazabilidad.

        Returns:
            event_id: ID único del evento emitido
        """
        event = CognitiveEvent(
            source_module=source_module,
            event_type=event_type,
            inputs=inputs or {},
            outputs=outputs or {},
            context=context or {},
            metrics=metrics or {},
            explanation=explanation,
            episode_id=episode_id,
            step_id=step_id,
            trace_id=trace_id,
            span_id=span_id,
            log_level=log_level,
            metadata=metadata or {}
        )

        self.trace_core.emit_event(event)
        return event.event_id

    # Métodos convenientes para tipos comunes de eventos

    def emit_decision(self, source_module: str, decision: Any, confidence: float = None,
                     context: Dict[str, Any] = None, episode_id: str = None, step_id: int = None) -> str:
        """Emite evento de decisión"""
        return self.emit(
            EventType.DECISION,
            source_module,
            outputs={'decision': decision, 'confidence': confidence},
            context=context,
            episode_id=episode_id,
            step_id=step_id
        )

    def emit_reward(self, source_module: str, reward: float, reward_type: str = "scalar",
                   context: Dict[str, Any] = None, episode_id: str = None, step_id: int = None) -> str:
        """Emite evento de reward"""
        return self.emit(
            EventType.REWARD,
            source_module,
            outputs={'reward': reward, 'reward_type': reward_type},
            context=context,
            episode_id=episode_id,
            step_id=step_id
        )

    def emit_update(self, source_module: str, parameter: str, old_value: Any, new_value: Any,
                   update_reason: str = None, context: Dict[str, Any] = None) -> str:
        """Emite evento de actualización de parámetro"""
        return self.emit(
            EventType.UPDATE,
            source_module,
            inputs={'parameter': parameter, 'old_value': old_value},
            outputs={'new_value': new_value},
            context=context,
            explanation=update_reason
        )

    def emit_gating_change(self, source_module: str, gate_name: str, new_state: bool,
                          reason: str = None, context: Dict[str, Any] = None) -> str:
        """Emite evento de cambio de gating"""
        return self.emit(
            EventType.GATING_CHANGE,
            source_module,
            outputs={'gate_name': gate_name, 'new_state': new_state},
            context=context,
            explanation=reason
        )

    def emit_security_check(self, source_module: str, check_type: str, result: bool,
                           details: Dict[str, Any] = None, context: Dict[str, Any] = None) -> str:
        """Emite evento de verificación de seguridad"""
        return self.emit(
            EventType.SECURITY_CHECK,
            source_module,
            outputs={'check_type': check_type, 'result': result, 'details': details},
            context=context
        )

    def emit_module_interaction(self, source_module: str, target_module: str,
                               interaction_type: str, data: Dict[str, Any] = None,
                               context: Dict[str, Any] = None) -> str:
        """Emite evento de interacción entre módulos"""
        return self.emit(
            EventType.MODULE_INTERACTION,
            source_module,
            outputs={'target_module': target_module, 'interaction_type': interaction_type, 'data': data},
            context=context
        )

    # Decoradores para instrumentación automática

    def trace_method(self, event_type: EventType = EventType.MODULE_INTERACTION,
                    log_inputs: bool = True, log_outputs: bool = True,
                    log_level: LogLevel = LogLevel.DEBUG):
        """
        Decorador para instrumentar métodos automáticamente.
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.decorators_active:
                    return func(*args, **kwargs)

                # Obtener nombre del módulo
                module_name = getattr(func, '__module__', 'unknown')
                if module_name == '__main__':
                    module_name = 'main'

                start_time = time.time()

                # Preparar inputs
                inputs = {}
                if log_inputs:
                    try:
                        sig = inspect.signature(func)
                        bound_args = sig.bind(*args, **kwargs)
                        bound_args.apply_defaults()
                        inputs = dict(bound_args.arguments)
                        # Remover 'self' si existe
                        inputs.pop('self', None)
                    except:
                        inputs = {'args_count': len(args), 'kwargs_keys': list(kwargs.keys())}

                try:
                    # Ejecutar función
                    result = func(*args, **kwargs)

                    # Calcular métricas
                    execution_time = time.time() - start_time
                    metrics = {'execution_time_ms': execution_time * 1000}

                    # Preparar outputs
                    outputs = {}
                    if log_outputs:
                        if result is not None:
                            outputs['result'] = str(result)[:500]  # Limitar tamaño
                        outputs['success'] = True

                    # Emitir evento
                    self.emit(
                        event_type,
                        module_name,
                        inputs=inputs,
                        outputs=outputs,
                        metrics=metrics,
                        explanation=f"Method {func.__name__} executed",
                        log_level=log_level
                    )

                    return result

                except Exception as e:
                    # Error en ejecución
                    execution_time = time.time() - start_time
                    metrics = {'execution_time_ms': execution_time * 1000, 'error': str(e)}

                    self.emit(
                        EventType.MODULE_INTERACTION,  # Usar interaction para errores también
                        module_name,
                        inputs=inputs,
                        outputs={'success': False, 'error': str(e)},
                        metrics=metrics,
                        explanation=f"Method {func.__name__} failed: {e}",
                        log_level=LogLevel.INFO  # Los errores siempre se loggean como INFO
                    )
                    raise

            return wrapper
        return decorator

    @contextmanager
    def trace_context(self, operation_name: str, source_module: str,
                     context: Dict[str, Any] = None, log_level: LogLevel = LogLevel.DEBUG):
        """
        Context manager para trazar bloques de código.
        """
        start_time = time.time()
        trace_id = None

        if self.decorators_active:
            trace_id = self.emit(
                EventType.MODULE_INTERACTION,
                source_module,
                outputs={'operation': operation_name, 'phase': 'start'},
                context=context,
                log_level=log_level
            )

        try:
            yield
        finally:
            if self.decorators_active:
                execution_time = time.time() - start_time
                self.emit(
                    EventType.MODULE_INTERACTION,
                    source_module,
                    outputs={'operation': operation_name, 'phase': 'end'},
                    context=context,
                    metrics={'execution_time_ms': execution_time * 1000},
                    trace_id=trace_id,
                    log_level=log_level
                )

    def set_decorators_active(self, active: bool) -> None:
        """Activa/desactiva los decoradores de trazabilidad"""
        self.decorators_active = active
        status = "activados" if active else "desactivados"
        print(f"📡 Decoradores de trazabilidad {status}")

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del bus de eventos"""
        return {
            'decorators_active': self.decorators_active,
            **self.trace_core.get_stats()
        }

# Función global para acceso fácil
def emit_event(event_type: EventType, source_module: str, **kwargs) -> str:
    """Función global para emitir eventos fácilmente"""
    return get_event_bus().emit(event_type, source_module, **kwargs)