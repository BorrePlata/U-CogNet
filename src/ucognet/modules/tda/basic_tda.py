from typing import Optional
from ucognet.core.interfaces import TDAManager
from ucognet.core.types import SystemState, TopologyConfig

class BasicTDAManager(TDAManager):
    """TDA Manager básico que ajusta la topología basada en métricas y carga del sistema."""

    def __init__(self):
        # Estado de la topología actual
        self.current_config = TopologyConfig(
            active_modules=["input_handler", "vision_detector", "cognitive_core",
                          "semantic_feedback", "evaluator", "visual_interface"],
            connections={
                "input_handler": ["vision_detector"],
                "vision_detector": ["cognitive_core"],
                "cognitive_core": ["semantic_feedback", "evaluator"],
                "evaluator": ["tda_manager"],
                "semantic_feedback": ["visual_interface"],
                "tda_manager": ["vision_detector", "cognitive_core", "visual_interface"]
            },
            resource_allocation={
                "input_handler": 0.1,
                "vision_detector": 0.4,  # Más recursos para visión
                "cognitive_core": 0.2,
                "semantic_feedback": 0.1,
                "evaluator": 0.1,
                "visual_interface": 0.1
            }
        )

        # Historial de métricas para detectar tendencias
        self.metrics_history = []
        self.max_history = 10

        # Thresholds para activar cambios
        self.low_performance_threshold = 0.6  # F1-score bajo
        self.high_load_threshold = 0.8  # Carga alta del sistema

    def update(self, state: SystemState) -> TopologyConfig:
        """Actualiza la topología basada en el estado del sistema."""

        # Agregar métricas al historial
        if state.metrics:
            self.metrics_history.append(state.metrics)
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)

        # Analizar tendencias y decidir cambios
        changes_needed = self._analyze_system_state(state)

        if changes_needed:
            self._apply_topology_changes(changes_needed, state)

        return self.current_config

    def _analyze_system_state(self, state: SystemState) -> dict:
        """Analiza el estado del sistema y determina qué cambios aplicar."""
        changes = {}

        if not state.metrics or len(self.metrics_history) < 3:
            return changes  # No hay suficiente data

        recent_metrics = self.metrics_history[-3:]  # Últimas 3 mediciones
        avg_f1 = sum(m.f1 for m in recent_metrics) / len(recent_metrics)
        avg_precision = sum(m.precision for m in recent_metrics) / len(recent_metrics)
        avg_recall = sum(m.recall for m in recent_metrics) / len(recent_metrics)

        # Regla 1: Si F1-score está bajo persistentemente, aumentar recursos para visión
        if avg_f1 < self.low_performance_threshold:
            changes['increase_vision_resources'] = True
            changes['reason'] = f"F1-score bajo ({avg_f1:.2f})"

        # Regla 2: Si precisión es baja pero recall alto, problema de falsos positivos
        elif avg_precision < avg_recall - 0.1:
            changes['increase_evaluation_frequency'] = True
            changes['reason'] = f"Precisión baja vs recall ({avg_precision:.2f} vs {avg_recall:.2f})"

        # Regla 3: Si el sistema está sobrecargado, reducir complejidad
        if hasattr(state, 'system_load') and state.system_load > self.high_load_threshold:
            changes['reduce_complexity'] = True
            changes['reason'] = f"Carga alta del sistema ({state.system_load:.2f})"

        # Regla 4: Activar/desactivar módulos según contexto
        if state.metrics.map > 0.8:  # Buen rendimiento general
            changes['enable_advanced_features'] = True
        elif state.metrics.map < 0.5:  # Mal rendimiento
            changes['simplify_pipeline'] = True

        return changes

    def _apply_topology_changes(self, changes: dict, state: SystemState):
        """Aplica los cambios determinados a la topología."""

        if changes.get('increase_vision_resources'):
            # Aumentar recursos para visión, reducir otros
            self.current_config.resource_allocation['vision_detector'] = min(0.6,
                self.current_config.resource_allocation['vision_detector'] + 0.1)
            self.current_config.resource_allocation['semantic_feedback'] = max(0.05,
                self.current_config.resource_allocation['semantic_feedback'] - 0.05)

            print(f"🔧 TDA: Aumentando recursos de visión - {changes.get('reason', '')}")

        if changes.get('increase_evaluation_frequency'):
            # Aumentar frecuencia de evaluación
            self.current_config.resource_allocation['evaluator'] = min(0.2,
                self.current_config.resource_allocation['evaluator'] + 0.05)

            print(f"🔧 TDA: Aumentando evaluación - {changes.get('reason', '')}")

        if changes.get('reduce_complexity'):
            # Desactivar módulos no críticos
            if 'trainer_loop' in self.current_config.active_modules:
                self.current_config.active_modules.remove('trainer_loop')
                print("🔧 TDA: Desactivando trainer_loop por alta carga")

        if changes.get('enable_advanced_features'):
            # Activar módulos avanzados si no están activos
            if 'trainer_loop' not in self.current_config.active_modules:
                self.current_config.active_modules.append('trainer_loop')
                self.current_config.resource_allocation['trainer_loop'] = 0.1
                print("🔧 TDA: Activando características avanzadas (buen rendimiento)")

        if changes.get('simplify_pipeline'):
            # Simplificar pipeline removiendo conexiones complejas
            if 'tda_manager' in self.current_config.connections.get('evaluator', []):
                self.current_config.connections['evaluator'].remove('tda_manager')
                print("🔧 TDA: Simplificando pipeline (rendimiento bajo)")

        # Normalizar asignación de recursos
        total_resources = sum(self.current_config.resource_allocation.values())
        if total_resources > 1.0:
            for module in self.current_config.resource_allocation:
                self.current_config.resource_allocation[module] /= total_resources