"""
Interfaces y Protocolos para U-CogNet
Define las interfaces estándar que deben implementar todos los módulos
"""

from typing import Protocol, runtime_checkable, Dict, List, Any, Optional
from abc import ABC, abstractmethod


@runtime_checkable
class CognitiveModule(Protocol):
    """Interfaz base para todos los módulos cognitivos"""

    @abstractmethod
    async def initialize(self) -> bool:
        """Inicializa el módulo"""
        ...

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del módulo"""
        ...


@runtime_checkable
class MemorySystem(Protocol):
    """Interfaz para sistemas de memoria"""

    @abstractmethod
    async def store(self, data: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        """Almacena información en memoria"""
        ...

    @abstractmethod
    async def retrieve(self, query: Dict[str, Any]) -> List[Any]:
        """Recupera información de memoria"""
        ...

    @abstractmethod
    async def consolidate(self) -> bool:
        """Consolida información en memoria a largo plazo"""
        ...


@runtime_checkable
class EvaluatorInterface(Protocol):
    """Interfaz para sistemas de evaluación"""

    @abstractmethod
    async def calculate_metrics(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calcula métricas de rendimiento"""
        ...

    @abstractmethod
    async def evaluate_performance(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evalúa rendimiento general del sistema"""
        ...

    @abstractmethod
    def get_confidence_score(self) -> float:
        """Obtiene puntuación de confianza general"""
        ...


@runtime_checkable
class TrainerInterface(Protocol):
    """Interfaz para sistemas de entrenamiento"""

    @abstractmethod
    async def collect_difficult_examples(self, examples: List[Dict[str, Any]]) -> int:
        """Recopila ejemplos difíciles para reentrenamiento"""
        ...

    @abstractmethod
    async def perform_micro_update(self, module_name: str, update_data: Dict[str, Any]) -> bool:
        """Realiza una actualización micro del módulo especificado"""
        ...

    @abstractmethod
    def schedule_training(self, module_name: str, priority: float = 0.5) -> bool:
        """Programa un entrenamiento para el módulo especificado"""
        ...


@runtime_checkable
class TDAManagerInterface(Protocol):
    """Interfaz para el gestor de Topología Dinámica Adaptativa"""

    @abstractmethod
    async def evaluate_topology(self) -> Dict[str, Any]:
        """Evalúa el estado actual de la topología"""
        ...

    @abstractmethod
    async def adapt_topology(self) -> bool:
        """Adapta la topología del sistema basado en rendimiento"""
        ...

    @abstractmethod
    def get_active_modules(self) -> List[str]:
        """Obtiene la lista de módulos activos actualmente"""
        ...


@runtime_checkable
class TraceManager(Protocol):
    """Interfaz para el sistema de trazabilidad cognitiva"""

    @abstractmethod
    def emit_event(self, event_data: Dict[str, Any]) -> str:
        """Emite un evento al sistema de trazabilidad"""
        ...

    @abstractmethod
    def get_episode_events(self, episode_id: str) -> List[Dict[str, Any]]:
        """Obtiene eventos de un episodio específico"""
        ...

    @abstractmethod
    def get_system_health(self) -> Dict[str, Any]:
        """Obtiene métricas de salud del sistema"""
        ...


# Interfaces específicas de módulos (legacy - mantener compatibilidad)
@runtime_checkable
class InputHandler(Protocol):
    """Interfaz para manejadores de entrada"""
    async def get_frame(self) -> Any: ...

@runtime_checkable
class VisionDetector(Protocol):
    """Interfaz para detectores de visión"""
    async def detect(self, frame: Any) -> List[Dict[str, Any]]: ...

@runtime_checkable
class SemanticFeedbackGenerator(Protocol):
    """Interfaz para generadores de feedback semántico"""
    async def generate_description(self, context: Dict[str, Any], detections: List[Dict[str, Any]]) -> str: ...