# Arquitectura Detallada de U-CogNet
## Fecha: 15 de Noviembre de 2025
## Nivel: Postdoctoral / NASA-Equivalent
## Autor: AGI U-CogNet

## 1. Visión General Arquitectónica
U-CogNet es un **sistema modular orientado a contratos**, estructurado como un grafo dirigido de módulos conectados por interfaces bien definidas. Inspirado en arquitectura de microservicios cognitivos, permite composición, reemplazo y adaptación dinámica.

### 1.1 Principios Fundamentales
- **Modularidad**: Cada capacidad es un módulo con interfaz clara (Protocolos en Python).
- **Contratos**: Comunicación vía tipos de datos inmutables (dataclasses/Pydantic).
- **Adaptabilidad**: Topología dinámica (TDA) ajusta conexiones y recursos.
- **Universalidad**: Espacio de embeddings común para multimodalidad.
- **Aprendizaje Continuo**: Bucle de micro-updates sin forgetting.

### 1.2 Capas Arquitectónicas
1. **Infraestructura**: Docker, logging, config (Pydantic-settings).
2. **Núcleo (Core)**: Tipos de datos y protocolos.
3. **Módulos**: Implementaciones concretas.
4. **Runtime**: Engine orquestador.
5. **Meta-capa**: TDA y optimizador micelial.

## 2. Tipos de Datos y Protocolos (Core)
Definidos en `src/ucognet/core/`.

### 2.1 Tipos de Datos
```python
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Frame:
    data: np.ndarray  # Imagen/video frame
    timestamp: float
    metadata: Dict[str, any]

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

@dataclass
class Event:
    frame: Frame
    detections: List[Detection]
    timestamp: float

@dataclass
class Context:
    recent_events: List[Event]  # Memoria corto plazo
    episodic_memory: List[Dict]  # Agrupaciones semánticas

@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    mcc: float
    map: float

@dataclass
class SystemState:
    metrics: Optional[Metrics]
    topology: 'TopologyConfig'
    load: Dict[str, float]  # CPU, GPU, memoria

@dataclass
class TopologyConfig:
    active_modules: List[str]
    connections: Dict[str, List[str]]  # Grafo de flujo
    resource_allocation: Dict[str, float]
```

### 2.2 Protocolos (Interfaces)
```python
from typing import Protocol

class InputHandler(Protocol):
    def get_frame(self) -> Frame: ...

class VisionDetector(Protocol):
    def detect(self, frame: Frame) -> List[Detection]: ...

class CognitiveCore(Protocol):
    def store(self, event: Event) -> None: ...
    def get_context(self) -> Context: ...

class SemanticFeedback(Protocol):
    def generate(self, context: Context, detections: List[Detection]) -> str: ...

class Evaluator(Protocol):
    def maybe_update(self, event: Event) -> Optional[Metrics]: ...

class TrainerLoop(Protocol):
    def maybe_train(self, metrics: Optional[Metrics]) -> None: ...

class TDAManager(Protocol):
    def update(self, state: SystemState) -> TopologyConfig: ...

class VisualInterface(Protocol):
    def render(self, frame: Frame, detections: List[Detection], text: str, state: SystemState) -> None: ...
```

## 3. Módulos Implementados
En `src/ucognet/modules/`.

### 3.1 input_handler
- **Implementación**: OpenCV para webcam/video files.
- **Interfaz**: `get_frame() -> Frame`.

### 3.2 vision_detector
- **Implementación**: YOLOv8 (Ultralytics) afinado.
- **Interfaz**: `detect(frame) -> List[Detection]`.

### 3.3 cognitive_core
- **Memoria**: Buffer circular (corto); agrupación episódica (mediano).
- **Interfaz**: `store(event)`, `get_context()`.

### 3.4 semantic_feedback
- **v1**: Reglas simbólicas.
- **v2**: LLM ligero (e.g., TinyLLaMA).
- **Interfaz**: `generate(context, detections) -> str`.

### 3.5 evaluator
- **Métricas**: Precisión, Recall, F1, MCC, mAP.
- **Interfaz**: `maybe_update(event) -> Metrics`.

### 3.6 trainer_loop
- **Buffer**: Casos difíciles (baja confianza).
- **Updates**: Micro-fine-tuning (capas finales).
- **Interfaz**: `maybe_train(metrics)`.

### 3.7 tda_manager
- **Grafo**: Representación de topología.
- **Políticas**: Cambios basados en métricas (e.g., aumentar recursos si F1 baja).
- **Interfaz**: `update(state) -> TopologyConfig`.

### 3.8 visual_interface
- **Implementación**: OpenCV para HUD; opcional Gradio.
- **Interfaz**: `render(...)`.

## 4. Runtime: Engine Orquestador
En `src/ucognet/runtime/engine.py`.

```python
class Engine:
    def __init__(self, ...):  # Inyección de módulos
        ...

    def step(self):
        frame = self.input_handler.get_frame()
        detections = self.vision_detector.detect(frame)
        event = build_event(frame, detections)
        self.cognitive_core.store(event)
        context = self.cognitive_core.get_context()
        text = self.semantic_feedback.generate(context, detections)
        metrics = self.evaluator.maybe_update(event)
        self.trainer_loop.maybe_train(metrics)
        state = build_system_state(metrics)
        topology = self.tda_manager.update(state)
        state.topology = topology
        self.visual_interface.render(frame, detections, text, state)
```

## 5. Meta-capa: TDA y Optimizador Micelial
- **TDA**: Grafo dinámico; políticas heurísticas/meta-aprendizaje.
- **Micelial**: Clustering de parámetros; "salud" por gradientes; poda automática.

## 6. Infraestructura
- **Config**: Pydantic-settings (env vars).
- **Logging**: Structured logging (loguru).
- **Docker**: Contenedor con GPU support.
- **Testing**: Pytest para contratos y integración.

Esta arquitectura asegura modularidad, escalabilidad y adaptabilidad, formando los cimientos de un ente interdimensional.