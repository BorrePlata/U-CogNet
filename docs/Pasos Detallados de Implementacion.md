# Pasos Detallados de Implementación: U-CogNet Paso a Paso
## Fecha: 15 de Noviembre de 2025
## Nivel: Postdoctoral / NASA-Equivalent (ASGI Cósmica)
## Autor: AGI U-CogNet

Este documento detalla **mega detalladamente** los pasos para implementar U-CogNet, basado en el Roadmap de Implementación. Cada paso es **sencillo pero valioso**: actionable, incremental y validable. Aplicamos ingeniería inversa, empezando por la fundación y avanzando hacia el ente interdimensional.

**Principio**: Ejecuta un paso, valida (corre tests, verifica output), itera si falla. Usa Docker para aislamiento. Asume entorno Linux con Python 3.11+ y GPU RTX 4060.

---

## Fase 0: Fundación (Semanas 1-2) – Estructura y Contratos Básicos
**Objetivo**: Crear el esqueleto del sistema sin lógica compleja. Valor: Base sólida para todo lo demás.

### Paso 0.1: Configurar el Mono-Repo
- **Descripción**: Crear estructura de directorios y archivos base.
- **Acciones Detalladas**:
  1. En `/mnt/c/Users/desar/Documents/Science/UCogNet`, crea carpetas: `src/ucognet/`, `tests/`, `docker/`, `docs/`.
  2. Dentro de `src/ucognet/`, crea subcarpetas: `core/`, `modules/`, `runtime/`, `infra/`.
  3. Crea `pyproject.toml` en raíz con dependencias mínimas:
     ```
     [tool.poetry]
     name = "ucognet"
     version = "0.1.0"
     description = "Sistema Cognitivo Artificial Universal"
     authors = ["AGI U-CogNet <agi@ucognet.com>"]

     [tool.poetry.dependencies]
     python = "^3.11"
     pydantic = "^2.0"
     numpy = "^1.24"
     opencv-python = "^4.8"

     [tool.poetry.group.dev.dependencies]
     pytest = "^7.0"
     black = "^23.0"

     [build-system]
     requires = ["poetry-core"]
     build-backend = "poetry.core.masonry.api"
     ```
  4. Instala Poetry si no lo tienes: `curl -sSL https://install.python-poetry.org | python3 -`.
  5. Corre `poetry install` en la raíz.
- **Validación**: `poetry show` lista dependencias. Valor: Repo organizado, listo para código.

### Paso 0.2: Definir Tipos de Datos (Core)
- **Descripción**: Implementar dataclasses para contratos.
- **Acciones Detalladas**:
  1. Crea `src/ucognet/core/types.py`:
     ```python
     from dataclasses import dataclass
     from typing import List, Dict, Optional, Any
     import numpy as np

     @dataclass
     class Frame:
         data: np.ndarray
         timestamp: float
         metadata: Dict[str, Any]

     @dataclass
     class Detection:
         class_id: int
         class_name: str
         confidence: float
         bbox: List[float]

     @dataclass
     class Event:
         frame: Frame
         detections: List[Detection]
         timestamp: float

     @dataclass
     class Context:
         recent_events: List[Event]
         episodic_memory: List[Dict]

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
         load: Dict[str, float]

     @dataclass
     class TopologyConfig:
         active_modules: List[str]
         connections: Dict[str, List[str]]
         resource_allocation: Dict[str, float]
     ```
  2. Crea `src/ucognet/core/__init__.py` vacío.
- **Validación**: Importa en Python: `from ucognet.core.types import Frame`. Valor: Contratos claros, evita bugs futuros.

### Paso 0.3: Definir Protocolos (Interfaces)
- **Descripción**: Interfaces para modularidad.
- **Acciones Detalladas**:
  1. Crea `src/ucognet/core/interfaces.py`:
     ```python
     from typing import Protocol, List, Optional
     from .types import Frame, Detection, Event, Context, Metrics, SystemState, TopologyConfig

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
- **Validación**: Crea un test simple: Implementa una clase dummy que herede y verifica con `isinstance`. Valor: Modularidad garantizada.

### Paso 0.4: Implementar Módulos Dummy
- **Descripción**: Mocks para probar contratos.
- **Acciones Detalladas**:
  1. Crea `src/ucognet/modules/input/mock_input.py`:
     ```python
     import numpy as np
     from ucognet.core.interfaces import InputHandler
     from ucognet.core.types import Frame

     class MockInputHandler(InputHandler):
         def get_frame(self) -> Frame:
             return Frame(data=np.zeros((480, 640, 3), dtype=np.uint8), timestamp=0.0, metadata={})
     ```
  2. Repite para otros módulos (e.g., `mock_vision_detector.py` que retorna detecciones fijas).
- **Validación**: Instancia y llama métodos; verifica tipos. Valor: Sistema testable desde el inicio.

### Paso 0.5: Crear Engine Básico y Entrypoint
- **Descripción**: Orquestador mínimo.
- **Acciones Detalladas**:
  1. Crea `src/ucognet/runtime/engine.py`:
     ```python
     from ucognet.core.interfaces import *
     from ucognet.core.types import build_event, build_system_state

     class Engine:
         def __init__(self, input_handler: InputHandler, ...):  # Inyecta todos
             ...

         def step(self):
             frame = self.input_handler.get_frame()
             detections = self.vision_detector.detect(frame)
             event = build_event(frame, detections)
             # ... simplificado para mocks
             print("Step executed")
     ```
  2. Crea `src/ucognet/__main__.py`:
     ```python
     from ucognet.runtime.engine import Engine
     from ucognet.modules.input.mock_input import MockInputHandler
     # ... importa mocks

     def main():
         engine = Engine(MockInputHandler(), ...)
         for _ in range(10):
             engine.step()

     if __name__ == "__main__":
         main()
     ```
- **Validación**: Corre `python -m ucognet`; ve output sin errores. Valor: Bucle básico funcionando.

### Paso 0.6: Config Básica y Logging
- **Descripción**: Infra transversal.
- **Acciones Detalladas**:
  1. Instala `pydantic-settings` y `loguru`.
  2. Crea `src/ucognet/infra/config.py` con settings.
  3. Agrega logging a Engine.
- **Validación**: Logs aparecen en consola. Valor: Debugging fácil.

---

## Fase 1: Demo Táctico Funcional (Semanas 3-6) – Visión + Feedback
**Objetivo**: Conectar visión real y feedback básico.

### Paso 1.1: Integrar OpenCV en InputHandler
- **Descripción**: Leer de webcam/video.
- **Acciones Detalladas**:
  1. Modifica `modules/input/opencv_camera.py` para usar `cv2.VideoCapture`.
  2. Maneja errores (e.g., si no hay cam).
- **Validación**: Muestra frame en ventana. Valor: Input real.

### Paso 1.2: Conectar YOLOv8
- **Descripción**: Detector real.
- **Acciones Detalladas**:
  1. Instala `ultralytics`.
  2. En `modules/vision/yolov8_detector.py`, carga modelo preentrenado.
  3. Ajusta para clases (tanque, etc.).
- **Validación**: Detecta objetos en imagen de prueba. Valor: Visión funcional.

### Paso 1.3: Implementar CognitiveCore con Buffers
- **Descripción**: Memoria básica.
- **Acciones Detalladas**:
  1. Usa `collections.deque` para buffer circular.
- **Validación**: Almacena y recupera eventos. Valor: Contexto inicial.

### Paso 1.4: Reglas Simbólicas en SemanticFeedback
- **Descripción**: Feedback simple.
- **Acciones Detalladas**:
  1. Lógica if-else para "Convoy detectado".
- **Validación**: Genera texto coherente. Valor: Explicaciones básicas.

### Paso 1.5: HUD en VisualInterface
- **Descripción**: Render con OpenCV.
- **Acciones Detalladas**:
  1. Dibuja bboxes y texto.
- **Validación**: Ventana muestra detecciones. Valor: Demo visual.

---

## Fase 2: Cognición Viva (Semanas 7-12) – Memoria y Auto-evaluación
**Objetivo**: Hacerlo "pensar".

### Paso 2.1: Memoria Episódica
- **Descripción**: Agrupar eventos.
- **Acciones Detalladas**:
  1. Lógica para detectar "episodios" (e.g., objetos persistentes).
- **Validación**: Agrupa secuencias. Valor: Memoria mediano plazo.

### Paso 2.2: Métricas en Evaluator
- **Descripción**: Calcular F1, etc.
- **Acciones Detalladas**:
  1. Implementa fórmulas con sklearn.
- **Validación**: Métricas razonables. Valor: Auto-evaluación.

### Paso 2.3: TrainerLoop con Buffer
- **Descripción**: Aprendizaje básico.
- **Acciones Detalladas**:
  1. Almacena casos difíciles; simula updates.
- **Validación**: Mejora en detecciones. Valor: Aprendizaje continuo inicial.

---

## Fase 3: Adaptabilidad Dinámica (Semanas 13-18) – TDA y Micelio
**Objetivo**: Auto-adaptación.

### Paso 3.1: TDA Básico
- **Descripción**: Cambiar thresholds.
- **Acciones Detalladas**:
  1. Políticas simples en TDAManager.
- **Validación**: Ajustes basados en métricas. Valor: Dinamismo.

### Paso 3.2: Optimizador Micelial
- **Descripción**: Clustering de params.
- **Acciones Detalladas**:
  1. Usa scikit-learn para clusters.
- **Validación**: Poda automática. Valor: Eficiencia.

---

## Fase 4: Expansión Multimodal (Semanas 19-24) – Universalidad
**Objetivo**: Más modalidades.

### Paso 4.1: Nuevo Handler (e.g., Audio)
- **Descripción**: Integrar audio.
- **Acciones Detalladas**:
  1. Encoder simple a embeddings.
- **Validación**: Transfer a visión. Valor: Multimodalidad.

---

## Fase 5: Escalabilidad (Semanas 25-30) – Producción
**Objetivo**: Despliegue.

### Paso 5.1: Dockerizar
- **Descripción**: Contenedor.
- **Acciones Detalladas**:
  1. Crea Dockerfile y docker-compose.yml.
- **Validación**: Corre en contenedor. Valor: Escalabilidad.

Cada paso es incremental; valida contra el Examen. ¡Avanza paso a paso hacia el ente interdimensional!