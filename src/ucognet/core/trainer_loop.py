"""
Trainer Loop - Sistema de Aprendizaje Continuo para U-CogNet
Gestiona el reentrenamiento incremental y adaptación de modelos
"""

import asyncio
import time
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
import queue

from .interfaces import TrainerInterface
from .tracing import get_event_bus, EventType


class TrainingMode(Enum):
    """Modos de entrenamiento disponibles"""
    ONLINE = "online"      # Aprendizaje continuo en tiempo real
    BATCH = "batch"        # Entrenamiento por lotes
    INCREMENTAL = "incremental"  # Aprendizaje incremental
    FEW_SHOT = "few_shot"  # Aprendizaje con pocos ejemplos


@dataclass
class TrainingExample:
    """Ejemplo de entrenamiento"""
    data: Any
    label: Any
    confidence: float
    timestamp: float
    source: str
    difficulty: float  # 0-1, qué tan difícil fue clasificar


@dataclass
class TrainingBatch:
    """Lote de entrenamiento"""
    examples: List[TrainingExample]
    priority: float
    module_target: str
    training_mode: TrainingMode
    created_at: float


class TrainerLoop(TrainerInterface):
    """
    Bucle de entrenamiento continuo para U-CogNet
    Gestiona el aprendizaje incremental sin forgetting catastrófico
    """

    def __init__(self):
        self.event_bus = get_event_bus()

        # Colas de entrenamiento
        self.training_queue = queue.PriorityQueue()
        self.difficult_examples: List[TrainingExample] = []

        # Estado del entrenamiento
        self.is_training = False
        self.training_thread: Optional[threading.Thread] = None
        self.should_stop = False

        # Configuración de entrenamiento
        self.batch_size = 32
        self.learning_rate = 0.001
        self.regularization_strength = 0.01
        self.difficulty_threshold = 0.7  # umbral para considerar ejemplo "difícil"

        # Control de frecuencia
        self.min_training_interval = 60  # segundos entre entrenamientos
        self.max_examples_buffer = 1000
        self.last_training_time = 0

        # Estadísticas de entrenamiento
        self.training_stats = {
            "total_trainings": 0,
            "examples_processed": 0,
            "models_updated": 0,
            "performance_improvements": [],
            "training_times": []
        }

        # Módulos registrados para entrenamiento
        self.registered_modules: Dict[str, Dict[str, Any]] = {}

        print("🎓 Trainer Loop inicializado")

    async def initialize(self) -> bool:
        """Inicializa el sistema de entrenamiento"""
        try:
            # Registrar módulos de entrenamiento
            await self._register_training_modules()

            # Iniciar hilo de entrenamiento
            self.training_thread = threading.Thread(target=self._training_worker, daemon=True)
            self.training_thread.start()

            # Iniciar monitoreo de rendimiento
            asyncio.create_task(self._performance_monitor())

            # Emitir evento de inicialización
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "TrainerLoop",
                outputs={"initialized": True, "modules_registered": len(self.registered_modules)},
                explanation="Inicialización del Trainer Loop"
            )

            print("✅ Trainer Loop inicializado correctamente")
            return True

        except Exception as e:
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "TrainerLoop",
                outputs={"initialized": False, "error": str(e)},
                log_level=2
            )
            print(f"❌ Error inicializando Trainer Loop: {e}")
            return False

    async def _register_training_modules(self):
        """Registra módulos que pueden ser entrenados"""
        # Módulos principales de U-CogNet
        modules = [
            {
                "name": "vision_detector",
                "type": "detection",
                "framework": "torch",
                "update_frequency": "high",
                "regularization": "ewc"  # Elastic Weight Consolidation
            },
            {
                "name": "cognitive_core",
                "type": "reasoning",
                "framework": "torch",
                "update_frequency": "medium",
                "regularization": "l2"
            },
            {
                "name": "semantic_feedback",
                "type": "language",
                "framework": "transformers",
                "update_frequency": "low",
                "regularization": "dropout"
            },
            {
                "name": "incremental_tank_learner",
                "type": "classification",
                "framework": "sklearn",
                "update_frequency": "high",
                "regularization": "none"
            }
        ]

        for module_info in modules:
            self.registered_modules[module_info["name"]] = module_info

        print(f"📚 Registrados {len(modules)} módulos para entrenamiento")

    def _training_worker(self):
        """Hilo trabajador que procesa la cola de entrenamiento"""
        while not self.should_stop:
            try:
                # Obtener siguiente lote de entrenamiento
                if not self.training_queue.empty():
                    priority, batch = self.training_queue.get(timeout=1)

                    # Procesar lote
                    self._process_training_batch(batch)

                    self.training_queue.task_done()
                else:
                    # Esperar un poco antes de verificar nuevamente
                    time.sleep(2)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Error en hilo de entrenamiento: {e}")
                time.sleep(5)

    async def _performance_monitor(self):
        """Monitorea el rendimiento y decide cuándo entrenar"""
        while True:
            try:
                current_time = time.time()

                # Verificar si hay suficientes ejemplos difíciles
                if (len(self.difficult_examples) >= self.batch_size and
                    current_time - self.last_training_time >= self.min_training_interval):

                    # Crear lote de entrenamiento
                    batch = await self._create_training_batch()
                    if batch:
                        # Agregar a cola con prioridad
                        priority = 1.0 - batch.priority  # mayor prioridad = menor número
                        self.training_queue.put((priority, batch))

                # Limpiar ejemplos antiguos si es necesario
                await self._cleanup_old_examples()

                await asyncio.sleep(30)  # verificar cada 30 segundos

            except Exception as e:
                print(f"⚠️ Error en monitor de rendimiento: {e}")
                await asyncio.sleep(60)

    async def collect_difficult_examples(self, examples: List[Dict[str, Any]]) -> int:
        """
        Recopila ejemplos difíciles para reentrenamiento

        Args:
            examples: Lista de ejemplos con información de dificultad

        Returns:
            Número de ejemplos recopilados
        """
        collected = 0

        for example_data in examples:
            # Determinar dificultad
            difficulty = self._calculate_example_difficulty(example_data)

            if difficulty >= self.difficulty_threshold:
                example = TrainingExample(
                    data=example_data.get("data"),
                    label=example_data.get("label"),
                    confidence=example_data.get("confidence", 0.5),
                    timestamp=time.time(),
                    source=example_data.get("source", "unknown"),
                    difficulty=difficulty
                )

                self.difficult_examples.append(example)
                collected += 1

                # Limitar buffer
                if len(self.difficult_examples) > self.max_examples_buffer:
                    # Remover ejemplos más antiguos
                    self.difficult_examples = self.difficult_examples[-self.max_examples_buffer:]

        if collected > 0:
            # Emitir evento
            self.event_bus.emit(
                EventType.LEARNING_STEP,
                "TrainerLoop",
                outputs={"examples_collected": collected, "buffer_size": len(self.difficult_examples)},
                explanation="Ejemplos difíciles recopilados para entrenamiento"
            )

        return collected

    def _calculate_example_difficulty(self, example_data: Dict[str, Any]) -> float:
        """Calcula la dificultad de un ejemplo"""
        # Factores de dificultad
        confidence = example_data.get("confidence", 0.5)
        uncertainty = 1.0 - confidence  # mayor incertidumbre = mayor dificultad

        # Si hay múltiples predicciones, considerar discrepancia
        if "predictions" in example_data and len(example_data["predictions"]) > 1:
            predictions = np.array(example_data["predictions"])
            prediction_std = np.std(predictions)
            discrepancy = min(prediction_std / np.mean(predictions), 1.0) if np.mean(predictions) > 0 else 0
        else:
            discrepancy = 0.0

        # Si es una corrección manual, alta dificultad
        is_correction = example_data.get("is_correction", False)
        correction_factor = 1.0 if is_correction else 0.0

        # Combinar factores
        difficulty = (
            0.5 * uncertainty +
            0.3 * discrepancy +
            0.2 * correction_factor
        )

        return min(difficulty, 1.0)

    async def _create_training_batch(self) -> Optional[TrainingBatch]:
        """Crea un lote de entrenamiento a partir de ejemplos difíciles"""
        if len(self.difficult_examples) < self.batch_size:
            return None

        # Seleccionar ejemplos más recientes y difíciles
        sorted_examples = sorted(
            self.difficult_examples,
            key=lambda x: (x.difficulty, x.timestamp),
            reverse=True
        )

        # Tomar los mejores ejemplos
        batch_examples = sorted_examples[:self.batch_size]

        # Calcular prioridad del lote
        avg_difficulty = np.mean([ex.difficulty for ex in batch_examples])
        recency_factor = 1.0 - (time.time() - batch_examples[0].timestamp) / 3600  # decaimiento por hora
        priority = (0.7 * avg_difficulty + 0.3 * recency_factor)

        # Determinar módulo objetivo (simplificado)
        target_module = "vision_detector"  # por defecto

        # Determinar modo de entrenamiento
        if len(batch_examples) < 10:
            mode = TrainingMode.FEW_SHOT
        elif avg_difficulty > 0.8:
            mode = TrainingMode.INCREMENTAL
        else:
            mode = TrainingMode.ONLINE

        batch = TrainingBatch(
            examples=batch_examples,
            priority=priority,
            module_target=target_module,
            training_mode=mode,
            created_at=time.time()
        )

        return batch

    def _process_training_batch(self, batch: TrainingBatch):
        """Procesa un lote de entrenamiento"""
        try:
            self.is_training = True
            start_time = time.time()

            # Emitir evento de inicio de entrenamiento
            self.event_bus.emit(
                EventType.LEARNING_STEP,
                "TrainerLoop",
                inputs={
                    "batch_size": len(batch.examples),
                    "module_target": batch.module_target,
                    "training_mode": batch.training_mode.value
                },
                explanation="Inicio de entrenamiento de lote"
            )

            # Preparar datos de entrenamiento
            train_data, train_labels = self._prepare_training_data(batch.examples)

            # Seleccionar estrategia de entrenamiento
            if batch.training_mode == TrainingMode.INCREMENTAL:
                success = self._incremental_training(train_data, train_labels, batch.module_target)
            elif batch.training_mode == TrainingMode.FEW_SHOT:
                success = self._few_shot_training(train_data, train_labels, batch.module_target)
            else:
                success = self._online_training(train_data, train_labels, batch.module_target)

            # Medir tiempo de entrenamiento
            training_time = time.time() - start_time

            # Actualizar estadísticas
            self.training_stats["total_trainings"] += 1
            self.training_stats["examples_processed"] += len(batch.examples)
            self.training_stats["training_times"].append(training_time)

            if success:
                self.training_stats["models_updated"] += 1

            # Limpiar ejemplos procesados
            processed_ids = {id(ex) for ex in batch.examples}
            self.difficult_examples = [ex for ex in self.difficult_examples if id(ex) not in processed_ids]

            self.last_training_time = time.time()

            # Emitir evento de finalización
            self.event_bus.emit(
                EventType.LEARNING_STEP,
                "TrainerLoop",
                outputs={
                    "training_success": success,
                    "training_time": training_time,
                    "examples_processed": len(batch.examples)
                },
                explanation="Entrenamiento de lote completado"
            )

            print(f"✅ Entrenamiento completado: {batch.module_target} ({training_time:.2f}s)")

        except Exception as e:
            # Emitir evento de error
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "TrainerLoop",
                outputs={"training_error": str(e)},
                log_level=2
            )
            print(f"❌ Error en entrenamiento: {e}")
        finally:
            self.is_training = False

    def _prepare_training_data(self, examples: List[TrainingExample]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara datos para entrenamiento"""
        # Simplificación: asumir datos numéricos
        data_list = []
        label_list = []

        for example in examples:
            if isinstance(example.data, (list, np.ndarray)):
                data_list.append(example.data)
            else:
                # Placeholder para datos no numéricos
                data_list.append([0.0] * 10)  # vector dummy

            label_list.append(example.label if example.label is not None else 0)

        return np.array(data_list), np.array(label_list)

    def _incremental_training(self, data: np.ndarray, labels: np.ndarray, module: str) -> bool:
        """Entrenamiento incremental con regularización"""
        try:
            # Simular entrenamiento incremental
            # En implementación real, esto actualizaría el modelo del módulo

            # Simular mejora de rendimiento
            performance_improvement = np.random.normal(0.05, 0.02)  # ~5% mejora
            self.training_stats["performance_improvements"].append(performance_improvement)

            # Simular tiempo de entrenamiento
            time.sleep(0.1)  # simulación

            return True

        except Exception as e:
            print(f"Error en entrenamiento incremental: {e}")
            return False

    def _few_shot_training(self, data: np.ndarray, labels: np.ndarray, module: str) -> bool:
        """Entrenamiento few-shot para nuevos conceptos"""
        try:
            # Simular few-shot learning
            if len(data) < 5:
                return False

            # Simular adaptación rápida
            time.sleep(0.05)  # simulación más rápida

            performance_improvement = np.random.normal(0.08, 0.03)  # ~8% mejora
            self.training_stats["performance_improvements"].append(performance_improvement)

            return True

        except Exception as e:
            print(f"Error en few-shot training: {e}")
            return False

    def _online_training(self, data: np.ndarray, labels: np.ndarray, module: str) -> bool:
        """Entrenamiento online continuo"""
        try:
            # Simular actualización online
            time.sleep(0.02)  # simulación muy rápida

            performance_improvement = np.random.normal(0.03, 0.01)  # ~3% mejora
            self.training_stats["performance_improvements"].append(performance_improvement)

            return True

        except Exception as e:
            print(f"Error en online training: {e}")
            return False

    async def _cleanup_old_examples(self):
        """Limpia ejemplos antiguos del buffer"""
        current_time = time.time()
        max_age = 3600 * 24  # 24 horas

        initial_count = len(self.difficult_examples)
        self.difficult_examples = [
            ex for ex in self.difficult_examples
            if current_time - ex.timestamp < max_age
        ]

        removed = initial_count - len(self.difficult_examples)
        if removed > 0:
            print(f"🧹 Limpiados {removed} ejemplos antiguos del buffer")

    async def perform_micro_update(self, module_name: str, update_data: Dict[str, Any]) -> bool:
        """
        Realiza una actualización micro del módulo especificado

        Args:
            module_name: Nombre del módulo a actualizar
            update_data: Datos para la actualización

        Returns:
            True si la actualización fue exitosa
        """
        if module_name not in self.registered_modules:
            return False

        try:
            # Emitir evento de micro-update
            self.event_bus.emit(
                EventType.LEARNING_STEP,
                "TrainerLoop",
                inputs={"module": module_name, "update_type": "micro"},
                explanation="Inicio de micro-update"
            )

            # Simular micro-update
            time.sleep(0.01)  # simulación

            # Actualizar estadísticas
            self.training_stats["total_trainings"] += 1

            # Emitir evento de éxito
            self.event_bus.emit(
                EventType.LEARNING_STEP,
                "TrainerLoop",
                outputs={"micro_update_success": True, "module": module_name},
                explanation="Micro-update completado"
            )

            return True

        except Exception as e:
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "TrainerLoop",
                outputs={"micro_update_error": str(e), "module": module_name},
                log_level=2
            )
            return False

    def schedule_training(self, module_name: str, priority: float = 0.5) -> bool:
        """
        Programa un entrenamiento para el módulo especificado

        Args:
            module_name: Nombre del módulo
            priority: Prioridad del entrenamiento (0-1)

        Returns:
            True si el entrenamiento fue programado
        """
        if module_name not in self.registered_modules:
            return False

        # Crear lote dummy para forzar entrenamiento
        dummy_examples = [
            TrainingExample(
                data=[0.0] * 10,
                label=0,
                confidence=0.5,
                timestamp=time.time(),
                source="scheduled_training",
                difficulty=0.8
            ) for _ in range(self.batch_size)
        ]

        batch = TrainingBatch(
            examples=dummy_examples,
            priority=priority,
            module_target=module_name,
            training_mode=TrainingMode.BATCH,
            created_at=time.time()
        )

        # Agregar a cola
        self.training_queue.put((1.0 - priority, batch))

        # Emitir evento
        self.event_bus.emit(
            EventType.MODULE_INTERACTION,
            "TrainerLoop",
            outputs={"training_scheduled": True, "module": module_name, "priority": priority},
            explanation="Entrenamiento programado"
        )

        return True

    def get_training_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del sistema de entrenamiento"""
        return {
            "is_training": self.is_training,
            "queue_size": self.training_queue.qsize(),
            "difficult_examples_count": len(self.difficult_examples),
            "registered_modules": list(self.registered_modules.keys()),
            "training_stats": self.training_stats.copy(),
            "last_training_time": self.last_training_time
        }

    def stop(self):
        """Detiene el sistema de entrenamiento"""
        self.should_stop = True
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5)

        print("🛑 Trainer Loop detenido")