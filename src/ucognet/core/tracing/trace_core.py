"""
Cognitive Trace Core (CTC)
Núcleo central que recibe y procesa todos los eventos cognitivos del sistema
"""

import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
import time
from datetime import datetime, timedelta
import json
from pathlib import Path

from .cognitive_event import CognitiveEvent, EventType, LogLevel
from ..utils import ensure_directory

class CognitiveTraceCore:
    """
    Núcleo central de trazabilidad cognitiva.
    Recibe eventos de todos los módulos y los procesa de forma centralizada.
    """

    def __init__(self,
                 buffer_size: int = 10000,
                 storage_path: str = "cognitive_traces",
                 async_processing: bool = True):
        """
        Inicializa el núcleo de trazabilidad.

        Args:
            buffer_size: Tamaño máximo del buffer en memoria
            storage_path: Directorio para almacenamiento persistente
            async_processing: Si procesar eventos de forma asíncrona
        """
        self.buffer_size = buffer_size
        self.storage_path = Path(storage_path)
        self.async_processing = async_processing

        # Buffers y almacenamiento
        self.event_buffer: List[CognitiveEvent] = []
        self.episode_buffers: Dict[str, List[CognitiveEvent]] = defaultdict(list)

        # Callbacks para procesamiento
        self.event_processors: List[Callable[[CognitiveEvent], None]] = []

        # Estadísticas
        self.stats = {
            'events_processed': 0,
            'episodes_tracked': 0,
            'storage_operations': 0,
            'processing_time_ms': 0
        }

        # Configuración de niveles de log
        self.log_levels = {
            'default': LogLevel.INFO,
            'modules': {}  # Configuración por módulo
        }

        # Inicializar almacenamiento
        ensure_directory(self.storage_path)

        # Thread/Async setup
        if async_processing:
            self._processing_queue = asyncio.Queue()
            self._processing_task = None
        else:
            self._lock = threading.Lock()

        print("🧠 CognitiveTraceCore inicializado")

    def emit_event(self, event: CognitiveEvent) -> None:
        """
        Emite un evento al sistema de trazabilidad.
        Método principal llamado por todos los módulos.
        """
        start_time = time.time()

        # Filtrar por nivel de log
        if not self._should_log_event(event):
            return

        if self.async_processing:
            # Procesamiento asíncrono
            asyncio.create_task(self._process_event_async(event))
        else:
            # Procesamiento síncrono
            with self._lock:
                self._process_event_sync(event)

        # Actualizar estadísticas
        processing_time = (time.time() - start_time) * 1000
        self.stats['processing_time_ms'] += processing_time

    async def _process_event_async(self, event: CognitiveEvent) -> None:
        """Procesa evento de forma asíncrona"""
        await self._processing_queue.put(event)

        # Iniciar tarea de procesamiento si no existe
        if self._processing_task is None or self._processing_task.done():
            self._processing_task = asyncio.create_task(self._process_queue_worker())

    async def _process_queue_worker(self) -> None:
        """Worker para procesar cola de eventos"""
        while True:
            try:
                event = await self._processing_queue.get()
                await asyncio.get_event_loop().run_in_executor(None, self._process_event_sync, event)
                self._processing_queue.task_done()
            except Exception as e:
                print(f"❌ Error procesando evento: {e}")
                break

    def _process_event_sync(self, event: CognitiveEvent) -> None:
        """Procesa un evento de forma síncrona"""
        try:
            # 1. Agregar al buffer principal
            self.event_buffer.append(event)
            if len(self.event_buffer) > self.buffer_size:
                self.event_buffer.pop(0)

            # 2. Agregar al buffer de episodio si aplica
            if event.episode_id:
                self.episode_buffers[event.episode_id].append(event)

            # 3. Notificar procesadores
            for processor in self.event_processors:
                try:
                    processor(event)
                except Exception as e:
                    print(f"⚠️ Error en procesador: {e}")

            # 4. Persistir si es evento importante
            if self._should_persist_event(event):
                self._persist_event(event)

            # 5. Actualizar estadísticas
            self.stats['events_processed'] += 1
            if event.episode_id:
                self.stats['episodes_tracked'] = len(self.episode_buffers)

        except Exception as e:
            print(f"❌ Error procesando evento {event.event_id}: {e}")

    def _should_log_event(self, event: CognitiveEvent) -> bool:
        """Determina si un evento debe ser loggeado basado en nivel"""
        module_level = self.log_levels['modules'].get(event.source_module, self.log_levels['default'])

        level_hierarchy = {
            LogLevel.TRACE: 0,
            LogLevel.DEBUG: 1,
            LogLevel.INFO: 2,
            LogLevel.SUMMARY: 3
        }

        # Convertir log_level a enum si es necesario
        event_level = event.log_level
        if isinstance(event_level, int):
            # Mapear enteros a enums (legacy support)
            int_to_level = {0: LogLevel.TRACE, 1: LogLevel.DEBUG, 2: LogLevel.INFO, 3: LogLevel.SUMMARY}
            event_level = int_to_level.get(event_level, LogLevel.INFO)
        elif isinstance(event_level, str):
            # Mapear strings a enums
            str_to_level = {"trace": LogLevel.TRACE, "debug": LogLevel.DEBUG, "info": LogLevel.INFO, "summary": LogLevel.SUMMARY}
            event_level = str_to_level.get(event_level.lower(), LogLevel.INFO)

        return level_hierarchy[event_level] >= level_hierarchy[module_level]

    def _should_persist_event(self, event: CognitiveEvent) -> bool:
        """Determina si un evento debe persistirse"""
        # Persistir eventos importantes
        important_types = {
            EventType.DECISION,
            EventType.GATING_CHANGE,
            EventType.TOPOLOGY_CHANGE,
            EventType.SECURITY_CHECK,
            EventType.LEARNING_STEP
        }

        return event.event_type in important_types or event.log_level in [LogLevel.INFO, LogLevel.SUMMARY]

    def _persist_event(self, event: CognitiveEvent) -> None:
        """Persiste un evento en almacenamiento"""
        try:
            # Crear directorio por fecha
            date_dir = self.storage_path / event.timestamp.strftime("%Y-%m-%d")
            ensure_directory(date_dir)

            # Archivo por hora
            hour_file = date_dir / f"events_{event.timestamp.strftime('%H')}.jsonl"

            # Agregar evento como línea JSON
            with open(hour_file, 'a', encoding='utf-8') as f:
                f.write(event.to_json() + '\n')

            self.stats['storage_operations'] += 1

        except Exception as e:
            print(f"❌ Error persistiendo evento: {e}")

    def register_processor(self, processor: Callable[[CognitiveEvent], None]) -> None:
        """Registra un procesador de eventos"""
        self.event_processors.append(processor)

    def set_log_level(self, level: LogLevel, module: Optional[str] = None) -> None:
        """Configura nivel de log para módulo o global"""
        if module:
            self.log_levels['modules'][module] = level
        else:
            self.log_levels['default'] = level

    def get_episode_events(self, episode_id: str) -> List[CognitiveEvent]:
        """Obtiene todos los eventos de un episodio"""
        return self.episode_buffers.get(episode_id, [])

    def get_recent_events(self, limit: int = 100) -> List[CognitiveEvent]:
        """Obtiene eventos recientes"""
        return self.event_buffer[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de trazabilidad"""
        return {
            **self.stats,
            'buffer_size': len(self.event_buffer),
            'active_episodes': len(self.episode_buffers),
            'log_levels': {k: v.value for k, v in self.log_levels['modules'].items()},
            'default_log_level': self.log_levels['default'].value
        }

    def cleanup_old_episodes(self, max_age_hours: int = 24) -> None:
        """Limpia episodios antiguos de memoria"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        episodes_to_remove = []
        for episode_id, events in self.episode_buffers.items():
            if events and events[0].timestamp < cutoff_time:
                episodes_to_remove.append(episode_id)

        for episode_id in episodes_to_remove:
            del self.episode_buffers[episode_id]

    def shutdown(self) -> None:
        """Cierra el sistema de trazabilidad"""
        if self.async_processing and self._processing_task:
            self._processing_task.cancel()

        # Persistir buffer restante
        for event in self.event_buffer[-100:]:  # Últimos 100 eventos
            self._persist_event(event)

        print("🧠 CognitiveTraceCore cerrado")