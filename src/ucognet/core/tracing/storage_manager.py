"""
Trace Storage Manager
Gestiona el almacenamiento persistente y retención de trazas cognitivas
"""

import os
import json
import gzip
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
import threading
from collections import defaultdict

from .cognitive_event import CognitiveEvent, LogLevel

class StorageConfig:
    """Configuración de almacenamiento"""

    def __init__(self,
                 base_path: str = "cognitive_traces",
                 compression_enabled: bool = True,
                 max_file_size_mb: int = 100,
                 retention_days: int = 30,
                 archive_old_data: bool = True):
        self.base_path = Path(base_path)
        self.compression_enabled = compression_enabled
        self.max_file_size_mb = max_file_size_mb
        self.retention_days = retention_days
        self.archive_old_data = archive_old_data

class TraceStorageManager:
    """
    Gestor de almacenamiento para trazas cognitivas.
    Maneja persistencia, compresión, retención y recuperación de datos.
    """

    def __init__(self, config: StorageConfig):
        self.config = config
        self._ensure_storage_structure()
        self._lock = threading.Lock()
        self.stats = {
            'files_created': 0,
            'data_compressed': 0,
            'data_archived': 0,
            'data_deleted': 0
        }

    def _ensure_storage_structure(self) -> None:
        """Asegura que la estructura de directorios existe"""
        # Directorios principales
        self.config.base_path.mkdir(parents=True, exist_ok=True)

        # Subdirectorios
        (self.config.base_path / "active").mkdir(exist_ok=True)
        (self.config.base_path / "archive").mkdir(exist_ok=True)
        (self.config.base_path / "compressed").mkdir(exist_ok=True)
        (self.config.base_path / "metadata").mkdir(exist_ok=True)

    def store_event(self, event: CognitiveEvent) -> None:
        """Almacena un evento en el sistema de archivos"""
        with self._lock:
            try:
                # Determinar archivo de destino
                file_path = self._get_event_file_path(event)

                # Preparar datos para almacenamiento
                event_data = event.to_dict()

                # Almacenar
                self._append_to_file(file_path, event_data)

                # Verificar tamaño del archivo
                if self._should_rotate_file(file_path):
                    self._rotate_file(file_path)

            except Exception as e:
                print(f"❌ Error almacenando evento {event.event_id}: {e}")

    def store_events_batch(self, events: List[CognitiveEvent]) -> None:
        """Almacena múltiples eventos eficientemente"""
        with self._lock:
            # Agrupar por archivo de destino
            file_groups = defaultdict(list)

            for event in events:
                file_path = self._get_event_file_path(event)
                file_groups[str(file_path)].append(event)

            # Almacenar por grupos
            for file_path_str, group_events in file_groups.items():
                file_path = Path(file_path_str)
                try:
                    event_data_list = [event.to_dict() for event in group_events]
                    self._append_batch_to_file(file_path, event_data_list)

                    # Verificar rotación
                    if self._should_rotate_file(file_path):
                        self._rotate_file(file_path)

                except Exception as e:
                    print(f"❌ Error almacenando batch en {file_path}: {e}")

    def retrieve_events(self, start_date: datetime, end_date: datetime,
                       source_modules: Optional[List[str]] = None,
                       event_types: Optional[List[str]] = None) -> List[CognitiveEvent]:
        """Recupera eventos en un rango temporal"""
        events = []

        # Encontrar archivos relevantes
        relevant_files = self._find_files_in_range(start_date, end_date)

        for file_path in relevant_files:
            try:
                file_events = self._read_events_from_file(file_path)

                # Aplicar filtros
                for event_data in file_events:
                    event = CognitiveEvent.from_dict(event_data)

                    # Filtro temporal
                    if not (start_date <= event.timestamp <= end_date):
                        continue

                    # Filtro de módulo
                    if source_modules and event.source_module not in source_modules:
                        continue

                    # Filtro de tipo
                    if event_types and event.event_type.value not in event_types:
                        continue

                    events.append(event)

            except Exception as e:
                print(f"⚠️ Error leyendo archivo {file_path}: {e}")

        return sorted(events, key=lambda e: e.timestamp)

    def get_episode_events(self, episode_id: str) -> List[CognitiveEvent]:
        """Recupera todos los eventos de un episodio específico"""
        events = []

        # Buscar en archivos recientes primero
        active_files = list((self.config.base_path / "active").glob("*.jsonl*"))

        for file_path in active_files:
            try:
                file_events = self._read_events_from_file(file_path)

                for event_data in file_events:
                    if event_data.get('episode_id') == episode_id:
                        events.append(CognitiveEvent.from_dict(event_data))

            except Exception as e:
                print(f"⚠️ Error leyendo archivo {file_path}: {e}")

        return sorted(events, key=lambda e: e.timestamp)

    def cleanup_old_data(self) -> Dict[str, int]:
        """Limpia datos antiguos según política de retención"""
        cleanup_stats = {'archived': 0, 'deleted': 0, 'compressed': 0}

        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)

        # Procesar archivos activos
        active_dir = self.config.base_path / "active"
        for file_path in active_dir.glob("*.jsonl*"):
            try:
                file_date = self._extract_date_from_filename(file_path)

                if file_date < cutoff_date:
                    if self.config.archive_old_data:
                        # Archivar
                        self._archive_file(file_path)
                        cleanup_stats['archived'] += 1
                    else:
                        # Eliminar
                        file_path.unlink()
                        cleanup_stats['deleted'] += 1

            except Exception as e:
                print(f"⚠️ Error procesando archivo para cleanup {file_path}: {e}")

        # Comprimir archivos archivados antiguos
        if self.config.compression_enabled:
            archive_dir = self.config.base_path / "archive"
            for file_path in archive_dir.glob("*.jsonl"):
                try:
                    # Comprimir si no está comprimido y es antiguo
                    if file_path.stat().st_size > 1024 * 1024:  # > 1MB
                        self._compress_file(file_path)
                        cleanup_stats['compressed'] += 1

                except Exception as e:
                    print(f"⚠️ Error comprimiendo archivo {file_path}: {e}")

        # Actualizar estadísticas globales
        self.stats['data_archived'] += cleanup_stats['archived']
        self.stats['data_deleted'] += cleanup_stats['deleted']
        self.stats['data_compressed'] += cleanup_stats['compressed']

        return cleanup_stats

    def get_storage_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de almacenamiento"""
        stats = dict(self.stats)

        # Calcular tamaños
        active_size = self._calculate_directory_size(self.config.base_path / "active")
        archive_size = self._calculate_directory_size(self.config.base_path / "archive")
        compressed_size = self._calculate_directory_size(self.config.base_path / "compressed")

        stats.update({
            'active_size_mb': active_size / (1024 * 1024),
            'archive_size_mb': archive_size / (1024 * 1024),
            'compressed_size_mb': compressed_size / (1024 * 1024),
            'total_size_mb': (active_size + archive_size + compressed_size) / (1024 * 1024)
        })

        # Contar archivos
        stats.update({
            'active_files': len(list((self.config.base_path / "active").glob("*"))),
            'archive_files': len(list((self.config.base_path / "archive").glob("*"))),
            'compressed_files': len(list((self.config.base_path / "compressed").glob("*")))
        })

        return stats

    def _get_event_file_path(self, event: CognitiveEvent) -> Path:
        """Determina la ruta del archivo para un evento"""
        # Formato: active/events_YYYY-MM-DD_HH.jsonl
        timestamp = event.timestamp
        filename = f"events_{timestamp.strftime('%Y-%m-%d_%H')}.jsonl"

        return self.config.base_path / "active" / filename

    def _append_to_file(self, file_path: Path, event_data: Dict[str, Any]) -> None:
        """Agrega un evento a un archivo"""
        # Crear directorio si no existe
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Agregar línea JSON
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(event_data, f, ensure_ascii=False)
            f.write('\n')

    def _append_batch_to_file(self, file_path: Path, events_data: List[Dict[str, Any]]) -> None:
        """Agrega múltiples eventos a un archivo eficientemente"""
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'a', encoding='utf-8') as f:
            for event_data in events_data:
                json.dump(event_data, f, ensure_ascii=False)
                f.write('\n')

    def _should_rotate_file(self, file_path: Path) -> bool:
        """Determina si un archivo debe rotarse por tamaño"""
        if not file_path.exists():
            return False

        size_mb = file_path.stat().st_size / (1024 * 1024)
        return size_mb >= self.config.max_file_size_mb

    def _rotate_file(self, file_path: Path) -> None:
        """Rota un archivo cuando excede el tamaño máximo"""
        # Crear nombre rotado
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        rotated_path = file_path.parent / rotated_name

        # Renombrar archivo
        file_path.rename(rotated_path)

        # Comprimir si está habilitado
        if self.config.compression_enabled:
            self._compress_file(rotated_path)

    def _compress_file(self, file_path: Path) -> None:
        """Comprime un archivo usando gzip"""
        compressed_path = self.config.base_path / "compressed" / f"{file_path.name}.gz"

        try:
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Eliminar archivo original
            file_path.unlink()

        except Exception as e:
            print(f"❌ Error comprimiendo {file_path}: {e}")

    def _archive_file(self, file_path: Path) -> None:
        """Mueve un archivo a la carpeta de archivo"""
        archive_path = self.config.base_path / "archive" / file_path.name
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.move(str(file_path), str(archive_path))
        except Exception as e:
            print(f"❌ Error archivando {file_path}: {e}")

    def _read_events_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Lee eventos desde un archivo"""
        events = []

        try:
            # Manejar archivos comprimidos
            if file_path.suffix == '.gz':
                opener = gzip.open
            else:
                opener = open

            with opener(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            event_data = json.loads(line)
                            events.append(event_data)
                        except json.JSONDecodeError:
                            print(f"⚠️ Línea corrupta en {file_path}: {line[:100]}...")

        except Exception as e:
            print(f"❌ Error leyendo archivo {file_path}: {e}")

        return events

    def _find_files_in_range(self, start_date: datetime, end_date: datetime) -> List[Path]:
        """Encuentra archivos que contienen datos en el rango temporal"""
        files = []

        # Buscar en directorio activo
        active_dir = self.config.base_path / "active"
        for file_path in active_dir.glob("*.jsonl*"):
            try:
                file_date = self._extract_date_from_filename(file_path)
                if start_date.date() <= file_date.date() <= end_date.date():
                    files.append(file_path)
            except:
                continue

        # Buscar en archivo
        archive_dir = self.config.base_path / "archive"
        for file_path in archive_dir.glob("*.jsonl*"):
            try:
                file_date = self._extract_date_from_filename(file_path)
                if start_date.date() <= file_date.date() <= end_date.date():
                    files.append(file_path)
            except:
                continue

        return files

    def _extract_date_from_filename(self, file_path: Path) -> datetime:
        """Extrae fecha del nombre del archivo"""
        # Formato esperado: events_YYYY-MM-DD_HH.jsonl
        filename = file_path.stem
        if '_events_' in filename:
            date_str = filename.split('_events_')[1][:10]  # YYYY-MM-DD
        else:
            # Buscar patrón YYYY-MM-DD
            import re
            match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
            if match:
                date_str = match.group(1)
            else:
                raise ValueError(f"No se puede extraer fecha de {filename}")

        return datetime.strptime(date_str, '%Y-%m-%d')

    def _calculate_directory_size(self, directory: Path) -> int:
        """Calcula el tamaño total de un directorio"""
        total_size = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except:
            pass
        return total_size