"""
Utilidades Comunes para U-CogNet
Funciones de utilidad compartidas por todos los módulos
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
from datetime import datetime
from .types import Event, Frame, Detection, SystemState, Metrics, TopologyConfig


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Carga configuración desde archivo

    Args:
        config_path: Ruta al archivo de configuración

    Returns:
        Diccionario con configuración
    """
    if config_path is None:
        # Buscar en ubicaciones estándar
        search_paths = [
            Path.cwd() / "config.json",
            Path.cwd() / "ucognet_config.json",
            Path.home() / ".ucognet" / "config.json"
        ]

        for path in search_paths:
            if path.exists():
                config_path = str(path)
                break

    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            elif config_path.endswith(('.yml', '.yaml')):
                return yaml.safe_load(f)

    # Configuración por defecto
    return {
        "system": {
            "name": "U-CogNet",
            "version": "1.0.0",
            "debug": False
        },
        "modules": {
            "tracing": {"enabled": True},
            "tda": {"enabled": True},
            "mycelial_optimizer": {"enabled": True}
        },
        "performance": {
            "max_memory_gb": 8,
            "target_fps": 30,
            "batch_size": 32
        }
    }


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Configura el sistema de logging

    Args:
        level: Nivel de logging
        log_file: Archivo para guardar logs

    Returns:
        Logger configurado
    """
    # Crear logger
    logger = logging.getLogger("ucognet")
    logger.setLevel(getattr(logging, level.upper()))

    # Formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para archivo si se especifica
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def calculate_metrics(predictions: List[Any], ground_truth: List[Any]) -> Dict[str, float]:
    """
    Calcula métricas de evaluación básicas

    Args:
        predictions: Predicciones del modelo
        ground_truth: Valores reales

    Returns:
        Diccionario con métricas
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predicciones y ground truth deben tener la misma longitud")

    if not predictions:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Convertir a arrays numpy
    y_pred = np.array(predictions)
    y_true = np.array(ground_truth)

    # Accuracy
    accuracy = np.mean(y_pred == y_true)

    # Para clasificación binaria/multiclass
    unique_labels = np.unique(np.concatenate([y_pred, y_true]))

    if len(unique_labels) == 2:
        # Clasificación binaria
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        # Multiclass - aproximaciones simples
        precision = accuracy
        recall = accuracy
        f1 = accuracy

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def save_checkpoint(data: Dict[str, Any], filepath: str, compress: bool = True) -> bool:
    """
    Guarda un checkpoint del sistema

    Args:
        data: Datos a guardar
        filepath: Ruta del archivo
        compress: Si comprimir el archivo

    Returns:
        True si se guardó correctamente
    """
    try:
        # Crear directorio si no existe
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Agregar metadata
        checkpoint_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "compressed": compress
            },
            "data": data
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        return True

    except Exception as e:
        print(f"Error guardando checkpoint: {e}")
        return False


def load_checkpoint(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Carga un checkpoint del sistema

    Args:
        filepath: Ruta del archivo

    Returns:
        Datos cargados o None si falla
    """
    try:
        if not Path(filepath).exists():
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)

        return checkpoint_data.get("data")

    except Exception as e:
        print(f"Error cargando checkpoint: {e}")
        return None


def ensure_directory(path: str) -> bool:
    """
    Asegura que un directorio existe

    Args:
        path: Ruta del directorio

    Returns:
        True si el directorio existe o se creó
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def get_system_info() -> Dict[str, Any]:
    """Obtiene información del sistema"""
    return {
        "platform": os.sys.platform,
        "python_version": os.sys.version,
        "cpu_count": os.cpu_count(),
        "working_directory": str(Path.cwd()),
        "timestamp": datetime.now().isoformat()
    }


def format_duration(seconds: float) -> str:
    """Formatea duración en segundos a string legible"""
    if seconds < 60:
        return ".2f"
    elif seconds < 3600:
        return ".1f"
    else:
        return ".1f"


def create_experiment_id(prefix: str = "exp") -> str:
    """Crea un ID único para experimento"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Valida configuración del sistema

    Args:
        config: Configuración a validar

    Returns:
        Lista de errores encontrados
    """
    errors = []

    # Validar estructura básica
    required_sections = ["system", "modules", "performance"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Sección requerida faltante: {section}")

    # Validar módulos
    if "modules" in config:
        valid_modules = ["tracing", "tda", "mycelial_optimizer", "cognitive_core"]
        for module in config["modules"]:
            if module not in valid_modules:
                errors.append(f"Módulo desconocido: {module}")

    # Validar rendimiento
    if "performance" in config:
        perf = config["performance"]
        if "max_memory_gb" in perf and perf["max_memory_gb"] <= 0:
            errors.append("max_memory_gb debe ser positivo")
        if "target_fps" in perf and perf["target_fps"] <= 0:
            errors.append("target_fps debe ser positivo")

    return errors


def build_event(frame: Frame, detections: list[Detection]) -> Event:
    """Construye un evento a partir de frame y detecciones"""
    return Event(frame=frame, detections=detections, timestamp=frame.timestamp)


def build_system_state(metrics: Optional[Metrics]) -> SystemState:
    """Construye estado del sistema"""
    return SystemState(metrics=metrics, topology=TopologyConfig([], {}, {}), load={})