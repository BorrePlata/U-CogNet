"""
Utilidades comunes para U-CogNet.
"""

import time
import logging
from typing import Callable, Any
from functools import wraps

def setup_logging(level: int = logging.INFO) -> None:
    """Configura el sistema de logging."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def time_function(func: Callable) -> Callable:
    """Decorador para medir tiempo de ejecución."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} tomó {end - start:.4f} segundos")
        return result
    return wrapper

def calculate_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    """Calcula métricas de clasificación."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 if ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mcc': mcc
    }