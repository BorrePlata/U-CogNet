"""
Tipos de datos para procesamiento de audio.
Extensión de los tipos comunes para multimodalidad.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class AudioData:
    """
    Datos de audio crudos para procesamiento.
    """
    waveform: np.ndarray  # Forma: (samples, channels)
    sample_rate: int
    duration: float
    source: str
    timestamp: float
    metadata: Optional[dict] = None

@dataclass
class AudioFeatures:
    """
    Características extraídas del audio.
    """
    mfcc: np.ndarray  # Mel-frequency cepstral coefficients
    chroma: np.ndarray  # Chroma features
    spectral_centroid: np.ndarray
    spectral_bandwidth: np.ndarray
    zero_crossing_rate: np.ndarray
    rms: np.ndarray  # Root mean square energy
    tempo: float
    beat_positions: np.ndarray
    sample_rate: int
    duration: float

@dataclass
class AudioEvent:
    """
    Evento de audio detectado.
    """
    event_type: str  # e.g., "speech", "music", "noise", "silence"
    confidence: float
    start_time: float
    end_time: float
    features: AudioFeatures
    transcription: Optional[str] = None  # Si es habla
    language: Optional[str] = None

@dataclass
class MultimodalEvent:
    """
    Evento que combina visión y audio.
    """
    visual_events: list  # Lista de Detection/Event visuales
    audio_events: list   # Lista de AudioEvent
    timestamp: float
    correlation_score: float  # Qué tan bien se correlacionan visión y audio
    joint_context: str  # Descripción semántica conjunta