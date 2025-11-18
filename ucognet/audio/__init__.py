"""
Módulo de procesamiento de audio para U-CogNet.
Incluye extracción, razonamiento cognitivo, interiorización e imaginación.
"""

from .audio_cognitive_processor import (
    AudioCognitiveProcessor,
    AudioExtractionStrategy,
    MoviePyAudioStrategy,
    AudioReasoning,
    AudioImagination,
    AudioCognitiveMetrics
)

__all__ = [
    'AudioCognitiveProcessor',
    'AudioExtractionStrategy',
    'MoviePyAudioStrategy',
    'AudioReasoning',
    'AudioImagination',
    'AudioCognitiveMetrics'
]