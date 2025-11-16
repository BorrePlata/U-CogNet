# U-CogNet Audio-Visual Module
# Universal Audio-Visual Perception and Expression System

"""
U-CogNet Audio-Visual Module

This module provides a universal audio-visual perception and expression system
that can perceive environmental sounds and express them visually.
"""

from .audio_types import (
    AudioFeatures,
    AudioPerception,
    VisualExpression,
    VisualSymbol,
    RenderedVisual,
    SynthesisResult,
    EvaluationMetrics,
    AdaptationParameters
)

# Temporarily disable problematic imports for testing
# from .audio_protocols import (
#     AudioInputProtocol,
#     AudioFeatureExtractionProtocol,
#     AudioPerceptionProtocol,
#     VisualExpressionProtocol,
#     VisualRenderingProtocol,
#     AudioVisualSynthesisProtocol,
#     AudioVisualEvaluationProtocol
# )

# from .feature_extractor import LibrosaFeatureExtractor, FallbackFeatureExtractor
# from .audio_perception import CognitiveAudioPerception
# from .visual_expression import ArtisticVisualExpression
# from .visual_rendering import ArtisticVisualRenderer
# from .audio_visual_synthesis import AudioVisualSynthesizer
# from .audio_evaluation import AudioVisualEvaluator

__all__ = [
    # Core Types
    'AudioFeatures',
    'AudioPerception',
    'VisualExpression',
    'VisualSymbol',
    'RenderedVisual',
    'SynthesisResult',
    'EvaluationMetrics',
    'AdaptationParameters',

    # Protocols (temporarily disabled)
    # 'AudioInputProtocol',
    # 'AudioFeatureExtractionProtocol',
    # 'AudioPerceptionProtocol',
    # 'VisualExpressionProtocol',
    # 'VisualRenderingProtocol',
    # 'AudioVisualSynthesisProtocol',
    # 'AudioVisualEvaluationProtocol',

    # Implementations (temporarily disabled)
    # 'LibrosaFeatureExtractor',
    # 'FallbackFeatureExtractor',
    # 'CognitiveAudioPerception',
    # 'ArtisticVisualExpression',
    # 'ArtisticVisualRenderer',
    # 'AudioVisualSynthesizer',
    # 'AudioVisualEvaluator'
]

__version__ = "1.0.0"
__author__ = "U-CogNet Development Team"
__description__ = "Universal Audio-Visual Perception and Expression System"