# U-CogNet Audio-Visual Module Types
# Universal Auditory-Perceptual System with Artistic Expression

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np

# Audio Processing Types
@dataclass
class AudioFrame:
    """Represents a frame of audio data with metadata."""
    timestamp: datetime
    data: np.ndarray  # Raw audio samples
    sample_rate: int
    channels: int
    duration: float  # Duration in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AudioFeatures:
    """Extracted audio features for cognitive processing."""
    timestamp: datetime

    # Spectral features
    mfcc: np.ndarray  # Mel-frequency cepstral coefficients
    chroma: np.ndarray  # Chroma features
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    zero_crossing_rate: float

    # Temporal features
    rms_energy: float
    onset_strength: float
    tempo: float
    beat_positions: List[float]

    # Advanced features
    harmonic_ratio: float
    percussive_ratio: float
    tonnetz: np.ndarray  # Tonal centroid features

    # Raw spectrogram data
    spectrogram: np.ndarray
    mel_spectrogram: np.ndarray

    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AudioPerception:
    """Cognitive interpretation of audio features."""
    sound_type: str  # 'music', 'speech', 'nature', 'mechanical', etc.
    emotional_valence: float  # -1 (negative) to 1 (positive)
    arousal_level: float  # 0 (calm) to 1 (excited)
    attention_weight: float  # 0 (ignore) to 1 (focus)
    environment_context: str  # 'indoor', 'outdoor', 'urban', 'natural', etc.
    confidence: float  # 0 to 1

@dataclass
class VisualExpression:
    """Artistic visual representation of audio perception."""
    style: str  # 'organic', 'geometric', 'abstract', 'minimalist', etc.
    intensity: float  # 0 to 1
    colors: List[str]  # Hex color codes
    composition: Dict[str, Any]  # Shape and pattern specifications
    confidence: float  # 0 to 1

@dataclass
class VisualSymbol:
    """Symbolic visual element."""
    symbol_type: str
    position: Tuple[float, float]  # x, y coordinates (0-1)
    size: float  # Relative size (0-1)
    color: str  # Hex color code
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RenderedVisual:
    """Final rendered visual output."""
    data: Any  # Image data in requested format
    format: str  # 'image', 'numpy', 'base64', etc.
    dimensions: Tuple[int, int]  # width, height
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SynthesisResult:
    """Complete synthesis result."""
    synthesis_id: str
    timestamp: datetime
    audio_input: np.ndarray
    features: Optional[AudioFeatures] = None
    perception: Optional[AudioPerception] = None
    expression: Optional[VisualExpression] = None
    rendered_visual: Optional[RenderedVisual] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationMetrics:
    """Evaluation metrics for synthesis quality."""
    evaluation_id: str
    synthesis_id: str
    timestamp: datetime
    overall_score: float  # 0 to 1
    quality_level: str  # 'excellent', 'good', 'acceptable', 'poor', 'unacceptable'
    detailed_scores: Dict[str, float]
    recommendations: List[str]
    user_satisfaction: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AdaptationParameters:
    """Parameters for system adaptation."""
    learning_rate: float = 0.1
    feature_extraction_params: Dict[str, Any] = field(default_factory=dict)
    perception_params: Dict[str, Any] = field(default_factory=dict)
    expression_params: Dict[str, Any] = field(default_factory=dict)
    rendering_params: Dict[str, Any] = field(default_factory=dict)
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)