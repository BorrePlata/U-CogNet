# U-CogNet Audio-Visual Module Protocols
# Universal Auditory-Perceptual Interfaces

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, Optional, Dict, Any, List
import numpy as np
from .audio_types import AudioFrame, AudioFeatures, AudioPerception, VisualExpression, RenderedVisual, SynthesisResult, EvaluationMetrics

# Audio Processing Protocols
@runtime_checkable
class AudioInputProtocol(Protocol):
    """Protocol for audio input sources."""

    @property
    def sample_rate(self) -> int:
        """Get the audio sample rate."""
        pass

    @property
    def channels(self) -> int:
        """Get the number of audio channels."""
        pass

    async def capture_audio(self, duration: float = 1.0) -> AudioFrame:
        """Capture audio for the specified duration."""
        pass

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the audio input."""
        pass

    async def cleanup(self) -> None:
        """Clean up audio input resources."""
        pass

@runtime_checkable
class AudioFeatureExtractionProtocol(Protocol):
    """Protocol for audio feature extraction."""

    async def extract_features(self, audio_frame: AudioFrame) -> AudioFeatures:
        """Extract features from audio data."""
        pass

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the feature extractor."""
        pass

    async def cleanup(self) -> None:
        """Clean up feature extraction resources."""
        pass

@runtime_checkable
class AudioPerceptionProtocol(Protocol):
    """Protocol for audio perception and interpretation."""

    async def perceive_audio(self, features: AudioFeatures) -> AudioPerception:
        """Perceive and interpret audio features."""
        pass

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the perception engine."""
        pass

    async def cleanup(self) -> None:
        """Clean up perception resources."""
        pass

@runtime_checkable
class VisualExpressionProtocol(Protocol):
    """Protocol for visual expression generation."""

    async def express_visually(self, perception: AudioPerception) -> VisualExpression:
        """Generate visual expression from audio perception."""
        pass

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the visual expressor."""
        pass

    async def cleanup(self) -> None:
        """Clean up visual expression resources."""
        pass

@runtime_checkable
class VisualRenderingProtocol(Protocol):
    """Protocol for visual rendering."""

    async def render_visual(self, expression: VisualExpression, format_type: str = "image") -> RenderedVisual:
        """Render visual expression to output format."""
        pass

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the visual renderer."""
        pass

    async def cleanup(self) -> None:
        """Clean up rendering resources."""
        pass

@runtime_checkable
class AudioVisualSynthesisProtocol(Protocol):
    """Protocol for audio-visual synthesis coordination."""

    async def synthesize_audio_visual(self, audio: np.ndarray, context: Optional[Dict[str, Any]] = None) -> SynthesisResult:
        """Synthesize audio-visual representation."""
        pass

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the synthesizer."""
        pass

    async def cleanup(self) -> None:
        """Clean up synthesis resources."""
        pass

@runtime_checkable
class AudioVisualEvaluationProtocol(Protocol):
    """Protocol for audio-visual evaluation."""

    async def evaluate_synthesis(self, result: SynthesisResult, user_feedback: Optional[Dict[str, Any]] = None) -> EvaluationMetrics:
        """Evaluate synthesis quality."""
        pass

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the evaluator."""
        pass

    async def cleanup(self) -> None:
        """Clean up evaluation resources."""
        pass