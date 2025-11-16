"""Simplified audio types tests for U-CogNet."""

import pytest
import numpy as np
from datetime import datetime
import sys
sys.path.insert(0, 'src')
from ucognet.modules.audio.audio_types import (
    AudioFeatures, AudioPerception, VisualExpression, 
    RenderedVisual, SynthesisResult, EvaluationMetrics, AdaptationParameters
)

class TestAudioTypes:
    """Test audio-visual data types."""
    
    def test_audio_features_creation(self):
        """Test AudioFeatures creation."""
        mfcc = np.random.rand(13, 100)
        chroma = np.random.rand(12, 100)
        
        features = AudioFeatures(
            timestamp=datetime.now(),
            mfcc=mfcc,
            chroma=chroma,
            spectral_centroid=2500.0,
            spectral_bandwidth=1000.0,
            spectral_rolloff=3000.0,
            zero_crossing_rate=0.15,
            rms_energy=0.3,
            onset_strength=0.6,
            tempo=120.0,
            beat_positions=[0.1, 0.5, 1.0],
            harmonic_ratio=0.7,
            percussive_ratio=0.3,
            tonnetz=np.random.rand(6),
            spectrogram=np.random.rand(1025, 100),
            mel_spectrogram=np.random.rand(128, 100)
        )
        
        assert features.spectral_centroid == 2500.0
        assert features.tempo == 120.0
        assert features.mfcc.shape == (13, 100)
        
    def test_audio_perception_creation(self):
        """Test AudioPerception creation."""
        perception = AudioPerception(
            sound_type='music',
            emotional_valence=0.5,
            arousal_level=0.6,
            attention_weight=0.7,
            environment_context='concert',
            confidence=0.8
        )
        
        assert perception.sound_type == 'music'
        assert perception.emotional_valence == 0.5
        assert perception.confidence == 0.8
        
    def test_visual_expression_creation(self):
        """Test VisualExpression creation."""
        expression = VisualExpression(
            style='organic',
            intensity=0.75,
            colors=['#228B22', '#32CD32', '#006400'],
            composition={'shapes': ['circle', 'wave'], 'patterns': ['flowing']},
            confidence=0.85
        )
        
        assert expression.style == 'organic'
        assert expression.intensity == 0.75
        assert len(expression.colors) == 3
        
    def test_adaptation_parameters_creation(self):
        """Test AdaptationParameters creation."""
        params = AdaptationParameters()
        
        assert params.learning_rate == 0.1
        assert params.feature_extraction_params == {}
        assert params.perception_params == {}
        assert params.expression_params == {}
        assert params.rendering_params == {}
        assert params.adaptation_history == []
        
    def test_adaptation_parameters_with_values(self):
        """Test AdaptationParameters with custom values."""
        params = AdaptationParameters(
            learning_rate=0.15,
            feature_extraction_params={"quality_boost": 0.2},
            perception_params={"emotion_sensitivity": 0.8},
            adaptation_history=[{"timestamp": datetime.now(), "change": "test"}]
        )
        
        assert params.learning_rate == 0.15
        assert params.feature_extraction_params["quality_boost"] == 0.2
        assert params.perception_params["emotion_sensitivity"] == 0.8
        assert len(params.adaptation_history) == 1
