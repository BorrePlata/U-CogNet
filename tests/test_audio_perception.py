# U-CogNet Audio-Visual Module Tests - Cognitive Perception
# Comprehensive testing for audio perception engine

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock
from ucognet.modules.audio.audio_perception import CognitiveAudioPerception
from ucognet.modules.audio.audio_types import AudioFeatures, AudioPerception


class TestCognitiveAudioPerception:
    """Test CognitiveAudioPerception implementation."""

    @pytest.fixture
    def perception_engine(self):
        """Create a CognitiveAudioPerception instance."""
        return CognitiveAudioPerception()

    @pytest.fixture
    def sample_features(self):
        """Create sample audio features for testing."""
        return AudioFeatures(
            timestamp=datetime.now(),
            mfcc=np.random.rand(13, 50),
            chroma=np.random.rand(12, 50),
            spectral_centroid=2500.0,
            zero_crossing_rate=0.15,
            rms_energy=0.3,
            harmonic_ratio=0.7,
            percussive_ratio=0.3,
            onset_strength=0.6,
            tempo=120.0,
            beat_positions=np.array([0.1, 0.5, 1.0]),
            confidence=0.85
        )

    @pytest.mark.asyncio
    async def test_initialization(self, perception_engine):
        """Test perception engine initialization."""
        config = {
            'emotion_sensitivity': 0.8,
            'context_awareness': 0.9
        }

        await perception_engine.initialize(config)
        assert perception_engine is not None

    @pytest.mark.asyncio
    async def test_perceive_audio_basic(self, perception_engine, sample_features):
        """Test basic audio perception."""
        perception = await perception_engine.perceive_audio(sample_features)

        assert isinstance(perception, AudioPerception)
        assert isinstance(perception.timestamp, datetime)
        assert perception.features is sample_features
        assert isinstance(perception.sound_type, str)
        assert isinstance(perception.emotional_valence, (int, float))
        assert isinstance(perception.arousal_level, (int, float))
        assert isinstance(perception.familiarity, (int, float))
        assert isinstance(perception.environment_context, str)
        assert isinstance(perception.temporal_pattern, str)
        assert isinstance(perception.spatial_characteristics, dict)
        assert isinstance(perception.attention_weight, (int, float))
        assert isinstance(perception.memory_importance, (int, float))

    @pytest.mark.asyncio
    async def test_sound_type_classification(self, perception_engine):
        """Test sound type classification for different feature patterns."""
        # Test birdsong features
        birdsong_features = AudioFeatures(
            timestamp=datetime.now(),
            mfcc=np.random.rand(13, 50),
            chroma=np.random.rand(12, 50),
            spectral_centroid=3500.0,  # High frequency
            zero_crossing_rate=0.12,
            rms_energy=0.2,
            harmonic_ratio=0.8,  # Very harmonic
            percussive_ratio=0.2,
            onset_strength=0.4,
            tempo=140.0  # Fast tempo
        )

        sound_type = await perception_engine.classify_sound_type(birdsong_features)
        # Should classify as birdsong or similar
        assert sound_type in ['birdsong', 'nature', 'unknown']

        # Test explosion features
        explosion_features = AudioFeatures(
            timestamp=datetime.now(),
            mfcc=np.random.rand(13, 50),
            chroma=np.random.rand(12, 50),
            spectral_centroid=1500.0,  # Lower frequency
            zero_crossing_rate=0.08,
            rms_energy=0.5,  # High energy
            harmonic_ratio=0.3,  # Not very harmonic
            percussive_ratio=0.7,  # Very percussive
            onset_strength=0.9,  # Strong onset
            tempo=80.0
        )

        explosion_type = await perception_engine.classify_sound_type(explosion_features)
        assert explosion_type in ['explosion', 'urban', 'unknown']

    @pytest.mark.asyncio
    async def test_emotional_assessment(self, perception_engine):
        """Test emotional content assessment."""
        # Test positive emotional features (harmonic, moderate tempo)
        positive_features = AudioFeatures(
            timestamp=datetime.now(),
            mfcc=np.random.rand(13, 50),
            chroma=np.random.rand(12, 50),
            spectral_centroid=2000.0,
            zero_crossing_rate=0.1,
            rms_energy=0.3,
            harmonic_ratio=0.8,  # High harmony = positive
            percussive_ratio=0.2,
            onset_strength=0.5,
            tempo=110.0  # Moderate tempo
        )

        emotions = await perception_engine.assess_emotional_content(positive_features)
        assert 'valence' in emotions
        assert 'arousal' in emotions
        assert 'emotional_intensity' in emotions

        # Valence should be positive due to high harmonic ratio
        assert emotions['valence'] > 0
        assert 0 <= emotions['arousal'] <= 1

        # Test negative emotional features (discordant, high energy)
        negative_features = AudioFeatures(
            timestamp=datetime.now(),
            mfcc=np.random.rand(13, 50),
            chroma=np.random.rand(12, 50),
            spectral_centroid=800.0,  # Low frequency
            zero_crossing_rate=0.05,
            rms_energy=0.6,  # High energy
            harmonic_ratio=0.2,  # Low harmony
            percussive_ratio=0.8,  # High percussion
            onset_strength=0.8,
            tempo=60.0  # Slow tempo
        )

        neg_emotions = await perception_engine.assess_emotional_content(negative_features)
        # Should be more negative valence
        assert neg_emotions['valence'] < emotions['valence']

    @pytest.mark.asyncio
    async def test_environment_contextualization(self, perception_engine):
        """Test environment contextualization."""
        # Forest-like features
        forest_features = AudioFeatures(
            timestamp=datetime.now(),
            mfcc=np.random.rand(13, 50),
            chroma=np.random.rand(12, 50),
            spectral_centroid=1800.0,
            zero_crossing_rate=0.08,
            rms_energy=0.15,  # Quiet
            harmonic_ratio=0.75,  # Harmonic
            percussive_ratio=0.25,
            onset_strength=0.3,
            tempo=90.0
        )

        context = await perception_engine.contextualize_environment(forest_features)
        assert isinstance(context, str)
        assert context in ['forest', 'nature', 'unknown']

        # Urban-like features
        urban_features = AudioFeatures(
            timestamp=datetime.now(),
            mfcc=np.random.rand(13, 50),
            chroma=np.random.rand(12, 50),
            spectral_centroid=1200.0,
            zero_crossing_rate=0.12,  # Higher ZCR
            rms_energy=0.4,
            harmonic_ratio=0.4,  # Less harmonic
            percussive_ratio=0.6,
            onset_strength=0.7,
            tempo=100.0
        )

        urban_context = await perception_engine.contextualize_environment(urban_features)
        assert isinstance(urban_context, str)

    @pytest.mark.asyncio
    async def test_temporal_pattern_analysis(self, perception_engine):
        """Test temporal pattern analysis."""
        # Pulsed pattern features
        pulsed_features = AudioFeatures(
            timestamp=datetime.now(),
            mfcc=np.random.rand(13, 50),
            chroma=np.random.rand(12, 50),
            spectral_centroid=2000.0,
            zero_crossing_rate=0.1,
            rms_energy=0.4,  # High energy
            harmonic_ratio=0.5,
            percussive_ratio=0.5,
            onset_strength=0.8,  # Strong onset
            tempo=100.0
        )

        perception = await perception_engine.perceive_audio(pulsed_features)
        assert perception.temporal_pattern in ['pulsed', 'rhythmic', 'continuous', 'intermittent', 'random']

    @pytest.mark.asyncio
    async def test_attention_evaluation(self, perception_engine):
        """Test attention worthiness evaluation."""
        # Create a perception object for testing
        features = AudioFeatures(
            timestamp=datetime.now(),
            mfcc=np.random.rand(13, 50),
            chroma=np.random.rand(12, 50),
            spectral_centroid=2000.0,
            zero_crossing_rate=0.1,
            rms_energy=0.4,
            harmonic_ratio=0.5,
            percussive_ratio=0.5,
            onset_strength=0.8,
            tempo=100.0
        )

        perception = AudioPerception(
            timestamp=datetime.now(),
            features=features,
            sound_type="explosion",
            emotional_valence=-0.5,
            arousal_level=0.8,
            familiarity=0.3,
            environment_context="urban",
            temporal_pattern="pulsed",
            spatial_characteristics={"distance": 0.3, "direction": 0.0, "room_size": 0.5},
            attention_weight=0.0,  # Will be calculated
            memory_importance=0.0
        )

        attention_weight = await perception_engine.evaluate_attention_worthiness(perception)
        assert 0 <= attention_weight <= 1

        # High attention scenarios should get higher weights
        assert attention_weight > 0.5  # Explosion should be attention-worthy

    @pytest.mark.asyncio
    async def test_perception_consistency(self, perception_engine, sample_features):
        """Test that perception results are consistent for same input."""
        perception1 = await perception_engine.perceive_audio(sample_features)
        perception2 = await perception_engine.perceive_audio(sample_features)

        # Should produce similar results (not exactly same due to randomness)
        assert perception1.sound_type == perception2.sound_type
        assert abs(perception1.emotional_valence - perception2.emotional_valence) < 0.1
        assert abs(perception1.arousal_level - perception2.arousal_level) < 0.1

    @pytest.mark.asyncio
    async def test_edge_case_handling(self, perception_engine):
        """Test handling of edge cases."""
        # Features with extreme values
        extreme_features = AudioFeatures(
            timestamp=datetime.now(),
            mfcc=np.random.rand(13, 50),
            chroma=np.random.rand(12, 50),
            spectral_centroid=50.0,  # Very low
            zero_crossing_rate=0.0,  # No zero crossings
            rms_energy=0.0,  # Silent
            harmonic_ratio=0.0,  # No harmony
            percussive_ratio=1.0,  # All percussion
            onset_strength=0.0,  # No onset
            tempo=0.0  # No tempo
        )

        perception = await perception_engine.perceive_audio(extreme_features)
        assert isinstance(perception, AudioPerception)

        # Should still produce valid values
        assert 0 <= perception.attention_weight <= 1
        assert 0 <= perception.memory_importance <= 1

    @pytest.mark.asyncio
    async def test_configuration_updates(self, perception_engine):
        """Test that configuration updates work."""
        # Update sound type thresholds
        config = {
            'sound_type_thresholds': {
                'birdsong': {
                    'spectral_centroid_min': 4000,  # Higher threshold
                    'harmonic_ratio_min': 0.8
                }
            },
            'emotion_mappings': {
                'valence': {
                    'high_harmonic': 1.0  # Stronger harmonic influence
                }
            }
        }

        await perception_engine.initialize(config)

        # Test with features that should now be classified differently
        test_features = AudioFeatures(
            timestamp=datetime.now(),
            mfcc=np.random.rand(13, 50),
            chroma=np.random.rand(12, 50),
            spectral_centroid=3500.0,  # Below new threshold
            zero_crossing_rate=0.1,
            rms_energy=0.2,
            harmonic_ratio=0.75,  # Below new threshold
            percussive_ratio=0.25,
            onset_strength=0.4,
            tempo=120.0
        )

        sound_type = await perception_engine.classify_sound_type(test_features)
        # Should not be classified as birdsong with new thresholds
        assert sound_type != 'birdsong' or sound_type == 'unknown'

    @pytest.mark.asyncio
    async def test_cleanup(self, perception_engine):
        """Test cleanup method."""
        await perception_engine.initialize({})
        await perception_engine.cleanup()
        assert perception_engine is not None


class TestPerceptionMetrics:
    """Test perception quality metrics."""

    @pytest.fixture
    def perception_engine(self):
        """Create a CognitiveAudioPerception instance."""
        return CognitiveAudioPerception()

    @pytest.mark.asyncio
    async def test_confidence_calculation(self, perception_engine, sample_features):
        """Test confidence calculation in perception."""
        perception = await perception_engine.perceive_audio(sample_features)

        # Confidence should be between 0 and 1
        assert 0 <= perception.confidence <= 1

        # Higher confidence for clearer signals
        clear_features = AudioFeatures(
            timestamp=datetime.now(),
            mfcc=np.random.rand(13, 50),
            chroma=np.random.rand(12, 50),
            spectral_centroid=2000.0,
            zero_crossing_rate=0.1,
            rms_energy=0.4,  # Good signal strength
            harmonic_ratio=0.7,
            percussive_ratio=0.3,
            onset_strength=0.6,
            tempo=120.0,
            confidence=0.9  # High input confidence
        )

        clear_perception = await perception_engine.perceive_audio(clear_features)
        assert clear_perception.confidence > 0.5

    @pytest.mark.asyncio
    async def test_familiarity_estimation(self, perception_engine, sample_features):
        """Test familiarity estimation."""
        perception = await perception_engine.perceive_audio(sample_features)

        # Familiarity should be between 0 and 1
        assert 0 <= perception.familiarity <= 1

        # Currently returns default value (would be learned in real system)
        assert perception.familiarity == 0.5
