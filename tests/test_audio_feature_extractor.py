# U-CogNet Audio-Visual Module Tests - Feature Extractor
# Comprehensive testing for audio feature extraction implementations

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock
from ucognet.modules.audio.feature_extractor import (
    LibrosaFeatureExtractor, FallbackFeatureExtractor
)
from ucognet.modules.audio.audio_types import AudioFeatures


class TestLibrosaFeatureExtractor:
    """Test LibrosaFeatureExtractor implementation."""

    @pytest.fixture
    def extractor(self):
        """Create a LibrosaFeatureExtractor instance."""
        return LibrosaFeatureExtractor()

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data for testing."""
        # Generate 1 second of 440 Hz sine wave
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = 440.0
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        return audio.astype(np.float32)

    @pytest.mark.asyncio
    async def test_initialization(self, extractor):
        """Test extractor initialization."""
        config = {
            'fft_window_size': 1024,
            'hop_length': 256,
            'enable_mfcc': True,
            'enable_chroma': True
        }

        await librosa_extractor.initialize(config)
        # Should not raise any exceptions
        assert extractor is not None

    @pytest.mark.asyncio
    async def test_feature_extraction_basic(self, librosa_extractor, sample_audio_frame):
        """Test basic feature extraction."""
        features = await librosa_extractor.extract_features(sample_audio_frame)

        assert isinstance(features, AudioFeatures)
        assert isinstance(features.timestamp, datetime)
        assert isinstance(features.mfcc, np.ndarray)
        assert isinstance(features.chroma, np.ndarray)
        assert isinstance(features.spectral_centroid, (int, float))
        assert isinstance(features.zero_crossing_rate, (int, float))
        assert isinstance(features.rms_energy, (int, float))
        assert isinstance(features.harmonic_ratio, (int, float))
        assert isinstance(features.percussive_ratio, (int, float))
        assert isinstance(features.onset_strength, (int, float))
        assert isinstance(features.tempo, (int, float))

    @pytest.mark.asyncio
    async def test_feature_extraction_with_context(self, librosa_extractor, sample_audio_frame):
        """Test feature extraction with context parameters."""
        context = {
            'quality_preset': 'high_quality',
            'fft_window_size': 2048,
            'hop_length': 512
        }

        features = await librosa_extractor.extract_features(sample_audio, context)

        assert isinstance(features, AudioFeatures)
        # With higher quality settings, we might get different dimensions
        assert features.mfcc.shape[0] == 13  # MFCC coefficients
        assert features.chroma.shape[0] == 12  # Chroma bins

    @pytest.mark.asyncio
    async def test_different_audio_formats(self, extractor):
        """Test feature extraction with different audio formats."""
        # Test with numpy array (float64)
        audio_f64 = np.random.rand(22050).astype(np.float64)
        features_f64 = await librosa_extractor.extract_features(audio_f64)
        assert isinstance(features_f64, AudioFeatures)

        # Test with numpy array (int16) - common audio format
        audio_i16 = (np.random.rand(22050) * 32767).astype(np.int16)
        features_i16 = await librosa_extractor.extract_features(audio_i16)
        assert isinstance(features_i16, AudioFeatures)

        # Test with bytes (mock WAV data)
        mock_wav_bytes = b'RIFF' + b'\x00' * 44 + np.random.bytes(44100)
        features_bytes = await librosa_extractor.extract_features(mock_wav_bytes)
        assert isinstance(features_bytes, AudioFeatures)

    @pytest.mark.asyncio
    async def test_feature_ranges(self, librosa_extractor, sample_audio_frame):
        """Test that extracted features are in expected ranges."""
        features = await librosa_extractor.extract_features(sample_audio_frame)

        # Check value ranges
        assert 0 <= features.zero_crossing_rate <= 1
        assert features.rms_energy >= 0
        assert 0 <= features.harmonic_ratio <= 1
        assert 0 <= features.percussive_ratio <= 1
        assert features.onset_strength >= 0
        assert features.tempo >= 0
        assert features.spectral_centroid >= 0

        # Check array shapes
        assert features.mfcc.ndim == 2
        assert features.chroma.ndim == 2
        assert features.mfcc.shape[0] == 13  # Default MFCC coefficients
        assert features.chroma.shape[0] == 12  # Default chroma bins

    @pytest.mark.asyncio
    async def test_empty_audio_handling(self, extractor):
        """Test handling of empty or very short audio."""
        # Empty audio
        empty_audio = np.array([])
        features = await librosa_extractor.extract_features(empty_audio)
        assert isinstance(features, AudioFeatures)

        # Very short audio
        short_audio = np.random.rand(100)  # Very short
        features_short = await librosa_extractor.extract_features(short_audio)
        assert isinstance(features_short, AudioFeatures)

    @pytest.mark.asyncio
    async def test_cleanup(self, extractor):
        """Test cleanup method."""
        await librosa_extractor.initialize({})
        await librosa_extractor.cleanup()
        # Should not raise any exceptions
        assert extractor is not None

    @pytest.mark.asyncio
    async def test_librosa_not_available(self):
        """Test behavior when librosa is not available."""
        with patch.dict('sys.modules', {'librosa': None}):
            extractor = LibrosaFeatureExtractor()
            audio = np.random.rand(22050)

            # Should still work (fallback behavior)
            features = await librosa_extractor.extract_features(audio)
            assert isinstance(features, AudioFeatures)


class TestFallbackFeatureExtractor:
    """Test FallbackFeatureExtractor implementation."""

    @pytest.fixture
    def extractor(self):
        """Create a FallbackFeatureExtractor instance."""
        return FallbackFeatureExtractor()

    @pytest.fixture
    def sample_audio(self):
        from ucognet.modules.audio.audio_types import AudioFrame
        """Create sample audio frame for testing."""
        audio_data = np.random.rand(22050).astype(np.float32)
        return AudioFrame(
            timestamp=datetime.now(),
            data=audio_data,
            sample_rate=22050,
            channels=1,
            duration=1.0
        )

    @pytest.mark.asyncio
    async def test_initialization(self, extractor):
        """Test extractor initialization."""
        config = {'fallback_quality': 'basic'}
        await extractor.initialize(config)
        assert extractor is not None

    @pytest.mark.asyncio
    async def test_feature_extraction_basic(self, extractor, sample_audio):
        """Test basic feature extraction with fallback."""
        features = await extractor.extract_features(sample_audio)

        assert isinstance(features, AudioFeatures)
        assert isinstance(features.timestamp, datetime)
        assert isinstance(features.mfcc, np.ndarray)
        assert isinstance(features.chroma, np.ndarray)

        # Check that we have reasonable default values
        assert features.spectral_centroid >= 0
        assert 0 <= features.zero_crossing_rate <= 1
        assert features.rms_energy >= 0

    @pytest.mark.asyncio
    async def test_fallback_feature_computation(self, extractor, sample_audio):
        """Test that fallback computes features correctly."""
        features = await extractor.extract_features(sample_audio)

        # RMS energy should be computed correctly
        expected_rms = np.sqrt(np.mean(sample_audio.data ** 2))
        assert abs(features.rms_energy - expected_rms) < 0.01

        # Zero crossing rate should be reasonable
        assert 0 <= features.zero_crossing_rate <= 1

        # Spectral centroid should be positive
        assert features.spectral_centroid > 0

    @pytest.mark.asyncio
    async def test_fallback_with_different_audio_lengths(self, extractor):
        """Test fallback with different audio lengths."""
        from ucognet.modules.audio.audio_types import AudioFrame
        # Short audio
        short_audio_data = np.random.rand(1000)
        short_audio_frame = AudioFrame(
            timestamp=datetime.now(),
            data=short_audio_data.astype(np.float32),
            sample_rate=22050,
            channels=1,
            duration=len(short_audio_data) / 22050
        )
        features_short = await extractor.extract_features(short_audio_frame)
        assert isinstance(features_short, AudioFeatures)

        # Long audio
        long_audio_data = np.random.rand(88200)  # 4 seconds at 22050 Hz
        long_audio_frame = AudioFrame(
            timestamp=datetime.now(),
            data=long_audio_data.astype(np.float32),
            sample_rate=22050,
            channels=1,
            duration=len(long_audio_data) / 22050
        )
        features_long = await extractor.extract_features(long_audio_frame)
        assert isinstance(features_long, AudioFeatures)

    @pytest.mark.asyncio
    async def test_fallback_mfcc_generation(self, extractor, sample_audio):
        """Test MFCC generation in fallback mode."""
        features = await extractor.extract_features(sample_audio)

        # Should generate some MFCC-like features
        assert features.mfcc.shape[0] > 0
        assert features.mfcc.shape[1] > 0

        # Values should be reasonable (not all zeros, not extreme)
        assert np.std(features.mfcc) > 0
        assert np.max(np.abs(features.mfcc)) < 1000

    @pytest.mark.asyncio
    async def test_fallback_chroma_generation(self, extractor, sample_audio):
        """Test chroma generation in fallback mode."""
        features = await extractor.extract_features(sample_audio)

        # Should generate some chroma-like features
        assert features.chroma.shape[0] > 0
        assert features.chroma.shape[1] > 0

        # Values should be reasonable
        assert np.std(features.chroma) > 0
        assert np.max(np.abs(features.chroma)) < 1000

    @pytest.mark.asyncio
    async def test_fallback_tempo_estimation(self, extractor):
        """Test tempo estimation in fallback mode."""
        from ucognet.modules.audio.audio_types import AudioFrame
        # Create audio with known rhythm
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # 120 BPM = 2 Hz
        frequency = 2.0
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        audio_frame = AudioFrame(
            timestamp=datetime.now(),
            data=audio_data.astype(np.float32),
            sample_rate=sample_rate,
            channels=1,
            duration=duration
        )

        features = await extractor.extract_features(audio_frame)

        # Tempo should be estimated (may not be perfect but should be reasonable)
        assert features.tempo > 0
        assert features.tempo < 300  # Reasonable upper bound

    @pytest.mark.asyncio
    async def test_cleanup(self, extractor):
        """Test cleanup method."""
        await extractor.initialize({})
        await extractor.cleanup()
        assert extractor is not None


class TestFeatureExtractorComparison:
    """Test comparison between Librosa and Fallback extractors."""

    @pytest.fixture
    def librosa_extractor(self):
        """Create Librosa extractor."""
        return LibrosaFeatureExtractor()

    @pytest.fixture
    def fallback_extractor(self):
        """Create Fallback extractor."""
        return FallbackFeatureExtractor()

    @pytest.fixture
    def test_audio(self):
        """Create consistent test audio."""
        np.random.seed(42)  # For reproducible results
        return np.random.rand(22050).astype(np.float32)

    @pytest.mark.asyncio
    async def test_extractors_produce_similar_structure(self, librosa_extractor, fallback_extractor, test_audio):
        """Test that both extractors produce features with similar structure."""
        librosa_features = await librosa_extractor.extract_features(test_audio)
        fallback_features = await fallback_extractor.extract_features(test_audio)

        # Both should produce AudioFeatures objects
        assert isinstance(librosa_features, AudioFeatures)
        assert isinstance(fallback_features, AudioFeatures)

        # Both should have same basic attributes
        attrs_to_check = [
            'spectral_centroid', 'zero_crossing_rate', 'rms_energy',
            'harmonic_ratio', 'percussive_ratio', 'onset_strength', 'tempo'
        ]

        for attr in attrs_to_check:
            assert hasattr(librosa_features, attr)
            assert hasattr(fallback_features, attr)
            # Values should be numeric
            assert isinstance(getattr(librosa_features, attr), (int, float))
            assert isinstance(getattr(fallback_features, attr), (int, float))

    @pytest.mark.asyncio
    async def test_extractors_handle_edge_cases(self, librosa_extractor, fallback_extractor):
        """Test that both extractors handle edge cases similarly."""
        # Silent audio
        silent_audio = np.zeros(22050)
        librosa_silent = await librosa_extractor.extract_features(silent_audio)
        fallback_silent = await fallback_extractor.extract_features(silent_audio)

        assert librosa_silent.rms_energy < 0.01
        assert fallback_silent.rms_energy < 0.01

        # Loud audio
        loud_audio = np.ones(22050)
        librosa_loud = await librosa_extractor.extract_features(loud_audio)
        fallback_loud = await fallback_extractor.extract_features(loud_audio)

        assert librosa_loud.rms_energy > 0.5
        assert fallback_loud.rms_energy > 0.5
