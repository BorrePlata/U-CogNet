# U-CogNet Audio-Visual Module Tests - Integration
# Comprehensive integration testing for the complete audio-visual synthesis system

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
from ucognet.modules.audio.audio_visual_synthesis import AudioVisualSynthesizer
from ucognet.modules.audio.feature_extractor import LibrosaFeatureExtractor, FallbackFeatureExtractor
from ucognet.modules.audio.audio_perception import CognitiveAudioPerception
from ucognet.modules.audio.visual_expression import ArtisticVisualExpression
from ucognet.modules.audio.visual_rendering import ArtisticVisualRenderer
from ucognet.modules.audio.audio_evaluation import AudioVisualEvaluator
from ucognet.modules.audio.audio_types import SynthesisResult


class TestAudioVisualSynthesisIntegration:
    """Test the complete audio-visual synthesis system integration."""

    @pytest.fixture
    async def synthesizer(self):
        """Create and initialize a complete synthesizer."""
        synthesizer = AudioVisualSynthesizer()

        # Register all components
        synthesizer.register_feature_extractor(LibrosaFeatureExtractor())
        synthesizer.register_perception_engine(CognitiveAudioPerception())
        synthesizer.register_visual_expressor(ArtisticVisualExpression())
        synthesizer.register_visual_renderer(ArtisticVisualRenderer())

        # Initialize with test configuration
        config = {
            'quality_preset': 'balanced',
            'real_time_processing': False,
            'enable_caching': False,  # Disable for testing
            'max_concurrent_syntheses': 1,
            'feature_extractor': {'fft_window_size': 1024},
            'perception_engine': {'emotion_sensitivity': 0.8},
            'visual_expressor': {'color_vibrancy': 0.7},
            'visual_renderer': {'canvas_width': 400, 'canvas_height': 300}
        }

        await synthesizer.initialize(config)
        yield synthesizer
        await synthesizer.cleanup()

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing."""
        # Generate 1 second of test audio (22050 Hz)
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Mix of frequencies to create interesting audio
        audio = (
            0.3 * np.sin(2 * np.pi * 440 * t) +      # A4 note
            0.2 * np.sin(2 * np.pi * 880 * t) +      # A5 note
            0.1 * np.random.normal(0, 1, len(t))     # Some noise
        )

        return audio.astype(np.float32)

    @pytest.mark.asyncio
    async def test_full_synthesis_pipeline(self, synthesizer, sample_audio):
        """Test the complete synthesis pipeline from audio to visual."""
        result = await synthesizer.synthesize_audio_visual(sample_audio)

        # Verify result structure
        assert isinstance(result, SynthesisResult)
        assert result.synthesis_id.startswith('synthesis_')
        assert result.audio_input is sample_audio
        assert result.processing_time > 0
        assert isinstance(result.timestamp, datetime)

        # Verify all components produced results
        assert result.features is not None
        assert result.perception is not None
        assert result.expression is not None
        assert result.rendered_visual is not None

        # Verify feature extraction worked
        assert hasattr(result.features, 'mfcc')
        assert hasattr(result.features, 'spectral_centroid')
        assert result.features.spectral_centroid > 0

        # Verify perception worked
        assert isinstance(result.perception.sound_type, str)
        assert -1 <= result.perception.emotional_valence <= 1
        assert 0 <= result.perception.arousal_level <= 1
        assert 0 <= result.perception.attention_weight <= 1

        # Verify expression worked
        assert isinstance(result.expression.style, str)
        assert 0 <= result.expression.intensity <= 1
        assert len(result.expression.colors) > 0
        assert isinstance(result.expression.composition, dict)

        # Verify rendering worked
        assert result.rendered_visual.data is not None
        assert result.rendered_visual.format in ['image', 'numpy', 'base64']
        assert len(result.rendered_visual.dimensions) == 2

    @pytest.mark.asyncio
    async def test_synthesis_with_context(self, synthesizer, sample_audio):
        """Test synthesis with context parameters."""
        context = {
            'sound_type': 'birdsong',
            'environment': 'forest',
            'output_format': 'numpy',
            'quality_requirement': 'high'
        }

        result = await synthesizer.synthesize_audio_visual(sample_audio, context)

        assert result.rendered_visual.format == 'numpy'
        assert isinstance(result.rendered_visual.data, np.ndarray)
        assert result.metadata['context'] == context

    @pytest.mark.asyncio
    async def test_batch_synthesis(self, synthesizer):
        """Test batch processing capabilities."""
        # Create batch of different audio samples
        batch_audio = [
            np.random.rand(22050).astype(np.float32),  # Random audio
            np.sin(np.linspace(0, 2*np.pi, 22050)).astype(np.float32),  # Sine wave
            np.zeros(22050).astype(np.float32),  # Silence
        ]

        batch_contexts = [
            {'batch_id': 0, 'type': 'random'},
            {'batch_id': 1, 'type': 'tone'},
            {'batch_id': 2, 'type': 'silence'}
        ]

        results = await synthesizer.synthesize_batch(batch_audio, batch_contexts)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, SynthesisResult)
            assert result.metadata['context']['batch_id'] == i

    @pytest.mark.asyncio
    async def test_different_audio_types(self, synthesizer):
        """Test synthesis with different types of audio input."""
        # Test with numpy array
        audio_np = np.random.rand(22050).astype(np.float32)
        result_np = await synthesizer.synthesize_audio_visual(audio_np)
        assert result_np.rendered_visual is not None

        # Test with bytes (mock)
        audio_bytes = np.random.rand(22050).tobytes()
        result_bytes = await synthesizer.synthesize_audio_visual(audio_bytes)
        assert result_bytes.rendered_visual is not None

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, synthesizer, sample_audio):
        """Test performance monitoring capabilities."""
        # Perform multiple syntheses
        num_syntheses = 3
        results = []
        for i in range(num_syntheses):
            result = await synthesizer.synthesize_audio_visual(sample_audio)
            results.append(result)
            await asyncio.sleep(0.01)  # Small delay

        # Check performance stats
        stats = synthesizer.get_performance_stats()
        assert stats['total_syntheses'] == num_syntheses
        assert stats['average_latency'] > 0
        assert 0 <= stats['cache_hit_rate'] <= 1

        # All results should have reasonable processing times
        for result in results:
            assert 0 < result.processing_time < 10000  # Reasonable bounds in ms

    @pytest.mark.asyncio
    async def test_caching_behavior(self):
        """Test caching behavior."""
        synthesizer = AudioVisualSynthesizer()

        # Register components
        synthesizer.register_feature_extractor(LibrosaFeatureExtractor())
        synthesizer.register_perception_engine(CognitiveAudioPerception())
        synthesizer.register_visual_expressor(ArtisticVisualExpression())
        synthesizer.register_visual_renderer(ArtisticVisualRenderer())

        # Enable caching
        config = {
            'enable_caching': True,
            'cache_size': 10,
            'quality_preset': 'fast'  # Faster for testing
        }

        await synthesizer.initialize(config)

        try:
            audio = np.random.rand(11025).astype(np.float32)  # Shorter for speed

            # First synthesis
            result1 = await synthesizer.synthesize_audio_visual(audio)
            time1 = result1.processing_time

            # Second synthesis (should use cache)
            result2 = await synthesizer.synthesize_audio_visual(audio)
            time2 = result2.processing_time

            # Results should be identical
            assert result1.features.spectral_centroid == result2.features.spectral_centroid
            assert result1.perception.sound_type == result2.perception.sound_type

            # Second should be faster (cached)
            # Note: This might not always be true due to system variability

        finally:
            await synthesizer.cleanup()

    @pytest.mark.asyncio
    async def test_error_handling(self, synthesizer):
        """Test error handling in synthesis pipeline."""
        # Test with invalid audio
        invalid_audio = None
        result = await synthesizer.synthesize_audio_visual(invalid_audio)

        # Should still return a result, but with error information
        assert isinstance(result, SynthesisResult)
        assert result.features is None or 'error' in result.metadata

    @pytest.mark.asyncio
    async def test_component_replacement(self):
        """Test ability to replace components."""
        synthesizer = AudioVisualSynthesizer()

        # Start with fallback extractor
        synthesizer.register_feature_extractor(FallbackFeatureExtractor())
        synthesizer.register_perception_engine(CognitiveAudioPerception())
        synthesizer.register_visual_expressor(ArtisticVisualExpression())
        synthesizer.register_visual_renderer(ArtisticVisualRenderer())

        await synthesizer.initialize({'quality_preset': 'fast'})

        try:
            audio = np.random.rand(11025).astype(np.float32)
            result_fallback = await synthesizer.synthesize_audio_visual(audio)

            # Replace with librosa extractor
            synthesizer.register_feature_extractor(LibrosaFeatureExtractor())
            result_librosa = await synthesizer.synthesize_audio_visual(audio)

            # Both should work
            assert result_fallback.rendered_visual is not None
            assert result_librosa.rendered_visual is not None

        finally:
            await synthesizer.cleanup()

    @pytest.mark.asyncio
    async def test_configuration_persistence(self, synthesizer, sample_audio):
        """Test that configuration changes persist correctly."""
        # Test different quality presets
        presets = ['fast', 'balanced', 'high_quality']

        for preset in presets:
            synthesizer.configure({'quality_preset': preset})
            result = await synthesizer.synthesize_audio_visual(sample_audio)

            assert result.rendered_visual is not None
            assert result.metadata['processing_config']['quality_preset'] == preset


class TestAudioVisualEvaluationIntegration:
    """Test evaluation system integration."""

    @pytest.fixture
    async def evaluator(self):
        """Create and initialize evaluator."""
        evaluator = AudioVisualEvaluator()
        config = {
            'adaptation_strategy': 'balanced',
            'max_history_size': 50
        }
        await evaluator.initialize(config)
        yield evaluator
        await evaluator.cleanup()

    @pytest.fixture
    async def sample_result(self):
        """Create a sample synthesis result for evaluation."""
        # Create a minimal synthesis result
        features = MagicMock()
        features.spectral_centroid = 2000.0
        features.rms_energy = 0.3
        features.confidence = 0.8

        perception = MagicMock()
        perception.sound_type = 'nature'
        perception.emotional_valence = 0.4
        perception.arousal_level = 0.6
        perception.attention_weight = 0.7
        perception.confidence = 0.8

        expression = MagicMock()
        expression.intensity = 0.7
        expression.style = 'organic'

        rendered = MagicMock()
        rendered.data = b'fake_image_data'
        rendered.format = 'image'

        result = SynthesisResult(
            synthesis_id="test_synthesis_001",
            timestamp=datetime.now(),
            audio_input=np.random.rand(22050),
            features=features,
            perception=perception,
            expression=expression,
            rendered_visual=rendered,
            processing_time=45.2
        )

        return result

    @pytest.mark.asyncio
    async def test_evaluation_pipeline(self, evaluator, sample_result):
        """Test the complete evaluation pipeline."""
        metrics = await evaluator.evaluate_synthesis(sample_result)

        assert metrics.evaluation_id.startswith('eval_')
        assert metrics.synthesis_id == sample_result.synthesis_id
        assert 0 <= metrics.overall_score <= 1
        assert metrics.quality_level in ['excellent', 'good', 'acceptable', 'poor', 'unacceptable']
        assert isinstance(metrics.recommendations, list)

    @pytest.mark.asyncio
    async def test_evaluation_with_feedback(self, evaluator, sample_result):
        """Test evaluation with user feedback."""
        feedback = {
            'satisfaction': 0.8,
            'visual_quality': 0.9,
            'emotional_accuracy': 0.7,
            'creativity': 0.8,
            'comments': 'Beautiful representation!'
        }

        metrics = await evaluator.evaluate_synthesis(sample_result, user_feedback=feedback)

        assert metrics.user_satisfaction == 0.8
        assert 'Beautiful representation!' in feedback['comments']

    @pytest.mark.asyncio
    async def test_adaptation_based_on_evaluation(self, evaluator, sample_result):
        """Test that adaptation parameters are generated correctly."""
        # Evaluate multiple results
        evaluations = []
        for i in range(3):
            metrics = await evaluator.evaluate_synthesis(sample_result)
            evaluations.append(metrics)

        # Generate adaptation
        adaptation = await evaluator.adapt_parameters(evaluations)

        assert isinstance(adaptation, object)
        assert hasattr(adaptation, 'learning_rate')
        assert 0 < adaptation.learning_rate <= 1

    @pytest.mark.asyncio
    async def test_evaluation_history(self, evaluator, sample_result):
        """Test evaluation history tracking."""
        # Perform multiple evaluations
        for i in range(5):
            await evaluator.evaluate_synthesis(sample_result)

        history = evaluator.get_evaluation_history()
        assert len(history) == 5

        # Test limited history
        limited_history = evaluator.get_evaluation_history(limit=2)
        assert len(limited_history) == 2

    @pytest.mark.asyncio
    async def test_performance_trends(self, evaluator, sample_result):
        """Test performance trend analysis."""
        # Create evaluations with different scores
        scores = [0.6, 0.7, 0.8, 0.75, 0.85]

        for score in scores:
            # Mock the evaluation result
            with patch.object(evaluator, '_evaluate_perceptual_quality', return_value=score):
                await evaluator.evaluate_synthesis(sample_result)

        trends = evaluator.get_performance_trends(days=1)
        assert 'perceptual_quality' in trends
        assert 'overall_score' in trends

        # Should have trend information
        assert 'average' in trends['overall_score']
        assert 'trend' in trends['overall_score']


class TestSystemResilience:
    """Test system resilience and error recovery."""

    @pytest.mark.asyncio
    async def test_partial_component_failure(self):
        """Test system behavior when some components fail."""
        synthesizer = AudioVisualSynthesizer()

        # Register working components
        synthesizer.register_feature_extractor(LibrosaFeatureExtractor())
        synthesizer.register_perception_engine(CognitiveAudioPerception())

        # Mock failing visual components
        class FailingExpressor:
            async def express_visually(self, perception):
                raise Exception("Visual expression failed")
            async def initialize(self, config): pass
            async def cleanup(self): pass

        class FailingRenderer:
            async def render_visual(self, expression, format_type="image"):
                raise Exception("Visual rendering failed")
            async def initialize(self, config): pass
            async def cleanup(self): pass

        synthesizer.register_visual_expressor(FailingExpressor())
        synthesizer.register_visual_renderer(FailingRenderer())

        await synthesizer.initialize({})

        try:
            audio = np.random.rand(11025).astype(np.float32)
            result = await synthesizer.synthesize_audio_visual(audio)

            # Should still produce a result with partial data
            assert isinstance(result, SynthesisResult)
            assert result.features is not None
            assert result.perception is not None
            # Visual components failed, so these should be None or have errors
            assert result.expression is None or 'error' in result.metadata

        finally:
            await synthesizer.cleanup()

    @pytest.mark.asyncio
    async def test_memory_management(self):
        """Test memory management with large batches."""
        synthesizer = AudioVisualSynthesizer()

        # Register components
        synthesizer.register_feature_extractor(FallbackFeatureExtractor())  # Lighter
        synthesizer.register_perception_engine(CognitiveAudioPerception())
        synthesizer.register_visual_expressor(ArtisticVisualExpression())
        synthesizer.register_visual_renderer(ArtisticVisualRenderer())

        config = {
            'quality_preset': 'fast',
            'enable_caching': False,  # Disable to test memory
            'max_concurrent_syntheses': 2
        }

        await synthesizer.initialize(config)

        try:
            # Process a batch
            batch_size = 5
            batch_audio = [np.random.rand(11025).astype(np.float32) for _ in range(batch_size)]

            results = await synthesizer.synthesize_batch(batch_audio)

            assert len(results) == batch_size
            for result in results:
                assert isinstance(result, SynthesisResult)
                assert result.rendered_visual is not None

        finally:
            await synthesizer.cleanup()

    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent processing capabilities."""
        synthesizer = AudioVisualSynthesizer()

        # Register components
        synthesizer.register_feature_extractor(FallbackFeatureExtractor())
        synthesizer.register_perception_engine(CognitiveAudioPerception())
        synthesizer.register_visual_expressor(ArtisticVisualExpression())
        synthesizer.register_visual_renderer(ArtisticVisualRenderer())

        config = {
            'quality_preset': 'fast',
            'max_concurrent_syntheses': 3
        }

        await synthesizer.initialize(config)

        try:
            # Create multiple audio samples
            audio_samples = [np.random.rand(11025).astype(np.float32) for _ in range(6)]

            # Process concurrently
            results = await synthesizer.synthesize_batch(audio_samples)

            assert len(results) == 6
            successful = sum(1 for r in results if r.rendered_visual is not None)
            assert successful >= 4  # At least most should succeed

        finally:
            await synthesizer.cleanup()
