# U-CogNet Audio-Visual Module Tests - Evaluation
# Comprehensive testing for the audio-visual evaluation system

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from ucognet.modules.audio.audio_evaluation import AudioVisualEvaluator
from ucognet.modules.audio.audio_types import (
    SynthesisResult, EvaluationMetrics, AdaptationParameters,
    AudioFeatures, AudioPerception, VisualExpression, RenderedVisual
)


class TestAudioVisualEvaluator:
    """Test the AudioVisualEvaluator class."""

    @pytest.fixture
    async def evaluator(self):
        """Create and initialize evaluator."""
        evaluator = AudioVisualEvaluator()
        config = {
            'adaptation_strategy': 'balanced',
            'max_history_size': 100,
            'evaluation_weights': {
                'perceptual_quality': 0.4,
                'technical_quality': 0.3,
                'emotional_accuracy': 0.2,
                'creativity': 0.1
            },
            'learning_rate': 0.1
        }
        await evaluator.initialize(config)
        yield evaluator
        await evaluator.cleanup()

    @pytest.fixture
    def sample_synthesis_result(self):
        """Create a sample synthesis result for testing."""
        features = AudioFeatures(
            mfcc=np.random.rand(13, 10),
            spectral_centroid=2500.0,
            chroma=np.random.rand(12, 10),
            rms_energy=0.4,
            zero_crossing_rate=0.15,
            tempo=120.0,
            confidence=0.85
        )

        perception = AudioPerception(
            sound_type='nature',
            emotional_valence=0.6,
            arousal_level=0.7,
            attention_weight=0.8,
            environment_context='forest',
            confidence=0.9
        )

        expression = VisualExpression(
            style='organic',
            intensity=0.75,
            colors=['#228B22', '#32CD32', '#006400'],
            composition={'shapes': ['circle', 'wave'], 'patterns': ['flowing']},
            confidence=0.8
        )

        rendered = RenderedVisual(
            data=b'fake_image_data',
            format='image',
            dimensions=(400, 300),
            metadata={'quality': 'high'}
        )

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
    async def test_evaluator_initialization(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator.is_initialized
        assert evaluator.adaptation_strategy == 'balanced'
        assert evaluator.max_history_size == 100
        assert len(evaluator.evaluation_history) == 0

    @pytest.mark.asyncio
    async def test_evaluate_synthesis_basic(self, evaluator, sample_synthesis_result):
        """Test basic synthesis evaluation."""
        metrics = await evaluator.evaluate_synthesis(sample_synthesis_result)

        # Verify result structure
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.evaluation_id.startswith('eval_')
        assert metrics.synthesis_id == sample_synthesis_result.synthesis_id
        assert isinstance(metrics.timestamp, datetime)
        assert 0 <= metrics.overall_score <= 1
        assert metrics.quality_level in ['excellent', 'good', 'acceptable', 'poor', 'unacceptable']
        assert isinstance(metrics.recommendations, list)

        # Verify detailed metrics
        assert 'perceptual_quality' in metrics.detailed_scores
        assert 'technical_quality' in metrics.detailed_scores
        assert 'emotional_accuracy' in metrics.detailed_scores
        assert 'creativity' in metrics.detailed_scores

        # All scores should be valid
        for score in metrics.detailed_scores.values():
            assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_evaluate_synthesis_with_feedback(self, evaluator, sample_synthesis_result):
        """Test evaluation with user feedback."""
        feedback = {
            'satisfaction': 0.9,
            'visual_quality': 0.8,
            'emotional_accuracy': 0.7,
            'creativity': 0.9,
            'comments': 'Excellent artistic interpretation!'
        }

        metrics = await evaluator.evaluate_synthesis(sample_synthesis_result, user_feedback=feedback)

        assert metrics.user_satisfaction == 0.9
        assert 'Excellent artistic interpretation!' in feedback['comments']

        # User feedback should influence overall score
        assert metrics.overall_score > 0.5  # Should be reasonably high

    @pytest.mark.asyncio
    async def test_evaluate_perceptual_quality(self, evaluator, sample_synthesis_result):
        """Test perceptual quality evaluation."""
        quality_score = await evaluator._evaluate_perceptual_quality(sample_synthesis_result)

        assert 0 <= quality_score <= 1

        # Test with different scenarios
        scenarios = [
            # High quality scenario
            {'features_confidence': 0.9, 'perception_confidence': 0.9, 'expression_confidence': 0.9},
            # Low quality scenario
            {'features_confidence': 0.3, 'perception_confidence': 0.3, 'expression_confidence': 0.3},
            # Mixed quality scenario
            {'features_confidence': 0.8, 'perception_confidence': 0.4, 'expression_confidence': 0.7}
        ]

        for scenario in scenarios:
            with patch.object(sample_synthesis_result.features, 'confidence', scenario['features_confidence']), \
                 patch.object(sample_synthesis_result.perception, 'confidence', scenario['perception_confidence']), \
                 patch.object(sample_synthesis_result.expression, 'confidence', scenario['expression_confidence']):

                score = await evaluator._evaluate_perceptual_quality(sample_synthesis_result)
                assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_evaluate_technical_quality(self, evaluator, sample_synthesis_result):
        """Test technical quality evaluation."""
        tech_score = await evaluator._evaluate_technical_quality(sample_synthesis_result)

        assert 0 <= tech_score <= 1

        # Test with different processing times
        fast_time = 10.0  # Fast processing
        slow_time = 200.0  # Slow processing

        with patch.object(sample_synthesis_result, 'processing_time', fast_time):
            fast_score = await evaluator._evaluate_technical_quality(sample_synthesis_result)

        with patch.object(sample_synthesis_result, 'processing_time', slow_time):
            slow_score = await evaluator._evaluate_technical_quality(sample_synthesis_result)

        # Faster processing should generally score higher
        assert fast_score >= slow_score

    @pytest.mark.asyncio
    async def test_evaluate_emotional_accuracy(self, evaluator, sample_synthesis_result):
        """Test emotional accuracy evaluation."""
        emotional_score = await evaluator._evaluate_emotional_accuracy(sample_synthesis_result)

        assert 0 <= emotional_score <= 1

        # Test with different emotional scenarios
        scenarios = [
            {'valence': 0.8, 'arousal': 0.7, 'intensity': 0.8},  # High emotional match
            {'valence': 0.2, 'arousal': 0.3, 'intensity': 0.2},  # Low emotional match
            {'valence': -0.5, 'arousal': 0.8, 'intensity': 0.6}  # Mixed emotions
        ]

        for scenario in scenarios:
            with patch.object(sample_synthesis_result.perception, 'emotional_valence', scenario['valence']), \
                 patch.object(sample_synthesis_result.perception, 'arousal_level', scenario['arousal']), \
                 patch.object(sample_synthesis_result.expression, 'intensity', scenario['intensity']):

                score = await evaluator._evaluate_emotional_accuracy(sample_synthesis_result)
                assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_evaluate_creativity(self, evaluator, sample_synthesis_result):
        """Test creativity evaluation."""
        creativity_score = await evaluator._evaluate_creativity(sample_synthesis_result)

        assert 0 <= creativity_score <= 1

        # Test with different style combinations
        creative_scenarios = [
            {'style': 'abstract', 'colors': ['#FF0000', '#00FF00', '#0000FF'], 'shapes': ['triangle', 'square']},
            {'style': 'minimalist', 'colors': ['#FFFFFF'], 'shapes': ['circle']},
            {'style': 'chaotic', 'colors': ['#000000', '#808080'], 'shapes': ['random']}
        ]

        for scenario in creative_scenarios:
            with patch.object(sample_synthesis_result.expression, 'style', scenario['style']), \
                 patch.object(sample_synthesis_result.expression, 'colors', scenario['colors']), \
                 patch('random.choice', side_effect=scenario['shapes']):

                score = await evaluator._evaluate_creativity(sample_synthesis_result)
                assert 0 <= creativity_score <= 1

    @pytest.mark.asyncio
    async def test_generate_recommendations(self, evaluator, sample_synthesis_result):
        """Test recommendation generation."""
        metrics = await evaluator.evaluate_synthesis(sample_synthesis_result)

        assert isinstance(metrics.recommendations, list)
        assert len(metrics.recommendations) > 0

        # Recommendations should be strings
        for rec in metrics.recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 10  # Reasonable length

    @pytest.mark.asyncio
    async def test_adapt_parameters(self, evaluator, sample_synthesis_result):
        """Test parameter adaptation."""
        # Generate some evaluation history
        evaluations = []
        for i in range(5):
            # Create varied evaluation results
            metrics = await evaluator.evaluate_synthesis(sample_synthesis_result)
            evaluations.append(metrics)

        adaptation = await evaluator.adapt_parameters(evaluations)

        assert isinstance(adaptation, AdaptationParameters)
        assert hasattr(adaptation, 'learning_rate')
        assert hasattr(adaptation, 'feature_weights')
        assert hasattr(adaptation, 'quality_thresholds')

        # Learning rate should be reasonable
        assert 0 < adaptation.learning_rate <= 1

        # Feature weights should sum to reasonable values
        total_weight = sum(adaptation.feature_weights.values())
        assert 0.8 <= total_weight <= 1.2  # Allow some flexibility

    @pytest.mark.asyncio
    async def test_evaluation_history_management(self, evaluator, sample_synthesis_result):
        """Test evaluation history management."""
        # Add multiple evaluations
        num_evaluations = 10
        for i in range(num_evaluations):
            await evaluator.evaluate_synthesis(sample_synthesis_result)

        # Check history size
        assert len(evaluator.evaluation_history) == num_evaluations

        # Test history retrieval
        history = evaluator.get_evaluation_history()
        assert len(history) == num_evaluations

        # Test limited history
        limited_history = evaluator.get_evaluation_history(limit=5)
        assert len(limited_history) == 5

        # Test history with time filter
        recent_history = evaluator.get_evaluation_history(hours=1)
        assert len(recent_history) <= num_evaluations

    @pytest.mark.asyncio
    async def test_performance_trends(self, evaluator, sample_synthesis_result):
        """Test performance trend analysis."""
        # Create evaluations with increasing scores
        base_score = 0.5
        for i in range(10):
            # Mock different scores
            score_modifier = i * 0.05  # Gradual improvement
            with patch.object(evaluator, '_calculate_overall_score', return_value=min(1.0, base_score + score_modifier)):
                await evaluator.evaluate_synthesis(sample_synthesis_result)

        trends = evaluator.get_performance_trends(days=1)

        assert 'overall_score' in trends
        assert 'perceptual_quality' in trends
        assert 'trend' in trends['overall_score']
        assert 'average' in trends['overall_score']
        assert 'improvement_rate' in trends['overall_score']

        # Should show improvement trend
        assert trends['overall_score']['trend'] in ['improving', 'stable', 'declining']

    @pytest.mark.asyncio
    async def test_quality_level_assignment(self, evaluator, sample_synthesis_result):
        """Test quality level assignment based on scores."""
        test_cases = [
            (0.95, 'excellent'),
            (0.85, 'good'),
            (0.70, 'acceptable'),
            (0.50, 'poor'),
            (0.20, 'unacceptable')
        ]

        for score, expected_level in test_cases:
            with patch.object(evaluator, '_calculate_overall_score', return_value=score):
                metrics = await evaluator.evaluate_synthesis(sample_synthesis_result)
                assert metrics.quality_level == expected_level

    @pytest.mark.asyncio
    async def test_error_handling(self, evaluator):
        """Test error handling in evaluation."""
        # Test with None result
        with pytest.raises(ValueError):
            await evaluator.evaluate_synthesis(None)

        # Test with incomplete result
        incomplete_result = SynthesisResult(
            synthesis_id="incomplete",
            timestamp=datetime.now(),
            audio_input=np.array([]),
            features=None,
            perception=None,
            expression=None,
            rendered_visual=None,
            processing_time=0
        )

        metrics = await evaluator.evaluate_synthesis(incomplete_result)
        assert isinstance(metrics, EvaluationMetrics)
        # Should still produce some evaluation, possibly with low scores

    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self, evaluator, sample_synthesis_result):
        """Test concurrent evaluation processing."""
        # Create multiple evaluation tasks
        num_concurrent = 5
        tasks = [
            evaluator.evaluate_synthesis(sample_synthesis_result)
            for _ in range(num_concurrent)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == num_concurrent
        for metrics in results:
            assert isinstance(metrics, EvaluationMetrics)
            assert 0 <= metrics.overall_score <= 1

    @pytest.mark.asyncio
    async def test_adaptation_strategies(self):
        """Test different adaptation strategies."""
        strategies = ['conservative', 'balanced', 'aggressive']

        for strategy in strategies:
            evaluator = AudioVisualEvaluator()
            config = {'adaptation_strategy': strategy}
            await evaluator.initialize(config)

            try:
                # Create some fake evaluations
                evaluations = [MagicMock() for _ in range(3)]

                adaptation = await evaluator.adapt_parameters(evaluations)

                assert isinstance(adaptation, AdaptationParameters)

                # Different strategies should produce different learning rates
                if strategy == 'conservative':
                    assert adaptation.learning_rate < 0.2
                elif strategy == 'balanced':
                    assert 0.05 <= adaptation.learning_rate <= 0.2
                elif strategy == 'aggressive':
                    assert adaptation.learning_rate > 0.15

            finally:
                await evaluator.cleanup()

    @pytest.mark.asyncio
    async def test_memory_cleanup(self, evaluator, sample_synthesis_result):
        """Test memory cleanup and resource management."""
        # Add many evaluations to fill history
        for i in range(150):  # More than max_history_size
            await evaluator.evaluate_synthesis(sample_synthesis_result)

        # History should be trimmed
        assert len(evaluator.evaluation_history) <= evaluator.max_history_size

        # Should still function normally
        metrics = await evaluator.evaluate_synthesis(sample_synthesis_result)
        assert isinstance(metrics, EvaluationMetrics)
