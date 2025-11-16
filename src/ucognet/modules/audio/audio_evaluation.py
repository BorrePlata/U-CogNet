# U-CogNet Audio-Visual Evaluation and Adaptation Module
# Self-Evaluation and Adaptive Learning System

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json
import os

from .audio_types import SynthesisResult, EvaluationMetrics, AdaptationParameters
from .audio_protocols import AudioVisualEvaluationProtocol

class AudioVisualEvaluator(AudioVisualEvaluationProtocol):
    """Evaluates synthesis quality and adapts system parameters for improvement."""

    def __init__(self):
        # Evaluation criteria weights
        self._evaluation_weights = {
            'perceptual_quality': 0.3,
            'emotional_accuracy': 0.25,
            'artistic_coherence': 0.2,
            'technical_performance': 0.15,
            'user_satisfaction': 0.1
        }

        # Quality thresholds
        self._quality_thresholds = {
            'excellent': 0.9,
            'good': 0.75,
            'acceptable': 0.6,
            'poor': 0.4
        }

        # Adaptation parameters
        self._adaptation_params = AdaptationParameters(
            feature_extraction_params={},
            perception_params={},
            expression_params={},
            rendering_params={},
            learning_rate=0.1,
            adaptation_history=[]
        )

        # Historical data for learning
        self._evaluation_history: List[Dict[str, Any]] = []
        self._max_history_size = 1000

        # Performance baselines
        self._performance_baselines = {
            'processing_time': 100,  # ms
            'memory_usage': 50,      # MB
            'accuracy': 0.8
        }

        # Adaptation strategies
        self._adaptation_strategies = {
            'conservative': {'learning_rate': 0.05, 'change_threshold': 0.1},
            'balanced': {'learning_rate': 0.1, 'change_threshold': 0.15},
            'aggressive': {'learning_rate': 0.2, 'change_threshold': 0.2}
        }

        self._current_strategy = 'balanced'

    async def evaluate_synthesis(self, result: SynthesisResult,
                               ground_truth: Optional[Dict[str, Any]] = None,
                               user_feedback: Optional[Dict[str, Any]] = None) -> EvaluationMetrics:
        """Evaluate the quality of a synthesis result."""
        evaluation_id = f"eval_{int(datetime.now().timestamp() * 1000)}"

        # Calculate individual metrics
        perceptual_quality = await self._evaluate_perceptual_quality(result)
        emotional_accuracy = await self._evaluate_emotional_accuracy(result, ground_truth)
        artistic_coherence = await self._evaluate_artistic_coherence(result)
        technical_performance = await self._evaluate_technical_performance(result)
        user_satisfaction = await self._evaluate_user_satisfaction(user_feedback)

        # Calculate overall score
        overall_score = self._calculate_overall_score({
            'perceptual_quality': perceptual_quality,
            'emotional_accuracy': emotional_accuracy,
            'artistic_coherence': artistic_coherence,
            'technical_performance': technical_performance,
            'user_satisfaction': user_satisfaction
        })

        # Determine quality level
        quality_level = self._determine_quality_level(overall_score)

        # Generate recommendations
        recommendations = await self._generate_recommendations(result, {
            'perceptual_quality': perceptual_quality,
            'emotional_accuracy': emotional_accuracy,
            'artistic_coherence': artistic_coherence,
            'technical_performance': technical_performance,
            'user_satisfaction': user_satisfaction
        })

        # Create evaluation metrics
        metrics = EvaluationMetrics(
            evaluation_id=evaluation_id,
            synthesis_id=result.synthesis_id,
            timestamp=datetime.now(),
            perceptual_quality=perceptual_quality,
            emotional_accuracy=emotional_accuracy,
            artistic_coherence=artistic_coherence,
            technical_performance=technical_performance,
            user_satisfaction=user_satisfaction,
            overall_score=overall_score,
            quality_level=quality_level,
            recommendations=recommendations,
            metadata={
                'evaluation_method': 'comprehensive_analysis',
                'has_ground_truth': ground_truth is not None,
                'has_user_feedback': user_feedback is not None
            }
        )

        # Store evaluation in history
        self._store_evaluation(metrics, result)

        return metrics

    async def _evaluate_perceptual_quality(self, result: SynthesisResult) -> float:
        """Evaluate how well the synthesis captures audio perception."""
        if not result.perception or not result.features:
            return 0.0

        score = 0.0
        criteria_count = 0

        # Check feature extraction quality
        if result.features.confidence > 0:
            score += result.features.confidence
            criteria_count += 1

        # Check perception confidence
        if hasattr(result.perception, 'confidence'):
            score += result.perception.confidence
            criteria_count += 1

        # Check attention weight appropriateness
        attention_weight = result.perception.attention_weight
        if 0 <= attention_weight <= 1:
            # Higher attention should correlate with stronger signals
            rms_energy = result.features.rms_energy
            expected_attention = min(rms_energy * 2, 1.0)
            attention_accuracy = 1.0 - abs(attention_weight - expected_attention)
            score += attention_accuracy
            criteria_count += 1

        # Check memory importance reasonableness
        memory_importance = result.perception.memory_importance
        if 0 <= memory_importance <= 1:
            score += 0.8  # Reasonable range
            criteria_count += 1

        return score / criteria_count if criteria_count > 0 else 0.5

    async def _evaluate_emotional_accuracy(self, result: SynthesisResult,
                                         ground_truth: Optional[Dict[str, Any]]) -> float:
        """Evaluate emotional content accuracy."""
        if not result.perception:
            return 0.0

        if ground_truth and 'emotional_content' in ground_truth:
            # Compare with ground truth
            gt_valence = ground_truth['emotional_content'].get('valence', 0)
            gt_arousal = ground_truth['emotional_content'].get('arousal', 0)

            perceived_valence = result.perception.emotional_valence
            perceived_arousal = result.perception.arousal_level

            valence_accuracy = 1.0 - abs(perceived_valence - gt_valence)
            arousal_accuracy = 1.0 - abs(perceived_arousal - gt_arousal)

            return (valence_accuracy + arousal_accuracy) / 2.0

        else:
            # Evaluate internal consistency
            valence = result.perception.emotional_valence
            arousal = result.perception.arousal_level
            sound_type = result.perception.sound_type

            # Check if emotional response is reasonable for sound type
            expected_emotions = {
                'birdsong': {'valence': (0.3, 0.8), 'arousal': (0.2, 0.6)},
                'explosion': {'valence': (-0.8, -0.2), 'arousal': (0.7, 1.0)},
                'alarm': {'valence': (-0.5, 0.1), 'arousal': (0.6, 0.9)},
                'nature': {'valence': (0.2, 0.7), 'arousal': (0.1, 0.5)},
                'urban': {'valence': (-0.3, 0.3), 'arousal': (0.3, 0.7)}
            }

            if sound_type in expected_emotions:
                expected = expected_emotions[sound_type]
                valence_in_range = expected['valence'][0] <= valence <= expected['valence'][1]
                arousal_in_range = expected['arousal'][0] <= arousal <= expected['arousal'][1]

                valence_score = 1.0 if valence_in_range else 0.5
                arousal_score = 1.0 if arousal_in_range else 0.5

                return (valence_score + arousal_score) / 2.0
            else:
                return 0.7  # Neutral score for unknown types

    async def _evaluate_artistic_coherence(self, result: SynthesisResult) -> float:
        """Evaluate artistic coherence of visual expression."""
        if not result.expression:
            return 0.0

        score = 0.0
        criteria_count = 0

        # Check color harmony
        harmony = result.expression.composition.get('harmony', 0.5)
        score += harmony
        criteria_count += 1

        # Check balance
        balance = result.expression.composition.get('balance', 0.5)
        score += balance
        criteria_count += 1

        # Check intensity appropriateness
        intensity = result.expression.intensity
        arousal = result.perception.arousal_level if result.perception else 0.5
        intensity_match = 1.0 - abs(intensity - arousal)
        score += intensity_match
        criteria_count += 1

        # Check symbol relevance
        symbols = result.expression.symbols
        sound_type = result.perception.sound_type if result.perception else 'unknown'
        symbol_relevance = self._evaluate_symbol_relevance(symbols, sound_type)
        score += symbol_relevance
        criteria_count += 1

        return score / criteria_count if criteria_count > 0 else 0.5

    async def _evaluate_technical_performance(self, result: SynthesisResult) -> float:
        """Evaluate technical performance."""
        score = 0.0
        criteria_count = 0

        # Check processing time
        processing_time = result.processing_time
        max_time = self._performance_baselines['processing_time']
        time_score = max(0, 1.0 - (processing_time / max_time))
        score += time_score
        criteria_count += 1

        # Check for errors
        has_error = 'error' in result.metadata
        error_score = 0.0 if has_error else 1.0
        score += error_score
        criteria_count += 1

        # Check data completeness
        completeness = self._evaluate_data_completeness(result)
        score += completeness
        criteria_count += 1

        return score / criteria_count if criteria_count > 0 else 0.5

    async def _evaluate_user_satisfaction(self, user_feedback: Optional[Dict[str, Any]]) -> float:
        """Evaluate user satisfaction from feedback."""
        if not user_feedback:
            return 0.5  # Neutral score when no feedback

        satisfaction_score = 0.0
        criteria_count = 0

        # Direct satisfaction rating
        if 'satisfaction' in user_feedback:
            satisfaction_score += user_feedback['satisfaction']
            criteria_count += 1

        # Quality ratings
        quality_ratings = ['visual_quality', 'emotional_accuracy', 'creativity', 'appropriateness']
        for rating in quality_ratings:
            if rating in user_feedback:
                satisfaction_score += user_feedback[rating]
                criteria_count += 1

        # Comments sentiment (simplified)
        if 'comments' in user_feedback:
            comments = user_feedback['comments'].lower()
            positive_words = ['good', 'great', 'excellent', 'amazing', 'beautiful']
            negative_words = ['bad', 'poor', 'terrible', 'awful', 'ugly']

            positive_count = sum(1 for word in positive_words if word in comments)
            negative_count = sum(1 for word in negative_words if word in comments)

            if positive_count > negative_count:
                sentiment_score = 0.8
            elif negative_count > positive_count:
                sentiment_score = 0.2
            else:
                sentiment_score = 0.5

            satisfaction_score += sentiment_score
            criteria_count += 1

        return satisfaction_score / criteria_count if criteria_count > 0 else 0.5

    def _calculate_overall_score(self, component_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score."""
        overall_score = 0.0

        for component, score in component_scores.items():
            weight = self._evaluation_weights.get(component, 0.2)
            overall_score += score * weight

        return overall_score

    def _determine_quality_level(self, overall_score: float) -> str:
        """Determine quality level from score."""
        if overall_score >= self._quality_thresholds['excellent']:
            return 'excellent'
        elif overall_score >= self._quality_thresholds['good']:
            return 'good'
        elif overall_score >= self._quality_thresholds['acceptable']:
            return 'acceptable'
        elif overall_score >= self._quality_thresholds['poor']:
            return 'poor'
        else:
            return 'unacceptable'

    async def _generate_recommendations(self, result: SynthesisResult,
                                      component_scores: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        # Analyze each component
        if component_scores['perceptual_quality'] < 0.7:
            recommendations.append("Improve audio feature extraction quality")
            recommendations.append("Enhance perception confidence thresholds")

        if component_scores['emotional_accuracy'] < 0.7:
            recommendations.append("Refine emotional mapping algorithms")
            recommendations.append("Incorporate more diverse emotional training data")

        if component_scores['artistic_coherence'] < 0.7:
            recommendations.append("Adjust color palette selection for better harmony")
            recommendations.append("Improve symbol selection relevance")

        if component_scores['technical_performance'] < 0.7:
            recommendations.append("Optimize processing pipeline for better performance")
            recommendations.append("Implement better error handling and recovery")

        if component_scores['user_satisfaction'] < 0.7:
            recommendations.append("Gather more user feedback for iterative improvement")
            recommendations.append("Consider user preferences in synthesis parameters")

        # General recommendations
        if result.processing_time > self._performance_baselines['processing_time']:
            recommendations.append("Consider performance optimizations or quality trade-offs")

        if len(recommendations) == 0:
            recommendations.append("System performing well - continue monitoring")

        return recommendations

    def _evaluate_symbol_relevance(self, symbols: List[Any], sound_type: str) -> float:
        """Evaluate relevance of symbols to sound type."""
        if not symbols:
            return 0.5

        # Simple relevance check based on symbol content
        relevant_symbols = 0
        type_keywords = {
            'birdsong': ['bird', 'nature', 'leaf', 'flower'],
            'explosion': ['boom', 'fire', 'burst', 'energy'],
            'alarm': ['alert', 'warning', 'signal', 'urgent'],
            'nature': ['tree', 'water', 'wind', 'earth'],
            'urban': ['city', 'car', 'building', 'street']
        }

        keywords = type_keywords.get(sound_type, [])

        for symbol in symbols:
            symbol_text = str(symbol.symbol).lower()
            if any(keyword in symbol_text for keyword in keywords):
                relevant_symbols += 1

        return relevant_symbols / len(symbols) if symbols else 0.5

    def _evaluate_data_completeness(self, result: SynthesisResult) -> float:
        """Evaluate completeness of synthesis data."""
        completeness = 0.0
        total_checks = 4

        if result.features is not None:
            completeness += 1
        if result.perception is not None:
            completeness += 1
        if result.expression is not None:
            completeness += 1
        if result.rendered_visual is not None:
            completeness += 1

        return completeness / total_checks

    def _store_evaluation(self, metrics: EvaluationMetrics, result: SynthesisResult) -> None:
        """Store evaluation in history."""
        evaluation_record = {
            'metrics': metrics.__dict__,
            'result_summary': {
                'synthesis_id': result.synthesis_id,
                'processing_time': result.processing_time,
                'has_error': 'error' in result.metadata
            },
            'timestamp': datetime.now().isoformat()
        }

        self._evaluation_history.append(evaluation_record)

        # Maintain history size
        if len(self._evaluation_history) > self._max_history_size:
            self._evaluation_history.pop(0)

    async def adapt_parameters(self, evaluation_metrics: List[EvaluationMetrics]) -> AdaptationParameters:
        """Adapt system parameters based on evaluation history."""
        if not evaluation_metrics:
            return self._adaptation_params

        # Analyze trends
        recent_evaluations = evaluation_metrics[-50:]  # Last 50 evaluations

        # Calculate average scores
        avg_scores = {}
        for component in self._evaluation_weights.keys():
            scores = [getattr(m, component) for m in recent_evaluations if hasattr(m, component)]
            if scores:
                avg_scores[component] = np.mean(scores)

        # Identify areas needing improvement
        improvement_areas = []
        for component, avg_score in avg_scores.items():
            if avg_score < 0.7:  # Below acceptable threshold
                improvement_areas.append(component)

        # Generate adaptation parameters
        adaptation = self._generate_adaptation_parameters(improvement_areas, avg_scores)

        # Update internal parameters
        self._adaptation_params = adaptation

        # Store adaptation in history
        self._adaptation_params.adaptation_history.append({
            'timestamp': datetime.now().isoformat(),
            'improvement_areas': improvement_areas,
            'avg_scores': avg_scores,
            'new_parameters': adaptation.__dict__
        })

        return adaptation

    def _generate_adaptation_parameters(self, improvement_areas: List[str],
                                      avg_scores: Dict[str, float]) -> AdaptationParameters:
        """Generate specific adaptation parameters."""
        strategy = self._adaptation_strategies[self._current_strategy]
        learning_rate = strategy['learning_rate']

        # Base parameters
        params = AdaptationParameters(
            feature_extraction_params={},
            perception_params={},
            expression_params={},
            rendering_params={},
            learning_rate=learning_rate,
            adaptation_history=[]
        )

        # Adapt feature extraction
        if 'perceptual_quality' in improvement_areas:
            params.feature_extraction_params = {
                'quality_boost': min(0.2, (0.8 - avg_scores.get('perceptual_quality', 0.5)) * learning_rate),
                'confidence_threshold_adjustment': learning_rate * 0.1
            }

        # Adapt perception
        if 'emotional_accuracy' in improvement_areas:
            params.perception_params = {
                'emotion_sensitivity': learning_rate * 0.15,
                'context_weight_increase': learning_rate * 0.1
            }

        # Adapt expression
        if 'artistic_coherence' in improvement_areas:
            params.expression_params = {
                'color_harmony_weight': learning_rate * 0.2,
                'symbol_relevance_boost': learning_rate * 0.15
            }

        # Adapt rendering
        if 'technical_performance' in improvement_areas:
            params.rendering_params = {
                'quality_optimization': learning_rate * 0.1,
                'performance_mode': 'balanced' if avg_scores.get('technical_performance', 0.5) < 0.6 else 'quality'
            }

        return params

    def get_evaluation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get evaluation history."""
        history = self._evaluation_history
        if limit:
            history = history[-limit:]
        return history.copy()

    def get_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get performance trends over time."""
        cutoff_date = datetime.now() - timedelta(days=days)

        recent_evaluations = [
            eval for eval in self._evaluation_history
            if datetime.fromisoformat(eval['timestamp']) > cutoff_date
        ]

        if not recent_evaluations:
            return {'error': 'No recent evaluations found'}

        # Calculate trends
        trends = {}
        components = list(self._evaluation_weights.keys()) + ['overall_score']

        for component in components:
            scores = []
            timestamps = []

            for eval_record in recent_evaluations:
                metrics = eval_record['metrics']
                if component in metrics:
                    scores.append(metrics[component])
                    timestamps.append(datetime.fromisoformat(eval_record['timestamp']))

            if scores:
                trends[component] = {
                    'average': np.mean(scores),
                    'trend': 'improving' if len(scores) > 1 and scores[-1] > scores[0] else 'declining',
                    'volatility': np.std(scores) if len(scores) > 1 else 0
                }

        return trends

    def export_evaluation_data(self, filepath: str) -> None:
        """Export evaluation data for analysis."""
        data = {
            'evaluation_history': self._evaluation_history,
            'adaptation_parameters': self._adaptation_params.__dict__,
            'performance_baselines': self._performance_baselines,
            'export_timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logging.info(f"Evaluation data exported to {filepath}")

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the evaluation module."""
        # Update configuration
        if 'evaluation_weights' in config:
            self._evaluation_weights.update(config['evaluation_weights'])

        if 'quality_thresholds' in config:
            self._quality_thresholds.update(config['quality_thresholds'])

        if 'adaptation_strategy' in config:
            self._current_strategy = config['adaptation_strategy']

        if 'max_history_size' in config:
            self._max_history_size = config['max_history_size']

        # Load existing evaluation data if available
        if 'evaluation_data_path' in config:
            await self._load_evaluation_data(config['evaluation_data_path'])

        logging.info("AudioVisualEvaluator initialized")

    async def _load_evaluation_data(self, filepath: str) -> None:
        """Load existing evaluation data."""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                if 'evaluation_history' in data:
                    self._evaluation_history = data['evaluation_history']

                if 'adaptation_parameters' in data:
                    # Reconstruct adaptation parameters
                    params_data = data['adaptation_parameters']
                    self._adaptation_params = AdaptationParameters(**params_data)

                logging.info(f"Loaded evaluation data from {filepath}")
            except Exception as e:
                logging.warning(f"Failed to load evaluation data: {e}")

    async def cleanup(self) -> None:
        """Clean up resources."""
        # Export final evaluation data
        if self._evaluation_history:
            export_path = f"evaluation_data_{int(datetime.now().timestamp())}.json"
            self.export_evaluation_data(export_path)

        logging.info("AudioVisualEvaluator cleaned up")
