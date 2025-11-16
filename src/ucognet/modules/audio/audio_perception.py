# U-CogNet Audio Perception Module
# Cognitive Interpretation of Audio Features

import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import logging

from .audio_types import AudioFeatures, AudioPerception
from .audio_protocols import AudioPerceptionProtocol

class CognitiveAudioPerception(AudioPerceptionProtocol):
    """Cognitive interpretation of audio features with emotional and contextual understanding."""

    def __init__(self):
        # Sound type classification thresholds
        self._sound_type_thresholds = {
            'birdsong': {
                'spectral_centroid_min': 3000,
                'zero_crossing_rate_min': 0.1,
                'harmonic_ratio_min': 0.6,
                'tempo_range': (100, 200)
            },
            'explosion': {
                'rms_energy_min': 0.3,
                'spectral_centroid_max': 2000,
                'percussive_ratio_min': 0.7,
                'onset_strength_min': 0.8
            },
            'alarm': {
                'spectral_centroid_range': (2000, 4000),
                'zero_crossing_rate_min': 0.15,
                'harmonic_ratio_max': 0.4,
                'tempo_range': (120, 180)
            },
            'nature': {
                'harmonic_ratio_min': 0.5,
                'rms_energy_max': 0.2,
                'spectral_centroid_range': (500, 2500),
                'zero_crossing_rate_max': 0.1
            },
            'urban': {
                'harmonic_ratio_max': 0.3,
                'rms_energy_range': (0.1, 0.4),
                'spectral_centroid_range': (1000, 3000),
                'zero_crossing_rate_min': 0.05
            }
        }

        # Emotional mapping
        self._emotion_mappings = {
            'valence': {  # Positive-negative dimension
                'high_harmonic': 0.8,    # Harmonious sounds are positive
                'low_centroid': -0.3,    # Low frequency can be somber
                'high_energy': 0.2,      # High energy can be exciting
                'high_tempo': 0.4        # Fast tempo is lively
            },
            'arousal': {  # Calm-excited dimension
                'high_energy': 0.9,      # High energy increases arousal
                'high_tempo': 0.7,       # Fast tempo increases arousal
                'high_zcr': 0.6,         # Complex sounds increase arousal
                'percussive': 0.8        # Percussive sounds are arousing
            }
        }

    async def perceive_audio(self, features: AudioFeatures) -> AudioPerception:
        """Transform audio features into cognitive perception."""
        # Classify sound type
        sound_type = await self.classify_sound_type(features)

        # Assess emotional content
        emotional_content = await self.assess_emotional_content(features)

        # Contextualize environment
        environment_context = await self.contextualize_environment(features)

        # Analyze temporal patterns
        temporal_pattern = self._analyze_temporal_pattern(features)

        # Estimate spatial characteristics
        spatial_characteristics = self._estimate_spatial_characteristics(features)

        # Calculate attention weight
        attention_weight = await self.evaluate_attention_worthiness(
            AudioPerception(
                timestamp=features.timestamp,
                features=features,
                sound_type=sound_type,
                emotional_valence=emotional_content['valence'],
                arousal_level=emotional_content['arousal'],
                familiarity=0.5,  # Placeholder - would need learning
                environment_context=environment_context,
                temporal_pattern=temporal_pattern,
                spatial_characteristics=spatial_characteristics,
                attention_weight=0.0,  # Will be calculated
                memory_importance=0.0   # Will be calculated
            )
        )

        # Calculate memory importance
        memory_importance = self._calculate_memory_importance(
            sound_type, emotional_content, attention_weight
        )

        perception = AudioPerception(
            timestamp=features.timestamp,
            features=features,
            sound_type=sound_type,
            emotional_valence=emotional_content['valence'],
            arousal_level=emotional_content['arousal'],
            familiarity=self._estimate_familiarity(features),
            environment_context=environment_context,
            temporal_pattern=temporal_pattern,
            spatial_characteristics=spatial_characteristics,
            attention_weight=attention_weight,
            memory_importance=memory_importance,
            confidence=self._calculate_confidence(features),
            metadata={
                'perception_method': 'cognitive_analysis',
                'feature_quality': self._assess_feature_quality(features)
            }
        )

        return perception

    async def classify_sound_type(self, features: AudioFeatures) -> str:
        """Classify the type of sound using feature analysis."""
        scores = {}

        for sound_type, thresholds in self._sound_type_thresholds.items():
            score = 0
            total_criteria = len(thresholds)

            # Check spectral centroid
            if 'spectral_centroid_min' in thresholds:
                if features.spectral_centroid >= thresholds['spectral_centroid_min']:
                    score += 1
            if 'spectral_centroid_max' in thresholds:
                if features.spectral_centroid <= thresholds['spectral_centroid_max']:
                    score += 1
            if 'spectral_centroid_range' in thresholds:
                min_val, max_val = thresholds['spectral_centroid_range']
                if min_val <= features.spectral_centroid <= max_val:
                    score += 1

            # Check zero crossing rate
            if 'zero_crossing_rate_min' in thresholds:
                if features.zero_crossing_rate >= thresholds['zero_crossing_rate_min']:
                    score += 1
            if 'zero_crossing_rate_max' in thresholds:
                if features.zero_crossing_rate <= thresholds['zero_crossing_rate_max']:
                    score += 1

            # Check harmonic/percussive ratios
            if 'harmonic_ratio_min' in thresholds:
                if features.harmonic_ratio >= thresholds['harmonic_ratio_min']:
                    score += 1
            if 'harmonic_ratio_max' in thresholds:
                if features.harmonic_ratio <= thresholds['harmonic_ratio_max']:
                    score += 1
            if 'percussive_ratio_min' in thresholds:
                if features.percussive_ratio >= thresholds['percussive_ratio_min']:
                    score += 1

            # Check RMS energy
            if 'rms_energy_min' in thresholds:
                if features.rms_energy >= thresholds['rms_energy_min']:
                    score += 1
            if 'rms_energy_max' in thresholds:
                if features.rms_energy <= thresholds['rms_energy_max']:
                    score += 1
            if 'rms_energy_range' in thresholds:
                min_val, max_val = thresholds['rms_energy_range']
                if min_val <= features.rms_energy <= max_val:
                    score += 1

            # Check onset strength
            if 'onset_strength_min' in thresholds:
                if features.onset_strength >= thresholds['onset_strength_min']:
                    score += 1

            # Check tempo
            if 'tempo_range' in thresholds:
                min_tempo, max_tempo = thresholds['tempo_range']
                if min_tempo <= features.tempo <= max_tempo:
                    score += 1

            scores[sound_type] = score / total_criteria

        # Return the sound type with highest score
        best_type = max(scores.items(), key=lambda x: x[1])
        return best_type[0] if best_type[1] > 0.3 else 'unknown'

    async def assess_emotional_content(self, features: AudioFeatures) -> Dict[str, float]:
        """Assess emotional valence and arousal from audio features."""
        valence_score = 0.0
        arousal_score = 0.0

        # Valence calculation
        valence_contributions = self._emotion_mappings['valence']
        valence_score += valence_contributions['high_harmonic'] * features.harmonic_ratio
        valence_score += valence_contributions['low_centroid'] * (1 - min(features.spectral_centroid / 5000, 1))
        valence_score += valence_contributions['high_energy'] * features.rms_energy
        valence_score += valence_contributions['high_tempo'] * min(features.tempo / 200, 1)

        # Normalize valence to [-1, 1]
        valence_score = max(-1.0, min(1.0, valence_score))

        # Arousal calculation
        arousal_contributions = self._emotion_mappings['arousal']
        arousal_score += arousal_contributions['high_energy'] * features.rms_energy
        arousal_score += arousal_contributions['high_tempo'] * min(features.tempo / 180, 1)
        arousal_score += arousal_contributions['high_zcr'] * features.zero_crossing_rate
        arousal_score += arousal_contributions['percussive'] * features.percussive_ratio

        # Normalize arousal to [0, 1]
        arousal_score = max(0.0, min(1.0, arousal_score))

        return {
            'valence': valence_score,
            'arousal': arousal_score,
            'emotional_intensity': np.sqrt(valence_score**2 + arousal_score**2)
        }

    async def contextualize_environment(self, features: AudioFeatures) -> str:
        """Determine environmental context from audio characteristics."""
        # Analyze spectral and temporal characteristics
        spectral_centroid = features.spectral_centroid
        harmonic_ratio = features.harmonic_ratio
        rms_energy = features.rms_energy
        zero_crossing_rate = features.zero_crossing_rate

        # Forest/Nature: High harmonic content, moderate spectral centroid
        if harmonic_ratio > 0.6 and 1000 < spectral_centroid < 3000:
            return 'forest'

        # Urban: Lower harmonic content, higher zero crossing rate
        elif harmonic_ratio < 0.4 and zero_crossing_rate > 0.08:
            return 'urban'

        # Industrial: Low harmonic content, high energy, low-mid frequencies
        elif harmonic_ratio < 0.3 and rms_energy > 0.25 and spectral_centroid < 1500:
            return 'industrial'

        # Domestic: Moderate everything, balanced characteristics
        elif 0.4 <= harmonic_ratio <= 0.7 and 0.05 <= zero_crossing_rate <= 0.15:
            return 'domestic'

        # Unknown/Default
        else:
            return 'unknown'

    def _analyze_temporal_pattern(self, features: AudioFeatures) -> str:
        """Analyze temporal patterns in the audio."""
        tempo = features.tempo
        onset_strength = features.onset_strength
        rms_energy = features.rms_energy

        # Analyze beat positions for rhythm
        if features.beat_positions:
            beat_intervals = np.diff(features.beat_positions)
            regularity = 1.0 / (1.0 + np.std(beat_intervals)) if len(beat_intervals) > 1 else 0.5
        else:
            regularity = 0.5

        # Classify temporal pattern
        if onset_strength > 0.7 and rms_energy > 0.3:
            return 'pulsed'  # Strong, distinct events
        elif regularity > 0.7 and tempo > 100:
            return 'rhythmic'  # Regular beat pattern
        elif onset_strength < 0.3 and rms_energy < 0.2:
            return 'continuous'  # Steady, ongoing sound
        elif np.random.random() < 0.3:  # Some randomness in classification
            return 'intermittent'
        else:
            return 'random'

    def _estimate_spatial_characteristics(self, features: AudioFeatures) -> Dict[str, float]:
        """Estimate spatial characteristics (simplified)."""
        # In a real implementation, this would use binaural cues, reverberation, etc.
        # For now, provide reasonable estimates based on frequency content

        spectral_centroid = features.spectral_centroid
        rms_energy = features.rms_energy

        # Distance estimation (higher frequencies attenuate faster)
        if spectral_centroid > 3000:
            distance = 0.3  # Close for high frequencies
        elif spectral_centroid > 1000:
            distance = 0.6  # Medium distance
        else:
            distance = 0.8  # Far for low frequencies

        # Direction estimation (simplified - would need stereo)
        direction = 0.5  # Center (mono audio)

        # Room size estimation based on reverberation cues (simplified)
        room_size = 0.5  # Medium room

        return {
            'distance': distance,
            'direction': direction,
            'room_size': room_size,
            'reverberation': 0.3  # Low reverberation estimate
        }

    async def evaluate_attention_worthiness(self, perception: AudioPerception) -> float:
        """Determine how much cognitive attention this audio deserves."""
        attention_score = 0.0

        # Novelty factor
        novelty = 1.0 - perception.familiarity
        attention_score += novelty * 0.4

        # Emotional intensity
        emotional_intensity = abs(perception.emotional_valence) + perception.arousal_level
        attention_score += emotional_intensity * 0.3

        # Sound type importance
        type_weights = {
            'explosion': 1.0,    # High priority
            'alarm': 0.9,        # High priority
            'birdsong': 0.6,     # Medium priority
            'nature': 0.4,       # Low-medium priority
            'urban': 0.3,        # Low priority
            'unknown': 0.7       # Investigate unknowns
        }
        type_weight = type_weights.get(perception.sound_type, 0.5)
        attention_score += type_weight * 0.3

        return min(1.0, attention_score)

    def _calculate_memory_importance(self, sound_type: str,
                                   emotional_content: Dict[str, float],
                                   attention_weight: float) -> float:
        """Calculate how important this is to store in memory."""
        importance = attention_weight * 0.5

        # Emotional significance
        emotional_significance = abs(emotional_content['valence']) + emotional_content['arousal']
        importance += emotional_significance * 0.3

        # Type significance
        type_significance = {
            'explosion': 0.9, 'alarm': 0.8, 'birdsong': 0.6,
            'nature': 0.4, 'urban': 0.2, 'unknown': 0.7
        }.get(sound_type, 0.5)
        importance += type_significance * 0.2

        return min(1.0, importance)

    def _estimate_familiarity(self, features: AudioFeatures) -> float:
        """Estimate how familiar this sound is (placeholder for learning system)."""
        # In a real system, this would compare against learned sound patterns
        # For now, return moderate familiarity
        return 0.5

    def _calculate_confidence(self, features: AudioFeatures) -> float:
        """Calculate confidence in the perception."""
        # Base confidence on feature quality and consistency
        feature_quality = self._assess_feature_quality(features)

        # Higher confidence for clear, strong signals
        signal_strength = min(features.rms_energy * 3, 1.0)

        confidence = (feature_quality + signal_strength) / 2.0
        return confidence

    def _assess_feature_quality(self, features: AudioFeatures) -> float:
        """Assess the quality of extracted features."""
        # Check for reasonable value ranges
        quality_score = 0.0
        checks = 0

        # Spectral centroid in reasonable range
        if 50 < features.spectral_centroid < 8000:
            quality_score += 1
        checks += 1

        # RMS energy not too low
        if features.rms_energy > 0.01:
            quality_score += 1
        checks += 1

        # Harmonic/percussive ratios reasonable
        if 0 <= features.harmonic_ratio <= 1 and 0 <= features.percussive_ratio <= 1:
            quality_score += 1
        checks += 1

        # MFCC dimensions reasonable
        if features.mfcc.shape[0] >= 10 and features.mfcc.shape[1] > 0:
            quality_score += 1
        checks += 1

        return quality_score / checks if checks > 0 else 0.5

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the perception module."""
        # Update thresholds if provided
        if 'sound_type_thresholds' in config:
            self._sound_type_thresholds.update(config['sound_type_thresholds'])

        if 'emotion_mappings' in config:
            self._emotion_mappings.update(config['emotion_mappings'])

        logging.info("Initialized CognitiveAudioPerception")

    async def cleanup(self) -> None:
        """Clean up resources."""
    pass
