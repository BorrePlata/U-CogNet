#!/usr/bin/env python3
"""
Advanced Multimodal Attention System for U-CogNet
Implements gating attention, temporal integration, and hierarchical fusion mechanisms.
Postdoctoral-level cognitive architecture for multimodal reinforcement learning.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class Modality(Enum):
    """Modalities available in the cognitive system."""
    VISUAL = "visual"
    AUDIO = "audio"
    TEXT = "text"
    TACTILE = "tactile"

class AttentionGate(Enum):
    """Attention gating states."""
    OPEN = "open"
    CLOSED = "closed"
    FILTERING = "filtering"

@dataclass
class ModalitySignal:
    """Represents a signal from a specific modality."""
    modality: Modality
    data: Any
    timestamp: float
    confidence: float
    priority: float
    context: Dict[str, Any]

@dataclass
class AttentionState:
    """Current state of the attention system."""
    active_modalities: Dict[Modality, AttentionGate]
    modality_weights: Dict[Modality, float]
    temporal_context: List[ModalitySignal]
    performance_history: List[float]
    adaptation_rate: float

class TemporalIntegrator:
    """Handles temporal integration of multimodal signals."""

    def __init__(self, buffer_size: int = 100, decay_rate: float = 0.95):
        self.buffer_size = buffer_size
        self.decay_rate = decay_rate
        self.signal_buffer: List[ModalitySignal] = []
        self.temporal_weights = np.ones(buffer_size)

    def add_signal(self, signal: ModalitySignal) -> None:
        """Add a new signal to the temporal buffer."""
        self.signal_buffer.append(signal)

        # Maintain buffer size
        if len(self.signal_buffer) > self.buffer_size:
            self.signal_buffer.pop(0)

        # Update temporal weights (more recent signals have higher weight)
        self._update_temporal_weights()

    def _update_temporal_weights(self) -> None:
        """Update temporal decay weights."""
        n_signals = len(self.signal_buffer)
        for i in range(n_signals):
            # Exponential decay: newer signals have higher weights
            age = n_signals - i - 1
            self.temporal_weights[i] = self.decay_rate ** age

    def get_integrated_context(self, modality: Optional[Modality] = None) -> Dict[str, Any]:
        """Get temporally integrated context for a modality or all modalities."""
        if not self.signal_buffer:
            return {}

        if modality:
            # Filter signals by modality
            relevant_signals = [s for s in self.signal_buffer if s.modality == modality]
        else:
            relevant_signals = self.signal_buffer

        if not relevant_signals:
            return {}

        # Calculate weighted average of confidence and priority
        weights = self.temporal_weights[-len(relevant_signals):]
        confidences = np.array([s.confidence for s in relevant_signals])
        priorities = np.array([s.priority for s in relevant_signals])

        integrated_confidence = np.average(confidences, weights=weights)
        integrated_priority = np.average(priorities, weights=weights)

        # Get most recent context
        latest_signal = relevant_signals[-1]

        return {
            'integrated_confidence': integrated_confidence,
            'integrated_priority': integrated_priority,
            'temporal_span': len(relevant_signals),
            'latest_context': latest_signal.context,
            'dominant_modality': latest_signal.modality.value
        }

class HierarchicalFusion:
    """Sophisticated multimodal fusion with hierarchical importance."""

    def __init__(self):
        self.modality_hierarchy = {
            Modality.VISUAL: 0.8,  # High priority for visual
            Modality.AUDIO: 0.6,   # Medium priority for audio
            Modality.TEXT: 0.4,    # Lower priority for text
            Modality.TACTILE: 0.2  # Lowest priority for tactile
        }

        self.conflict_resolution_matrix = {
            (Modality.VISUAL, Modality.AUDIO): 0.7,  # Visual dominates audio
            (Modality.AUDIO, Modality.TEXT): 0.8,    # Audio dominates text
            (Modality.VISUAL, Modality.TEXT): 0.9,   # Visual strongly dominates text
        }

    def fuse_signals(self, signals: List[ModalitySignal]) -> ModalitySignal:
        """Fuse multiple modality signals using hierarchical fusion."""
        if not signals:
            return None

        if len(signals) == 1:
            return signals[0]

        # Sort by hierarchical priority
        sorted_signals = sorted(signals,
                              key=lambda s: self.modality_hierarchy.get(s.modality, 0),
                              reverse=True)

        # Start with highest priority signal
        fused_signal = sorted_signals[0]

        # Iteratively fuse lower priority signals
        for signal in sorted_signals[1:]:
            fused_signal = self._fuse_pair(fused_signal, signal)

        return fused_signal

    def _fuse_pair(self, signal1: ModalitySignal, signal2: ModalitySignal) -> ModalitySignal:
        """Fuse two signals considering their hierarchical relationship."""
        # Determine dominance based on hierarchy
        hierarchy_diff = (self.modality_hierarchy.get(signal1.modality, 0) -
                         self.modality_hierarchy.get(signal2.modality, 0))

        if hierarchy_diff > 0:
            # signal1 dominates
            dominance_ratio = min(0.9, 0.5 + hierarchy_diff)
        else:
            # signal2 dominates or equal
            dominance_ratio = max(0.1, 0.5 + hierarchy_diff)

        # Check for specific conflict resolution
        conflict_key = (signal1.modality, signal2.modality)
        if conflict_key in self.conflict_resolution_matrix:
            dominance_ratio = self.conflict_resolution_matrix[conflict_key]

        # Fuse the signals
        fused_confidence = (signal1.confidence * dominance_ratio +
                           signal2.confidence * (1 - dominance_ratio))

        fused_priority = (signal1.priority * dominance_ratio +
                         signal2.priority * (1 - dominance_ratio))

        # Merge contexts
        fused_context = {**signal1.context, **signal2.context}
        fused_context['fusion_method'] = 'hierarchical'
        fused_context['dominance_ratio'] = dominance_ratio

        return ModalitySignal(
            modality=signal1.modality,  # Keep dominant modality
            data=self._merge_data(signal1.data, signal2.data, dominance_ratio),
            timestamp=max(signal1.timestamp, signal2.timestamp),
            confidence=fused_confidence,
            priority=fused_priority,
            context=fused_context
        )

    def _merge_data(self, data1: Any, data2: Any, dominance_ratio: float) -> Any:
        """Merge data from two signals based on dominance."""
        if isinstance(data1, dict) and isinstance(data2, dict):
            merged = {}
            for key in set(data1.keys()) | set(data2.keys()):
                if key in data1 and key in data2:
                    # Weighted average for overlapping keys
                    if isinstance(data1[key], (int, float)) and isinstance(data2[key], (int, float)):
                        merged[key] = data1[key] * dominance_ratio + data2[key] * (1 - dominance_ratio)
                    else:
                        merged[key] = data1[key] if dominance_ratio > 0.5 else data2[key]
                elif key in data1:
                    merged[key] = data1[key]
                else:
                    merged[key] = data2[key]
            return merged
        elif isinstance(data1, (int, float)) and isinstance(data2, (int, float)):
            return data1 * dominance_ratio + data2 * (1 - dominance_ratio)
        else:
            return data1 if dominance_ratio > 0.5 else data2

class GatingAttentionController:
    """Advanced gating attention system for modality control."""

    def __init__(self, adaptation_rate: float = 0.3, open_threshold: float = 0.5,
                 close_threshold: float = 0.2, filter_threshold: float = 0.35,
                 performance_window: int = 100):
        self.adaptation_rate = adaptation_rate
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        self.filter_threshold = filter_threshold
        self.performance_window = performance_window

        self.attention_state = AttentionState(
            active_modalities={modality: AttentionGate.CLOSED for modality in Modality},
            modality_weights={modality: 0.5 for modality in Modality},
            temporal_context=[],
            performance_history=[],
            adaptation_rate=adaptation_rate
        )

        self.temporal_integrator = TemporalIntegrator()
        self.hierarchical_fusion = HierarchicalFusion()

        # Performance tracking - increased window
        self.performance_window = 100  # Increased from 50
        self.gating_thresholds = {
            'open_gate': 0.5,      # Reduced from 0.7
            'close_gate': 0.2,     # Reduced from 0.3
            'filter_threshold': 0.35  # Reduced from 0.5
        }

    def process_multimodal_input(self, signals: List[ModalitySignal],
                               current_performance: float) -> Tuple[ModalitySignal, AttentionState]:
        """Process multimodal input through gating attention system."""

        # Update performance history
        self.attention_state.performance_history.append(current_performance)
        if len(self.attention_state.performance_history) > self.performance_window:
            self.attention_state.performance_history.pop(0)

        # Add signals to temporal integrator
        for signal in signals:
            self.temporal_integrator.add_signal(signal)

        # Update attention gates based on performance and signal quality
        self._update_attention_gates(signals)

        # Get active signals (those with open gates)
        active_signals = [s for s in signals
                         if self.attention_state.active_modalities.get(s.modality) == AttentionGate.OPEN]

        # Apply hierarchical fusion to active signals
        if active_signals:
            fused_signal = self.hierarchical_fusion.fuse_signals(active_signals)
        else:
            # If no gates are open, use the highest priority signal with filtering
            filtered_signals = [s for s in signals
                              if self.attention_state.active_modalities.get(s.modality) == AttentionGate.FILTERING]
            if filtered_signals:
                fused_signal = self.hierarchical_fusion.fuse_signals(filtered_signals)
            else:
                fused_signal = None

        # Adapt weights based on performance
        self._adapt_weights(current_performance)

        return fused_signal, self.attention_state

    def _update_attention_gates(self, signals: List[ModalitySignal]) -> None:
        """Update attention gates based on signal quality and temporal context."""

        for modality in Modality:
            modality_signals = [s for s in signals if s.modality == modality]

            if not modality_signals:
                # No signal for this modality - trend toward closing
                self._trend_gate(modality, AttentionGate.CLOSED)
                continue

            # Get temporal context for this modality
            temporal_context = self.temporal_integrator.get_integrated_context(modality)

            # Calculate gating decision factors
            current_confidence = modality_signals[0].confidence
            integrated_confidence = temporal_context.get('integrated_confidence', current_confidence)
            performance_trend = self._calculate_performance_trend()

            # Combined gating score
            gating_score = (current_confidence * 0.4 +
                          integrated_confidence * 0.4 +
                          performance_trend * 0.2)

            # Update gate based on score
            if gating_score > self.gating_thresholds['open_gate']:
                self.attention_state.active_modalities[modality] = AttentionGate.OPEN
            elif gating_score > self.gating_thresholds['filter_threshold']:
                self.attention_state.active_modalities[modality] = AttentionGate.FILTERING
            else:
                self.attention_state.active_modalities[modality] = AttentionGate.CLOSED

    def _trend_gate(self, modality: Modality, target_gate: AttentionGate) -> None:
        """Gradually trend a gate toward a target state."""
        current_gate = self.attention_state.active_modalities[modality]

        if current_gate == target_gate:
            return

        # Simple trending logic - could be more sophisticated
        gate_hierarchy = [AttentionGate.CLOSED, AttentionGate.FILTERING, AttentionGate.OPEN]
        current_idx = gate_hierarchy.index(current_gate)
        target_idx = gate_hierarchy.index(target_gate)

        if target_idx > current_idx:
            # Trend toward more open
            self.attention_state.active_modalities[modality] = gate_hierarchy[min(current_idx + 1, len(gate_hierarchy) - 1)]
        else:
            # Trend toward more closed
            self.attention_state.active_modalities[modality] = gate_hierarchy[max(current_idx - 1, 0)]

    def _calculate_performance_trend(self) -> float:
        """Calculate recent performance trend (0-1 scale)."""
        if len(self.attention_state.performance_history) < 10:
            return 0.5  # Neutral if insufficient data

        recent = np.mean(self.attention_state.performance_history[-10:])
        older = np.mean(self.attention_state.performance_history[-20:-10]) if len(self.attention_state.performance_history) >= 20 else recent

        if older == 0:
            return 0.5

        trend = (recent - older) / abs(older)
        # Normalize to 0-1 range
        return max(0, min(1, (trend + 1) / 2))

    def _adapt_weights(self, current_performance: float) -> None:
        """Adapt modality weights based on performance."""
        if len(self.attention_state.performance_history) < 5:
            return

        recent_avg = np.mean(self.attention_state.performance_history[-5:])
        overall_avg = np.mean(self.attention_state.performance_history)

        # If recent performance is better than overall, reinforce current weights
        # If worse, encourage exploration of different weight combinations
        adaptation_strength = self.adaptation_rate * (recent_avg - overall_avg)

        for modality in Modality:
            if self.attention_state.active_modalities[modality] == AttentionGate.OPEN:
                # Slightly increase weight for successful modalities
                self.attention_state.modality_weights[modality] += adaptation_strength * 0.1
            elif self.attention_state.active_modalities[modality] == AttentionGate.CLOSED:
                # Slightly decrease weight for unsuccessful modalities
                self.attention_state.modality_weights[modality] -= adaptation_strength * 0.05

            # Keep weights in reasonable bounds
            self.attention_state.modality_weights[modality] = max(0.1, min(1.0, self.attention_state.modality_weights[modality]))

    def get_attention_status(self) -> Dict[str, Any]:
        """Get current attention system status."""
        return {
            'active_gates': {mod.value: gate.value for mod, gate in self.attention_state.active_modalities.items()},
            'modality_weights': {mod.value: weight for mod, weight in self.attention_state.modality_weights.items()},
            'performance_trend': self._calculate_performance_trend(),
            'temporal_buffer_size': len(self.temporal_integrator.signal_buffer),
            'recent_performance': np.mean(self.attention_state.performance_history[-10:]) if self.attention_state.performance_history else 0
        }

# Convenience functions for integration
def create_visual_signal(data: Any, confidence: float = 0.8, priority: float = 0.7) -> ModalitySignal:
    """Create a visual modality signal."""
    return ModalitySignal(
        modality=Modality.VISUAL,
        data=data,
        timestamp=time.time(),
        confidence=confidence,
        priority=priority,
        context={'source': 'camera', 'type': 'visual_features'}
    )

def create_audio_signal(data: Any, confidence: float = 0.6, priority: float = 0.5) -> ModalitySignal:
    """Create an audio modality signal."""
    return ModalitySignal(
        modality=Modality.AUDIO,
        data=data,
        timestamp=time.time(),
        confidence=confidence,
        priority=priority,
        context={'source': 'microphone', 'type': 'audio_features'}
    )

def create_text_signal(data: Any, confidence: float = 0.9, priority: float = 0.4) -> ModalitySignal:
    """Create a text modality signal."""
    return ModalitySignal(
        modality=Modality.TEXT,
        data=data,
        timestamp=time.time(),
        confidence=confidence,
        priority=priority,
        context={'source': 'nlp', 'type': 'text_features'}
    )

def create_tactile_signal(data: Any, confidence: float = 0.8, priority: float = 0.9) -> ModalitySignal:
    """Create a tactile modality signal."""
    return ModalitySignal(
        modality=Modality.TACTILE,
        data=data,
        timestamp=time.time(),
        confidence=confidence,
        priority=priority,
        context={'source': 'sensors', 'type': 'proximity_tactile'}
    )