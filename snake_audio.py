#!/usr/bin/env python3
"""
Snake Audio Integration Module
Integrates audio feedback into the Snake learning environment for multimodal reinforcement learning.
"""

import os
import numpy as np
import pygame
import time
from typing import Optional, Dict, Any
from pathlib import Path

class SnakeAudioSystem:
    """Audio system for Snake game with cognitive audio integration."""

    def __init__(self, sounds_dir: str = "snake_sounds", enabled: bool = True):
        self.enabled = enabled
        self.sounds_dir = Path(sounds_dir)
        self.audio_initialized = False

        if enabled:
            self._initialize_audio()
            self._load_sounds()

    def _initialize_audio(self):
        """Initialize pygame audio system."""
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self.audio_initialized = True
            print("🎵 Audio system initialized")
        except Exception as e:
            print(f"⚠️ Audio initialization failed: {e}")
            self.enabled = False

    def _load_sounds(self):
        """Load sound files."""
        if not self.audio_initialized:
            return

        self.sounds = {}

        sound_files = {
            'eat': 'eat.wav',
            'death': 'death.wav',
            'achievement': 'achievement.wav',
            'move': 'move.wav'
        }

        for sound_name, filename in sound_files.items():
            filepath = self.sounds_dir / filename
            if filepath.exists():
                try:
                    self.sounds[sound_name] = pygame.mixer.Sound(str(filepath))
                    print(f"✅ Loaded sound: {sound_name}")
                except Exception as e:
                    print(f"⚠️ Failed to load {sound_name}: {e}")
            else:
                print(f"⚠️ Sound file not found: {filepath}")

    def play_sound(self, sound_name: str, volume: float = 1.0):
        """Play a sound effect."""
        if not self.enabled or not self.audio_initialized:
            return

        if sound_name in self.sounds:
            self.sounds[sound_name].set_volume(volume)
            self.sounds[sound_name].play()
        else:
            print(f"⚠️ Sound not found: {sound_name}")

    def play_eat_sound(self):
        """Play food eating sound."""
        self.play_sound('eat', 0.7)

    def play_death_sound(self):
        """Play death sound."""
        self.play_sound('death', 0.8)

    def play_achievement_sound(self):
        """Play achievement/milestone sound."""
        self.play_sound('achievement', 0.9)

    def play_move_sound(self):
        """Play movement sound (optional, subtle)."""
        self.play_sound('move', 0.3)

    def cleanup(self):
        """Clean up audio resources."""
        if self.audio_initialized:
            pygame.mixer.quit()
            self.audio_initialized = False

class CognitiveAudioFeedback:
    """Cognitive audio feedback system for reinforcement learning."""

    def __init__(self, audio_system: SnakeAudioSystem):
        self.audio_system = audio_system
        self.audio_enabled = audio_system.enabled

        # Audio feedback configuration
        self.feedback_config = {
            'positive_reward': {'sound': 'eat', 'volume': 0.7, 'delay': 0.1},
            'negative_reward': {'sound': 'death', 'volume': 0.8, 'delay': 0.2},
            'milestone': {'sound': 'achievement', 'volume': 0.9, 'delay': 0.3},
            'movement': {'sound': 'move', 'volume': 0.2, 'delay': 0.0}
        }

    def provide_feedback(self, event_type: str, **kwargs):
        """Provide audio feedback for game events."""
        if not self.audio_enabled or event_type not in self.feedback_config:
            return

        config = self.feedback_config[event_type]

        # Add delay for dramatic effect
        if config['delay'] > 0:
            time.sleep(config['delay'])

        # Play the sound
        if event_type == 'positive_reward':
            self.audio_system.play_eat_sound()
        elif event_type == 'negative_reward':
            self.audio_system.play_death_sound()
        elif event_type == 'milestone':
            self.audio_system.play_achievement_sound()
        elif event_type == 'movement':
            self.audio_system.play_move_sound()

    def get_audio_context(self) -> Dict[str, Any]:
        """Get current audio context for cognitive processing."""
        return {
            'audio_enabled': self.audio_enabled,
            'feedback_types': list(self.feedback_config.keys()),
            'audio_system_status': 'active' if self.audio_enabled else 'disabled'
        }

# Audio-enhanced reward function
def audio_enhanced_reward(base_reward: float, audio_context: Dict[str, Any]) -> float:
    """
    Enhance reward function with audio feedback consideration.

    This implements a cognitive hypothesis: audio feedback can modulate
    reinforcement learning by providing additional sensory reinforcement.
    """
    if not audio_context.get('audio_enabled', False):
        return base_reward

    # Audio can amplify emotional response to rewards
    # Positive audio feedback increases perceived reward value
    # Negative audio feedback increases perceived punishment

    enhancement_factor = 1.0

    if base_reward > 0:
        # Positive rewards are amplified by pleasant sounds
        enhancement_factor = 1.2  # 20% boost for positive feedback
    elif base_reward < 0:
        # Negative rewards are amplified by unpleasant sounds
        enhancement_factor = 1.3  # 30% boost for negative feedback

    return base_reward * enhancement_factor

if __name__ == "__main__":
    # Test the audio system
    print("🎵 Testing Snake Audio System")

    audio_system = SnakeAudioSystem()

    if audio_system.enabled:
        feedback = CognitiveAudioFeedback(audio_system)

        print("🎮 Testing audio feedback...")

        # Test sounds
        feedback.provide_feedback('positive_reward')
        time.sleep(0.5)

        feedback.provide_feedback('milestone')
        time.sleep(0.5)

        feedback.provide_feedback('negative_reward')
        time.sleep(1.0)

        audio_system.cleanup()

        print("✅ Audio system test completed")
    else:
        print("❌ Audio system not available")