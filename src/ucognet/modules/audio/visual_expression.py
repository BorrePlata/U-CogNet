# U-CogNet Visual Expression Module
# Artistic Visual Manifestation of Audio Perceptions

import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime
import colorsys
import logging

from .audio_types import AudioPerception, VisualExpression, VisualSymbol
from .audio_protocols import VisualExpressionProtocol

class ArtisticVisualExpression(VisualExpressionProtocol):
    """Transforms cognitive audio perceptions into artistic visual expressions."""

    def __init__(self):
        # Color palettes for different emotions and sound types
        self._color_palettes = {
            'birdsong': {
                'primary': [(0.3, 0.8, 0.9), (0.2, 0.9, 0.7), (0.4, 0.7, 0.95)],  # Blues and greens
                'secondary': [(0.1, 0.6, 0.8), (0.3, 0.8, 0.6)]
            },
            'explosion': {
                'primary': [(0.0, 0.9, 1.0), (0.1, 0.8, 0.9), (0.05, 1.0, 0.8)],  # Reds and oranges
                'secondary': [(0.0, 0.7, 0.8), (0.15, 0.9, 0.7)]
            },
            'alarm': {
                'primary': [(0.15, 1.0, 0.9), (0.0, 1.0, 0.7), (0.1, 0.9, 1.0)],  # Yellows and reds
                'secondary': [(0.2, 0.8, 0.8), (0.05, 0.95, 0.6)]
            },
            'nature': {
                'primary': [(0.25, 0.7, 0.8), (0.35, 0.6, 0.9), (0.45, 0.8, 0.7)],  # Greens and earth tones
                'secondary': [(0.3, 0.5, 0.7), (0.4, 0.7, 0.6)]
            },
            'urban': {
                'primary': [(0.6, 0.3, 0.8), (0.5, 0.4, 0.9), (0.7, 0.2, 0.7)],  # Purples and grays
                'secondary': [(0.55, 0.35, 0.75), (0.65, 0.25, 0.8)]
            },
            'unknown': {
                'primary': [(0.5, 0.5, 0.5), (0.6, 0.6, 0.6), (0.4, 0.4, 0.4)],  # Neutrals
                'secondary': [(0.45, 0.45, 0.45), (0.55, 0.55, 0.55)]
            }
        }

        # Shape mappings for different temporal patterns
        self._shape_mappings = {
            'pulsed': ['circle', 'burst', 'radial'],
            'rhythmic': ['wave', 'pulse', 'grid'],
            'continuous': ['flow', 'gradient', 'organic'],
            'intermittent': ['spark', 'dot', 'fragment'],
            'random': ['chaos', 'scatter', 'abstract']
        }

        # Symbol libraries for different sound types
        self._symbol_libraries = {
            'birdsong': ['🌿', '🐦', '🌸', '🍃', '🌼'],
            'explosion': ['💥', '🔥', '⚡', '💫', '🌟'],
            'alarm': ['🚨', '⚠️', '🔊', '📢', '🔔'],
            'nature': ['🌳', '🌊', '🍂', '🌱', '🦋'],
            'urban': ['🏙️', '🚗', '🏢', '🚦', '🌆'],
            'unknown': ['❓', '🎵', '🎶', '🎼', '🎹']
        }

    async def express_visually(self, perception: AudioPerception) -> VisualExpression:
        """Transform audio perception into visual expression."""
        # Generate base visual elements
        colors = await self._generate_color_palette(perception)
        shapes = await self._generate_shapes(perception)
        symbols = await self._generate_symbols(perception)

        # Create composition
        composition = await self._create_composition(perception, colors, shapes, symbols)

        # Add dynamic elements based on temporal patterns
        dynamics = await self._add_dynamics(perception)

        # Generate final expression
        expression = VisualExpression(
            timestamp=perception.timestamp,
            perception=perception,
            colors=colors,
            shapes=shapes,
            symbols=symbols,
            composition=composition,
            dynamics=dynamics,
            style=self._determine_style(perception),
            intensity=self._calculate_intensity(perception),
            metadata={
                'expression_method': 'artistic_synthesis',
                'emotional_mapping': self._map_emotion_to_visual(perception)
            }
        )

        return expression

    async def _generate_color_palette(self, perception: AudioPerception) -> List[Tuple[float, float, float]]:
        """Generate HSV color palette based on perception."""
        sound_type = perception.sound_type
        valence = perception.emotional_valence
        arousal = perception.arousal_level

        # Get base palette for sound type
        base_palette = self._color_palettes.get(sound_type, self._color_palettes['unknown'])

        # Modify based on emotional content
        modified_colors = []

        for h, s, v in base_palette['primary']:
            # Adjust hue based on valence (-1 to 1)
            hue_shift = valence * 0.1  # Small shift for emotional expression
            new_h = (h + hue_shift) % 1.0

            # Adjust saturation and value based on arousal
            new_s = min(1.0, s + arousal * 0.2)
            new_v = min(1.0, v + arousal * 0.1)

            modified_colors.append((new_h, new_s, new_v))

        # Add secondary colors with variation
        for h, s, v in base_palette['secondary']:
            # More variation for secondary colors
            hue_shift = valence * 0.15 + np.random.uniform(-0.05, 0.05)
            new_h = (h + hue_shift) % 1.0
            new_s = min(1.0, s + arousal * 0.3)
            new_v = min(1.0, v + arousal * 0.2)

            modified_colors.append((new_h, new_s, new_v))

        return modified_colors

    async def _generate_shapes(self, perception: AudioPerception) -> List[Dict[str, Any]]:
        """Generate visual shapes based on temporal patterns and features."""
        temporal_pattern = perception.temporal_pattern
        arousal = perception.arousal_level
        rms_energy = perception.features.rms_energy

        # Get shape types for temporal pattern
        shape_types = self._shape_mappings.get(temporal_pattern, ['abstract'])

        shapes = []

        # Generate shapes based on energy and arousal
        num_shapes = max(1, int(arousal * 5) + int(rms_energy * 3))

        for i in range(num_shapes):
            shape_type = np.random.choice(shape_types)

            # Base properties
            shape = {
                'type': shape_type,
                'size': np.random.uniform(0.1, 1.0) * (arousal + 0.5),
                'position': (np.random.uniform(0, 1), np.random.uniform(0, 1)),
                'rotation': np.random.uniform(0, 2 * np.pi),
                'opacity': np.random.uniform(0.3, 1.0)
            }

            # Shape-specific properties
            if shape_type == 'circle':
                shape['radius'] = shape['size'] * 50
            elif shape_type == 'wave':
                shape['amplitude'] = shape['size'] * 20
                shape['frequency'] = np.random.uniform(1, 5)
            elif shape_type == 'burst':
                shape['num_rays'] = np.random.randint(5, 12)
                shape['ray_length'] = shape['size'] * 30
            elif shape_type == 'flow':
                shape['flow_direction'] = np.random.uniform(0, 2 * np.pi)
                shape['flow_speed'] = arousal * 2
            elif shape_type == 'spark':
                shape['sparkle_intensity'] = arousal * 2
                shape['trail_length'] = np.random.uniform(5, 20)

            shapes.append(shape)

        return shapes

    async def _generate_symbols(self, perception: AudioPerception) -> List[VisualSymbol]:
        """Generate symbolic representations."""
        sound_type = perception.sound_type
        attention_weight = perception.attention_weight
        emotional_intensity = abs(perception.emotional_valence) + perception.arousal_level

        # Get symbol library
        symbols = self._symbol_libraries.get(sound_type, self._symbol_libraries['unknown'])

        visual_symbols = []

        # Number of symbols based on attention and emotional intensity
        num_symbols = max(1, int(attention_weight * 3) + int(emotional_intensity * 2))

        for i in range(num_symbols):
            symbol_char = np.random.choice(symbols)

            # Create visual symbol
            symbol = VisualSymbol(
                symbol=symbol_char,
                position=(np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)),
                size=np.random.uniform(20, 60) * (attention_weight + 0.5),
                color=self._get_symbol_color(perception),
                opacity=np.random.uniform(0.5, 1.0),
                animation=self._get_symbol_animation(perception),
                meaning=self._interpret_symbol_meaning(symbol_char, perception)
            )

            visual_symbols.append(symbol)

        return visual_symbols

    async def _create_composition(self, perception: AudioPerception,
                                colors: List[Tuple[float, float, float]],
                                shapes: List[Dict[str, Any]],
                                symbols: List[VisualSymbol]) -> Dict[str, Any]:
        """Create overall visual composition."""
        # Determine layout based on sound characteristics
        layout = self._determine_layout(perception)

        # Calculate composition metrics
        composition = {
            'layout': layout,
            'background_color': self._get_background_color(perception),
            'contrast_level': self._calculate_contrast(perception),
            'balance': self._calculate_balance(shapes, symbols),
            'harmony': self._calculate_harmony(colors),
            'elements': {
                'shapes': shapes,
                'symbols': symbols,
                'colors': colors
            },
            'dimensions': (800, 600),  # Default canvas size
            'style_properties': self._get_style_properties(perception)
        }

        return composition

    async def _add_dynamics(self, perception: AudioPerception) -> Dict[str, Any]:
        """Add dynamic elements based on temporal patterns."""
        temporal_pattern = perception.temporal_pattern
        tempo = perception.features.tempo
        arousal = perception.arousal_level

        dynamics = {
            'animation_speed': tempo / 120.0 * arousal,  # BPM normalized
            'pulse_rate': self._calculate_pulse_rate(perception),
            'flow_direction': np.random.uniform(0, 2 * np.pi),
            'transformation_sequence': self._get_transformation_sequence(temporal_pattern),
            'interaction_effects': self._generate_interaction_effects(perception)
        }

        return dynamics

    def _determine_style(self, perception: AudioPerception) -> str:
        """Determine artistic style based on perception."""
        sound_type = perception.sound_type
        emotional_valence = perception.emotional_valence
        arousal = perception.arousal_level

        # Style mapping
        if sound_type == 'birdsong':
            return 'organic' if emotional_valence > 0 else 'ethereal'
        elif sound_type == 'explosion':
            return 'dynamic' if arousal > 0.7 else 'intense'
        elif sound_type == 'alarm':
            return 'urgent' if arousal > 0.6 else 'warning'
        elif sound_type == 'nature':
            return 'naturalistic' if emotional_valence > 0 else 'mystical'
        elif sound_type == 'urban':
            return 'geometric' if arousal < 0.5 else 'chaotic'
        else:
            return 'abstract'

    def _calculate_intensity(self, perception: AudioPerception) -> float:
        """Calculate visual intensity."""
        rms_energy = perception.features.rms_energy
        arousal = perception.arousal_level
        attention = perception.attention_weight

        intensity = (rms_energy + arousal + attention) / 3.0
        return min(1.0, intensity)

    def _map_emotion_to_visual(self, perception: AudioPerception) -> Dict[str, Any]:
        """Map emotional content to visual properties."""
        valence = perception.emotional_valence
        arousal = perception.arousal_level

        # Color temperature (cool to warm)
        color_temp = (valence + 1) / 2  # 0 = cool, 1 = warm

        # Saturation (muted to vivid)
        saturation = arousal

        # Complexity (simple to complex)
        complexity = arousal * 0.5 + abs(valence) * 0.5

        return {
            'color_temperature': color_temp,
            'saturation': saturation,
            'complexity': complexity,
            'mood': self._classify_mood(valence, arousal)
        }

    def _get_symbol_color(self, perception: AudioPerception) -> Tuple[float, float, float]:
        """Get color for symbols."""
        valence = perception.emotional_valence
        arousal = perception.arousal_level

        # Base hue from valence
        hue = (valence + 1) / 2 * 0.7  # 0 = red, 0.7 = blue

        # Saturation and value from arousal
        saturation = 0.7 + arousal * 0.3
        value = 0.8 + arousal * 0.2

        return (hue, saturation, value)

    def _get_symbol_animation(self, perception: AudioPerception) -> str:
        """Determine symbol animation type."""
        temporal_pattern = perception.temporal_pattern
        arousal = perception.arousal_level

        if temporal_pattern == 'pulsed':
            return 'bounce' if arousal > 0.5 else 'fade'
        elif temporal_pattern == 'rhythmic':
            return 'pulse'
        elif temporal_pattern == 'continuous':
            return 'glow'
        elif temporal_pattern == 'intermittent':
            return 'sparkle'
        else:
            return 'drift'

    def _interpret_symbol_meaning(self, symbol: str, perception: AudioPerception) -> str:
        """Interpret the meaning of a symbol in context."""
        sound_type = perception.sound_type

        # Basic interpretations
        interpretations = {
            'birdsong': {
                '🌿': 'growth and renewal',
                '🐦': 'freedom and melody',
                '🌸': 'beauty and transience',
                '🍃': 'gentle movement',
                '🌼': 'joy and color'
            },
            'explosion': {
                '💥': 'sudden release',
                '🔥': 'intense energy',
                '⚡': 'electric power',
                '💫': 'dizzying impact',
                '🌟': 'destructive brilliance'
            },
            'alarm': {
                '🚨': 'urgent warning',
                '⚠️': 'caution needed',
                '🔊': 'loud attention',
                '📢': 'public announcement',
                '🔔': 'alert signal'
            }
        }

        type_interpretations = interpretations.get(sound_type, {})
        return type_interpretations.get(symbol, 'abstract representation')

    def _determine_layout(self, perception: AudioPerception) -> str:
        """Determine composition layout."""
        temporal_pattern = perception.temporal_pattern
        arousal = perception.arousal_level

        if temporal_pattern == 'pulsed':
            return 'radial' if arousal > 0.6 else 'scattered'
        elif temporal_pattern == 'rhythmic':
            return 'grid' if arousal < 0.5 else 'wave'
        elif temporal_pattern == 'continuous':
            return 'flowing'
        elif temporal_pattern == 'intermittent':
            return 'clustered'
        else:
            return 'organic'

    def _get_background_color(self, perception: AudioPerception) -> Tuple[float, float, float]:
        """Determine background color."""
        valence = perception.emotional_valence
        arousal = perception.arousal_level

        # Darker background for high arousal, lighter for positive valence
        value = 0.2 + (valence + 1) / 2 * 0.3 - arousal * 0.1
        value = max(0.1, min(0.8, value))

        # Slight hue variation
        hue = (valence + 1) / 2 * 0.1
        saturation = 0.1 + arousal * 0.2

        return (hue, saturation, value)

    def _calculate_contrast(self, perception: AudioPerception) -> float:
        """Calculate visual contrast level."""
        rms_energy = perception.features.rms_energy
        arousal = perception.arousal_level

        contrast = rms_energy * 0.6 + arousal * 0.4
        return min(1.0, contrast)

    def _calculate_balance(self, shapes: List[Dict[str, Any]],
                          symbols: List[VisualSymbol]) -> float:
        """Calculate visual balance of composition."""
        if not shapes and not symbols:
            return 0.5

        # Simple center of mass calculation
        total_weight = 0
        weighted_x = 0
        weighted_y = 0

        for shape in shapes:
            weight = shape.get('size', 1.0) * shape.get('opacity', 1.0)
            x, y = shape['position']
            weighted_x += x * weight
            weighted_y += y * weight
            total_weight += weight

        for symbol in symbols:
            weight = symbol.size * symbol.opacity
            x, y = symbol.position
            weighted_x += x * weight
            weighted_y += y * weight
            total_weight += weight

        if total_weight == 0:
            return 0.5

        center_x = weighted_x / total_weight
        center_y = weighted_y / total_weight

        # Distance from center (0.5, 0.5)
        distance_from_center = np.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2)
        balance = 1.0 - min(distance_from_center * 2, 1.0)  # Normalize to [0,1]

        return balance

    def _calculate_harmony(self, colors: List[Tuple[float, float, float]]) -> float:
        """Calculate color harmony."""
        if len(colors) < 2:
            return 1.0

        # Simple harmony based on hue differences
        hues = [h for h, s, v in colors]
        hue_diffs = []

        for i in range(len(hues)):
            for j in range(i+1, len(hues)):
                diff = min(abs(hues[i] - hues[j]), 1 - abs(hues[i] - hues[j]))
                hue_diffs.append(diff)

        avg_diff = np.mean(hue_diffs)
        # Harmony peaks at complementary colors (0.5) and analogous (0.1-0.2)
        harmony = 1.0 - abs(avg_diff - 0.5) * 2
        harmony = max(0.0, harmony)

        return harmony

    def _get_style_properties(self, perception: AudioPerception) -> Dict[str, Any]:
        """Get style-specific properties."""
        style = self._determine_style(perception)

        properties = {
            'organic': {
                'edge_softness': 0.8,
                'natural_variation': 0.7,
                'flow_emphasis': 0.6
            },
            'dynamic': {
                'motion_blur': 0.5,
                'energy_lines': 0.8,
                'impact_rings': 0.7
            },
            'urgent': {
                'sharp_edges': 0.9,
                'high_contrast': 0.8,
                'warning_colors': 0.7
            },
            'abstract': {
                'geometric_forms': 0.6,
                'color_fields': 0.5,
                'minimalist': 0.4
            }
        }

        return properties.get(style, {})

    def _calculate_pulse_rate(self, perception: AudioPerception) -> float:
        """Calculate animation pulse rate."""
        tempo = perception.features.tempo
        arousal = perception.arousal_level

        base_rate = tempo / 60.0  # Convert BPM to Hz
        pulse_rate = base_rate * (0.5 + arousal * 0.5)

        return pulse_rate

    def _get_transformation_sequence(self, temporal_pattern: str) -> List[str]:
        """Get sequence of visual transformations."""
        sequences = {
            'pulsed': ['scale', 'fade', 'burst'],
            'rhythmic': ['pulse', 'rotate', 'wave'],
            'continuous': ['flow', 'morph', 'drift'],
            'intermittent': ['appear', 'sparkle', 'vanish'],
            'random': ['transform', 'recolor', 'reposition']
        }

        return sequences.get(temporal_pattern, ['fade'])

    def _generate_interaction_effects(self, perception: AudioPerception) -> List[Dict[str, Any]]:
        """Generate interaction effects between elements."""
        arousal = perception.arousal_level
        num_effects = max(0, int(arousal * 3))

        effects = []
        effect_types = ['attraction', 'repulsion', 'orbit', 'merge', 'split']

        for _ in range(num_effects):
            effect = {
                'type': np.random.choice(effect_types),
                'strength': np.random.uniform(0.1, 1.0),
                'duration': np.random.uniform(1, 5),
                'elements': np.random.randint(2, 5)
            }
            effects.append(effect)

        return effects

    def _classify_mood(self, valence: float, arousal: float) -> str:
        """Classify emotional mood."""
        if valence > 0.5 and arousal > 0.5:
            return 'excited'
        elif valence > 0.5 and arousal < 0.5:
            return 'calm'
        elif valence < -0.5 and arousal > 0.5:
            return 'angry'
        elif valence < -0.5 and arousal < 0.5:
            return 'sad'
        else:
            return 'neutral'

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the visual expression module."""
        # Update palettes if provided
        if 'color_palettes' in config:
            self._color_palettes.update(config['color_palettes'])

        if 'symbol_libraries' in config:
            self._symbol_libraries.update(config['symbol_libraries'])

        logging.info("Initialized ArtisticVisualExpression")

    async def cleanup(self) -> None:
        """Clean up resources."""
    pass
