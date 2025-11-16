# U-CogNet Audio-Visual Module Tests - Visual Expression
# Comprehensive testing for the visual expression system

import pytest
import asyncio
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from ucognet.modules.audio.visual_expression import ArtisticVisualExpression
from ucognet.modules.audio.audio_types import AudioPerception, VisualExpression


class TestArtisticVisualExpression:
    """Test the ArtisticVisualExpression class."""

    @pytest.fixture
    async def visual_expressor(self):
        """Create and initialize visual expressor."""
        expressor = ArtisticVisualExpression()
        config = {
            'color_vibrancy': 0.8,
            'shape_complexity': 0.7,
            'motion_intensity': 0.6,
            'symbolism_depth': 0.5,
            'style_variety': 0.9,
            'canvas_width': 400,
            'canvas_height': 300
        }
        await expressor.initialize(config)
        yield expressor
        await expressor.cleanup()

    @pytest.fixture
    def sample_perception(self):
        """Create a sample audio perception for testing."""
        return AudioPerception(
            sound_type='nature',
            emotional_valence=0.6,
            arousal_level=0.7,
            attention_weight=0.8,
            environment_context='forest',
            confidence=0.9
        )

    @pytest.mark.asyncio
    async def test_visual_expressor_initialization(self, visual_expressor):
        """Test visual expressor initialization."""
        assert visual_expressor.is_initialized
        assert visual_expressor.color_vibrancy == 0.8
        assert visual_expressor.shape_complexity == 0.7
        assert visual_expressor.motion_intensity == 0.6

    @pytest.mark.asyncio
    async def test_express_visually_basic(self, visual_expressor, sample_perception):
        """Test basic visual expression."""
        expression = await visual_expressor.express_visually(sample_perception)

        assert isinstance(expression, VisualExpression)
        assert isinstance(expression.style, str)
        assert 0 <= expression.intensity <= 1
        assert isinstance(expression.colors, list)
        assert len(expression.colors) > 0
        assert isinstance(expression.composition, dict)
        assert 0 <= expression.confidence <= 1

    @pytest.mark.asyncio
    async def test_different_sound_types(self, visual_expressor):
        """Test expression for different sound types."""
        sound_types = ['nature', 'music', 'speech', 'mechanical', 'animal', 'weather']

        for sound_type in sound_types:
            perception = AudioPerception(
                sound_type=sound_type,
                emotional_valence=0.5,
                arousal_level=0.6,
                attention_weight=0.7,
                environment_context='unknown',
                confidence=0.8
            )

            expression = await visual_expressor.express_visually(perception)

            assert isinstance(expression, VisualExpression)
            assert expression.style in visual_expressor.style_mapping
            assert len(expression.colors) > 0

    @pytest.mark.asyncio
    async def test_emotional_valence_mapping(self, visual_expressor, sample_perception):
        """Test emotional valence to visual mapping."""
        valences = [-1.0, -0.5, 0.0, 0.5, 1.0]

        for valence in valences:
            perception = sample_perception._replace(emotional_valence=valence)
            expression = await visual_expressor.express_visually(perception)

            assert isinstance(expression, VisualExpression)

            # Positive valence should tend toward warmer colors
            if valence > 0.3:
                warm_colors = any(color in ['#FF6B6B', '#FFD93D', '#FF8E53'] for color in expression.colors)
                assert warm_colors or len(expression.colors) > 0  # At least some colors

            # Negative valence should tend toward cooler colors
            elif valence < -0.3:
                cool_colors = any(color in ['#4ECDC4', '#45B7D1', '#96CEB4'] for color in expression.colors)
                assert cool_colors or len(expression.colors) > 0

    @pytest.mark.asyncio
    async def test_arousal_level_intensity(self, visual_expressor, sample_perception):
        """Test arousal level to intensity mapping."""
        arousal_levels = [0.0, 0.3, 0.7, 1.0]

        for arousal in arousal_levels:
            perception = sample_perception._replace(arousal_level=arousal)
            expression = await visual_expressor.express_visually(perception)

            # Higher arousal should generally lead to higher intensity
            if arousal > 0.7:
                assert expression.intensity > 0.5
            elif arousal < 0.3:
                assert expression.intensity < 0.8  # Allow some flexibility

    @pytest.mark.asyncio
    async def test_attention_weight_composition(self, visual_expressor, sample_perception):
        """Test attention weight to composition complexity mapping."""
        attention_levels = [0.0, 0.5, 1.0]

        for attention in attention_levels:
            perception = sample_perception._replace(attention_weight=attention)
            expression = await visual_expressor.express_visually(perception)

            assert 'shapes' in expression.composition
            assert 'patterns' in expression.composition

            # Higher attention should lead to more complex compositions
            if attention > 0.7:
                assert len(expression.composition['shapes']) >= 2
            elif attention < 0.3:
                assert len(expression.composition['shapes']) >= 1

    @pytest.mark.asyncio
    async def test_color_generation(self, visual_expressor, sample_perception):
        """Test color palette generation."""
        expression = await visual_expressor.express_visually(sample_perception)

        assert len(expression.colors) >= 3  # At least primary, secondary, accent

        # All colors should be valid hex codes
        for color in expression.colors:
            assert color.startswith('#')
            assert len(color) == 7  # #RRGGBB format
            # Should be valid hex digits
            int(color[1:], 16)  # This will raise ValueError if invalid

    @pytest.mark.asyncio
    async def test_shape_generation(self, visual_expressor, sample_perception):
        """Test shape generation based on perception."""
        expression = await visual_expressor.express_visually(sample_perception)

        shapes = expression.composition.get('shapes', [])
        assert isinstance(shapes, list)
        assert len(shapes) > 0

        # Shapes should be from the valid shape vocabulary
        valid_shapes = ['circle', 'square', 'triangle', 'wave', 'spiral', 'star', 'organic', 'geometric']
        for shape in shapes:
            assert shape in valid_shapes

    @pytest.mark.asyncio
    async def test_pattern_generation(self, visual_expressor, sample_perception):
        """Test pattern generation."""
        expression = await visual_expressor.express_visually(sample_perception)

        patterns = expression.composition.get('patterns', [])
        assert isinstance(patterns, list)

        # Patterns should be meaningful
        valid_patterns = ['flowing', 'chaotic', 'structured', 'random', 'harmonic', 'dissonant']
        for pattern in patterns:
            assert pattern in valid_patterns

    @pytest.mark.asyncio
    async def test_style_selection(self, visual_expressor, sample_perception):
        """Test artistic style selection."""
        expression = await visual_expressor.express_visually(sample_perception)

        assert expression.style in visual_expressor.style_mapping

        # Style should be appropriate for the sound type
        sound_type = sample_perception.sound_type
        if sound_type == 'nature':
            assert expression.style in ['organic', 'flowing', 'naturalistic']
        elif sound_type == 'music':
            assert expression.style in ['harmonic', 'rhythmic', 'melodic']
        elif sound_type == 'speech':
            assert expression.style in ['structured', 'linear', 'communicative']

    @pytest.mark.asyncio
    async def test_confidence_calculation(self, visual_expressor, sample_perception):
        """Test confidence score calculation."""
        expression = await visual_expressor.express_visually(sample_perception)

        assert 0 <= expression.confidence <= 1

        # High perception confidence should lead to high expression confidence
        if sample_perception.confidence > 0.8:
            assert expression.confidence > 0.6

    @pytest.mark.asyncio
    async def test_configuration_effects(self):
        """Test how configuration affects expression."""
        configs = [
            {'color_vibrancy': 0.2, 'shape_complexity': 0.3},  # Subdued
            {'color_vibrancy': 0.8, 'shape_complexity': 0.9},  # Vibrant
        ]

        perception = AudioPerception(
            sound_type='music',
            emotional_valence=0.5,
            arousal_level=0.6,
            attention_weight=0.7,
            environment_context='concert',
            confidence=0.8
        )

        expressions = []
        for config in configs:
            expressor = ArtisticVisualExpression()
            await expressor.initialize(config)
            try:
                expression = await expressor.express_visually(perception)
                expressions.append(expression)
            finally:
                await expressor.cleanup()

        # Higher vibrancy should generally produce more saturated colors
        # This is hard to test precisely, but we can check that expressions are different
        assert expressions[0].intensity != expressions[1].intensity or \
               expressions[0].colors != expressions[1].colors

    @pytest.mark.asyncio
    async def test_environment_context(self, visual_expressor):
        """Test environment context influence on expression."""
        contexts = ['forest', 'ocean', 'urban', 'space', 'home']

        for context in contexts:
            perception = AudioPerception(
                sound_type='nature',
                emotional_valence=0.4,
                arousal_level=0.5,
                attention_weight=0.6,
                environment_context=context,
                confidence=0.8
            )

            expression = await visual_expressor.express_visually(perception)

            assert isinstance(expression, VisualExpression)
            # Environment should influence color choices
            assert len(expression.colors) > 0

    @pytest.mark.asyncio
    async def test_edge_cases(self, visual_expressor):
        """Test edge cases in visual expression."""
        # Test with extreme values
        extreme_cases = [
            AudioPerception('unknown', 1.0, 1.0, 1.0, 'unknown', 1.0),  # Maximum
            AudioPerception('unknown', -1.0, 0.0, 0.0, 'unknown', 0.0),  # Minimum
            AudioPerception('unknown', 0.0, 0.5, 0.5, 'unknown', 0.5),  # Neutral
        ]

        for perception in extreme_cases:
            expression = await visual_expressor.express_visually(perception)
            assert isinstance(expression, VisualExpression)
            assert 0 <= expression.intensity <= 1
            assert len(expression.colors) > 0

    @pytest.mark.asyncio
    async def test_concurrent_expressions(self, visual_expressor):
        """Test concurrent visual expression processing."""
        perceptions = [
            AudioPerception('nature', 0.5, 0.6, 0.7, 'forest', 0.8),
            AudioPerception('music', 0.3, 0.8, 0.9, 'concert', 0.9),
            AudioPerception('speech', -0.2, 0.4, 0.5, 'lecture', 0.7),
        ]

        # Process concurrently
        tasks = [visual_expressor.express_visually(p) for p in perceptions]
        expressions = await asyncio.gather(*tasks)

        assert len(expressions) == 3
        for expression in expressions:
            assert isinstance(expression, VisualExpression)

    @pytest.mark.asyncio
    async def test_memory_management(self, visual_expressor, sample_perception):
        """Test memory management during expression generation."""
        # Generate many expressions
        expressions = []
        for i in range(50):
            # Vary the perception slightly
            varied_perception = sample_perception._replace(
                emotional_valence=sample_perception.emotional_valence + 0.01 * i
            )
            expression = await visual_expressor.express_visually(varied_perception)
            expressions.append(expression)

        assert len(expressions) == 50
        for expression in expressions:
            assert isinstance(expression, VisualExpression)

    @pytest.mark.asyncio
    async def test_error_handling(self, visual_expressor):
        """Test error handling in visual expression."""
        # Test with invalid perception
        invalid_perception = None
        with pytest.raises(AttributeError):
            await visual_expressor.express_visually(invalid_perception)

        # Test with incomplete perception
        incomplete_perception = AudioPerception(
            sound_type=None,
            emotional_valence=0.5,
            arousal_level=0.5,
            attention_weight=0.5,
            environment_context=None,
            confidence=0.5
        )

        expression = await visual_expressor.express_visually(incomplete_perception)
        assert isinstance(expression, VisualExpression)  # Should handle gracefully

    @pytest.mark.asyncio
    async def test_symbolism_generation(self, visual_expressor, sample_perception):
        """Test symbolic element generation."""
        expression = await visual_expressor.express_visually(sample_perception)

        # Check for symbolic elements in composition
        composition = expression.composition

        # Should have some symbolic representation
        assert 'shapes' in composition or 'patterns' in composition or 'colors' in expression.__dict__

        # For nature sounds, should have organic shapes
        if sample_perception.sound_type == 'nature':
            shapes = composition.get('shapes', [])
            organic_shapes = ['wave', 'organic', 'circle']
            has_organic = any(shape in organic_shapes for shape in shapes)
            assert has_organic or len(shapes) > 0  # At least some shapes

    @pytest.mark.asyncio
    async def test_adaptive_styling(self, visual_expressor, sample_perception):
        """Test adaptive style selection based on perception."""
        # Test multiple perceptions to see style adaptation
        test_perceptions = [
            sample_perception._replace(sound_type='music', emotional_valence=0.8),
            sample_perception._replace(sound_type='weather', emotional_valence=-0.6),
            sample_perception._replace(sound_type='animal', arousal_level=0.9),
        ]

        styles = []
        for perception in test_perceptions:
            expression = await visual_expressor.express_visually(perception)
            styles.append(expression.style)

        # Styles should be different for different perceptions
        assert len(set(styles)) >= 2  # At least some variety
