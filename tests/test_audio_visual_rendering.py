# U-CogNet Audio-Visual Module Tests - Visual Rendering
# Comprehensive testing for the visual rendering system

import pytest
import asyncio
import numpy as np
from PIL import Image
from io import BytesIO
from unittest.mock import patch, MagicMock, AsyncMock
from ucognet.modules.audio.visual_rendering import ArtisticVisualRenderer
from ucognet.modules.audio.audio_types import VisualExpression, RenderedVisual


class TestArtisticVisualRenderer:
    """Test the ArtisticVisualRenderer class."""

    @pytest.fixture
    async def visual_renderer(self):
        """Create and initialize visual renderer."""
        renderer = ArtisticVisualRenderer()
        config = {
            'canvas_width': 400,
            'canvas_height': 300,
            'background_color': '#FFFFFF',
            'anti_aliasing': True,
            'color_depth': 32,
            'output_formats': ['image', 'numpy', 'base64'],
            'quality_preset': 'high'
        }
        await renderer.initialize(config)
        yield renderer
        await renderer.cleanup()

    @pytest.fixture
    def sample_expression(self):
        """Create a sample visual expression for testing."""
        return VisualExpression(
            style='organic',
            intensity=0.75,
            colors=['#228B22', '#32CD32', '#006400', '#90EE90'],
            composition={
                'shapes': ['circle', 'wave', 'organic'],
                'patterns': ['flowing', 'harmonic'],
                'layout': 'centered',
                'density': 0.7
            },
            confidence=0.85
        )

    @pytest.mark.asyncio
    async def test_visual_renderer_initialization(self, visual_renderer):
        """Test visual renderer initialization."""
        assert visual_renderer.is_initialized
        assert visual_renderer.canvas_width == 400
        assert visual_renderer.canvas_height == 300
        assert visual_renderer.background_color == '#FFFFFF'

    @pytest.mark.asyncio
    async def test_render_visual_basic(self, visual_renderer, sample_expression):
        """Test basic visual rendering."""
        rendered = await visual_renderer.render_visual(sample_expression)

        assert isinstance(rendered, RenderedVisual)
        assert rendered.data is not None
        assert rendered.format in ['image', 'numpy', 'base64']
        assert len(rendered.dimensions) == 2
        assert rendered.dimensions[0] == 400  # width
        assert rendered.dimensions[1] == 300  # height

    @pytest.mark.asyncio
    async def test_render_different_formats(self, visual_renderer, sample_expression):
        """Test rendering in different output formats."""
        formats = ['image', 'numpy', 'base64']

        for fmt in formats:
            rendered = await visual_renderer.render_visual(sample_expression, format_type=fmt)

            assert rendered.format == fmt
            assert rendered.data is not None

            if fmt == 'image':
                assert isinstance(rendered.data, bytes)
                # Should be valid image data
                image = Image.open(BytesIO(rendered.data))
                assert image.size == (400, 300)

            elif fmt == 'numpy':
                assert isinstance(rendered.data, np.ndarray)
                assert rendered.data.shape[:2] == (300, 400)  # height, width

            elif fmt == 'base64':
                assert isinstance(rendered.data, str)
                # Should be base64 encoded (contains valid chars)
                import base64
                # Remove data URL prefix if present
                data = rendered.data.split(',')[-1] if ',' in rendered.data else rendered.data
                base64.b64decode(data)  # Should not raise exception

    @pytest.mark.asyncio
    async def test_render_different_styles(self, visual_renderer):
        """Test rendering different artistic styles."""
        styles = ['organic', 'geometric', 'abstract', 'minimalist', 'chaotic']

        for style in styles:
            expression = VisualExpression(
                style=style,
                intensity=0.7,
                colors=['#FF6B6B', '#4ECDC4', '#FFD93D'],
                composition={
                    'shapes': ['circle', 'square', 'triangle'] if style == 'geometric' else ['circle', 'wave'],
                    'patterns': ['structured'] if style == 'geometric' else ['flowing'],
                    'layout': 'balanced'
                },
                confidence=0.8
            )

            rendered = await visual_renderer.render_visual(expression)

            assert isinstance(rendered, RenderedVisual)
            assert rendered.data is not None

    @pytest.mark.asyncio
    async def test_intensity_effects(self, visual_renderer, sample_expression):
        """Test how intensity affects rendering."""
        intensities = [0.1, 0.5, 0.9]

        renderings = []
        for intensity in intensities:
            expression = sample_expression._replace(intensity=intensity)
            rendered = await visual_renderer.render_visual(expression)
            renderings.append(rendered)

        # All should be valid renderings
        for rendered in renderings:
            assert isinstance(rendered, RenderedVisual)
            assert rendered.data is not None

    @pytest.mark.asyncio
    async def test_color_palette_rendering(self, visual_renderer):
        """Test rendering with different color palettes."""
        color_palettes = [
            ['#FF0000', '#00FF00', '#0000FF'],  # Primary colors
            ['#FFFFFF', '#808080', '#000000'],  # Monochrome
            ['#FF6B6B', '#4ECDC4', '#FFD93D', '#FF8E53'],  # Warm/cool
            ['#228B22', '#32CD32', '#006400', '#90EE90'],  # Nature greens
        ]

        for colors in color_palettes:
            expression = VisualExpression(
                style='organic',
                intensity=0.7,
                colors=colors,
                composition={
                    'shapes': ['circle', 'wave'],
                    'patterns': ['flowing'],
                    'layout': 'natural'
                },
                confidence=0.8
            )

            rendered = await visual_renderer.render_visual(expression)

            assert isinstance(rendered, RenderedVisual)
            assert rendered.data is not None

    @pytest.mark.asyncio
    async def test_shape_rendering(self, visual_renderer, sample_expression):
        """Test rendering different shapes."""
        shapes_list = [
            ['circle'],
            ['square', 'triangle'],
            ['wave', 'spiral'],
            ['circle', 'square', 'triangle', 'wave']
        ]

        for shapes in shapes_list:
            expression = sample_expression._replace(
                composition={
                    **sample_expression.composition,
                    'shapes': shapes
                }
            )

            rendered = await visual_renderer.render_visual(expression)

            assert isinstance(rendered, RenderedVisual)
            assert rendered.data is not None

    @pytest.mark.asyncio
    async def test_canvas_sizes(self):
        """Test rendering on different canvas sizes."""
        sizes = [(200, 150), (800, 600), (1920, 1080)]

        for width, height in sizes:
            renderer = ArtisticVisualRenderer()
            config = {'canvas_width': width, 'canvas_height': height}
            await renderer.initialize(config)

            try:
                expression = VisualExpression(
                    style='minimalist',
                    intensity=0.5,
                    colors=['#000000'],
                    composition={'shapes': ['circle'], 'patterns': ['simple']},
                    confidence=0.7
                )

                rendered = await renderer.render_visual(expression)

                assert rendered.dimensions == (width, height)

                if rendered.format == 'numpy':
                    assert rendered.data.shape[:2] == (height, width)
                elif rendered.format == 'image':
                    image = Image.open(BytesIO(rendered.data))
                    assert image.size == (width, height)

            finally:
                await renderer.cleanup()

    @pytest.mark.asyncio
    async def test_background_colors(self, visual_renderer, sample_expression):
        """Test different background colors."""
        backgrounds = ['#FFFFFF', '#000000', '#808080', '#FF6B6B']

        for bg_color in backgrounds:
            visual_renderer.background_color = bg_color
            rendered = await visual_renderer.render_visual(sample_expression)

            assert isinstance(rendered, RenderedVisual)
            assert rendered.data is not None

    @pytest.mark.asyncio
    async def test_quality_presets(self):
        """Test different quality presets."""
        presets = ['low', 'balanced', 'high']

        expression = VisualExpression(
            style='organic',
            intensity=0.7,
            colors=['#228B22', '#32CD32'],
            composition={'shapes': ['circle', 'wave'], 'patterns': ['flowing']},
            confidence=0.8
        )

        for preset in presets:
            renderer = ArtisticVisualRenderer()
            config = {
                'canvas_width': 400,
                'canvas_height': 300,
                'quality_preset': preset
            }
            await renderer.initialize(config)

            try:
                rendered = await renderer.render_visual(expression)

                assert isinstance(rendered, RenderedVisual)
                assert rendered.data is not None

                # Higher quality should potentially take longer or produce more detailed output
                # This is hard to test directly, but we can ensure it completes

            finally:
                await renderer.cleanup()

    @pytest.mark.asyncio
    async def test_concurrent_rendering(self, visual_renderer):
        """Test concurrent rendering processing."""
        expressions = [
            VisualExpression('organic', 0.6, ['#FF0000', '#00FF00'], {'shapes': ['circle']}, 0.8),
            VisualExpression('geometric', 0.7, ['#0000FF', '#FFFF00'], {'shapes': ['square']}, 0.9),
            VisualExpression('abstract', 0.8, ['#FF00FF', '#00FFFF'], {'shapes': ['triangle']}, 0.7),
        ]

        # Render concurrently
        tasks = [visual_renderer.render_visual(expr) for expr in expressions]
        renderings = await asyncio.gather(*tasks)

        assert len(renderings) == 3
        for rendered in renderings:
            assert isinstance(rendered, RenderedVisual)
            assert rendered.data is not None

    @pytest.mark.asyncio
    async def test_memory_management(self, visual_renderer, sample_expression):
        """Test memory management during rendering."""
        # Render many images
        renderings = []
        for i in range(20):
            # Vary expression slightly
            varied_expression = sample_expression._replace(
                intensity=sample_expression.intensity + 0.01 * i
            )
            rendered = await visual_renderer.render_visual(varied_expression)
            renderings.append(rendered)

        assert len(renderings) == 20
        for rendered in renderings:
            assert isinstance(rendered, RenderedVisual)
            assert rendered.data is not None

    @pytest.mark.asyncio
    async def test_error_handling(self, visual_renderer):
        """Test error handling in rendering."""
        # Test with invalid expression
        invalid_expression = None
        with pytest.raises(AttributeError):
            await visual_renderer.render_visual(invalid_expression)

        # Test with empty colors
        empty_expression = VisualExpression(
            style='minimalist',
            intensity=0.5,
            colors=[],  # Empty colors
            composition={'shapes': ['circle']},
            confidence=0.5
        )

        rendered = await visual_renderer.render_visual(empty_expression)
        assert isinstance(rendered, RenderedVisual)  # Should handle gracefully

        # Test with invalid format
        with pytest.raises(ValueError):
            await visual_renderer.render_visual(sample_expression, format_type='invalid')

    @pytest.mark.asyncio
    async def test_metadata_inclusion(self, visual_renderer, sample_expression):
        """Test metadata inclusion in rendered output."""
        rendered = await visual_renderer.render_visual(sample_expression)

        assert 'quality' in rendered.metadata
        assert 'render_time' in rendered.metadata
        assert 'style' in rendered.metadata
        assert rendered.metadata['style'] == sample_expression.style

    @pytest.mark.asyncio
    async def test_edge_cases(self, visual_renderer):
        """Test edge cases in rendering."""
        # Test with minimal expression
        minimal_expression = VisualExpression(
            style='minimalist',
            intensity=0.0,
            colors=['#FFFFFF'],
            composition={'shapes': []},
            confidence=0.0
        )

        rendered = await visual_renderer.render_visual(minimal_expression)
        assert isinstance(rendered, RenderedVisual)

        # Test with maximum complexity
        complex_expression = VisualExpression(
            style='chaotic',
            intensity=1.0,
            colors=['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF'],
            composition={
                'shapes': ['circle', 'square', 'triangle', 'wave', 'spiral', 'star'],
                'patterns': ['flowing', 'chaotic', 'structured', 'random'],
                'layout': 'dense'
            },
            confidence=1.0
        )

        rendered = await visual_renderer.render_visual(complex_expression)
        assert isinstance(rendered, RenderedVisual)

    @pytest.mark.asyncio
    async def test_rendering_performance(self, visual_renderer, sample_expression):
        """Test rendering performance characteristics."""
        import time

        start_time = time.time()
        rendered = await visual_renderer.render_visual(sample_expression)
        end_time = time.time()

        render_time = end_time - start_time
        assert render_time < 10.0  # Should complete within reasonable time

        # Check that metadata includes render time
        assert 'render_time' in rendered.metadata
        assert rendered.metadata['render_time'] > 0

    @pytest.mark.asyncio
    async def test_format_conversion(self, visual_renderer, sample_expression):
        """Test format conversion capabilities."""
        # Render to numpy first
        numpy_rendered = await visual_renderer.render_visual(sample_expression, format_type='numpy')

        # Convert to other formats
        image_rendered = await visual_renderer.render_visual(sample_expression, format_type='image')
        base64_rendered = await visual_renderer.render_visual(sample_expression, format_type='base64')

        # All should be valid
        assert numpy_rendered.format == 'numpy'
        assert image_rendered.format == 'image'
        assert base64_rendered.format == 'base64'

        # Check data types
        assert isinstance(numpy_rendered.data, np.ndarray)
        assert isinstance(image_rendered.data, bytes)
        assert isinstance(base64_rendered.data, str)

    @pytest.mark.asyncio
    async def test_composition_complexity(self, visual_renderer):
        """Test rendering with different composition complexities."""
        complexities = [
            {'shapes': ['circle'], 'patterns': []},  # Simple
            {'shapes': ['circle', 'square'], 'patterns': ['structured']},  # Medium
            {'shapes': ['circle', 'square', 'triangle', 'wave'], 'patterns': ['flowing', 'chaotic']}  # Complex
        ]

        for composition in complexities:
            expression = VisualExpression(
                style='organic',
                intensity=0.7,
                colors=['#228B22', '#32CD32'],
                composition=composition,
                confidence=0.8
            )

            rendered = await visual_renderer.render_visual(expression)

            assert isinstance(rendered, RenderedVisual)
            assert rendered.data is not None
