# U-CogNet Visual Rendering Module
# Concrete Visual Representation of Artistic Expressions

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import cv2
import logging
from PIL import Image, ImageDraw, ImageFont
import colorsys

from .audio_types import VisualExpression, RenderedVisual
from .audio_protocols import VisualRenderingProtocol

class ArtisticVisualRenderer(VisualRenderingProtocol):
    """Renders artistic visual expressions into concrete visual representations."""

    def __init__(self):
        self._canvas_width = 800
        self._canvas_height = 600
        self._font_cache = {}

        # Rendering quality settings
        self._antialiasing = True
        self._color_depth = 32
        self._frame_rate = 30

    async def render_visual(self, expression: VisualExpression,
                          format_type: str = 'image') -> RenderedVisual:
        """Render visual expression into concrete representation."""
        # Create base canvas
        canvas = self._create_canvas(expression)

        # Render background
        canvas = await self._render_background(canvas, expression)

        # Render shapes
        canvas = await self._render_shapes(canvas, expression)

        # Render symbols
        canvas = await self._render_symbols(canvas, expression)

        # Apply dynamics and effects
        canvas = await self._apply_dynamics(canvas, expression)

        # Convert to final format
        rendered_data = await self._convert_to_format(canvas, format_type)

        # Create rendered visual object
        rendered = RenderedVisual(
            timestamp=expression.timestamp,
            expression=expression,
            data=rendered_data,
            format=format_type,
            dimensions=(self._canvas_width, self._canvas_height),
            metadata={
                'render_method': 'artistic_synthesis',
                'quality_settings': self._get_quality_settings(),
                'render_time': datetime.now().isoformat()
            }
        )

        return rendered

    def _create_canvas(self, expression: VisualExpression) -> Image.Image:
        """Create base canvas for rendering."""
        # Use dimensions from expression composition
        width, height = expression.composition.get('dimensions', (self._canvas_width, self._canvas_height))

        # Create RGBA canvas
        canvas = Image.new('RGBA', (width, height), (0, 0, 0, 0))

        return canvas

    async def _render_background(self, canvas: Image.Image,
                               expression: VisualExpression) -> Image.Image:
        """Render background elements."""
        draw = ImageDraw.Draw(canvas, 'RGBA')

        # Get background color from composition
        bg_color_hsv = expression.composition.get('background_color', (0.5, 0.1, 0.2))
        bg_color_rgb = self._hsv_to_rgb(bg_color_hsv)
        bg_color_rgba = (*bg_color_rgb, 255)

        # Fill background
        draw.rectangle([0, 0, canvas.width, canvas.height], fill=bg_color_rgba)

        # Add background texture/pattern based on style
        style = expression.style
        if style == 'organic':
            canvas = await self._add_organic_texture(canvas, expression)
        elif style == 'dynamic':
            canvas = await self._add_dynamic_texture(canvas, expression)
        elif style == 'geometric':
            canvas = await self._add_geometric_pattern(canvas, expression)

        return canvas

    async def _render_shapes(self, canvas: Image.Image,
                           expression: VisualExpression) -> Image.Image:
        """Render geometric shapes."""
        draw = ImageDraw.Draw(canvas, 'RGBA')

        shapes = expression.shapes
        colors = expression.colors

        for shape in shapes:
            shape_type = shape['type']
            position = shape['position']
            size = shape['size']
            rotation = shape.get('rotation', 0)
            opacity = shape.get('opacity', 1.0)

            # Convert normalized position to pixel coordinates
            x = int(position[0] * canvas.width)
            y = int(position[1] * canvas.height)

            # Select color from palette
            color_idx = np.random.randint(len(colors))
            color_hsv = colors[color_idx]
            color_rgb = self._hsv_to_rgb(color_hsv)
            color_rgba = (*color_rgb, int(opacity * 255))

            # Render shape based on type
            if shape_type == 'circle':
                radius = shape.get('radius', size * 20)
                bbox = [x - radius, y - radius, x + radius, y + radius]
                draw.ellipse(bbox, fill=color_rgba)

            elif shape_type == 'wave':
                await self._render_wave(draw, x, y, shape, color_rgba, canvas.width)

            elif shape_type == 'burst':
                await self._render_burst(draw, x, y, shape, color_rgba)

            elif shape_type == 'flow':
                await self._render_flow(draw, x, y, shape, color_rgba, canvas.width, canvas.height)

            elif shape_type == 'spark':
                await self._render_spark(draw, x, y, shape, color_rgba)

            elif shape_type == 'radial':
                await self._render_radial(draw, x, y, shape, color_rgba)

        return canvas

    async def _render_symbols(self, canvas: Image.Image,
                            expression: VisualExpression) -> Image.Image:
        """Render symbolic elements."""
        draw = ImageDraw.Draw(canvas, 'RGBA')

        symbols = expression.symbols

        for symbol in symbols:
            # Get font for symbol
            font = self._get_font(symbol.size)

            # Convert position to pixel coordinates
            x = int(symbol.position[0] * canvas.width)
            y = int(symbol.position[1] * canvas.height)

            # Convert color
            color_rgb = self._hsv_to_rgb(symbol.color)
            color_rgba = (*color_rgb, int(symbol.opacity * 255))

            # Draw symbol
            draw.text((x, y), symbol.symbol, fill=color_rgba, font=font, anchor='mm')

            # Add animation effects if specified
            if symbol.animation:
                canvas = await self._apply_symbol_animation(canvas, symbol, x, y)

        return canvas

    async def _apply_dynamics(self, canvas: Image.Image,
                            expression: VisualExpression) -> Image.Image:
        """Apply dynamic effects and animations."""
        dynamics = expression.dynamics

        # Apply motion blur if high animation speed
        animation_speed = dynamics.get('animation_speed', 0)
        if animation_speed > 1.0:
            canvas = await self._apply_motion_blur(canvas, animation_speed)

        # Apply glow effects for intense expressions
        intensity = expression.intensity
        if intensity > 0.7:
            canvas = await self._apply_glow_effect(canvas, intensity)

        # Apply interaction effects
        interaction_effects = dynamics.get('interaction_effects', [])
        for effect in interaction_effects:
            canvas = await self._apply_interaction_effect(canvas, effect, expression)

        return canvas

    async def _convert_to_format(self, canvas: Image.Image,
                               format_type: str) -> bytes:
        """Convert canvas to specified format."""
        if format_type == 'image':
            # Convert to PNG bytes
            from io import BytesIO
            buffer = BytesIO()
            canvas.save(buffer, format='PNG')
            return buffer.getvalue()

        elif format_type == 'numpy':
            # Convert to numpy array
            return np.array(canvas)

        elif format_type == 'base64':
            # Convert to base64 string
            import base64
            from io import BytesIO
            buffer = BytesIO()
            canvas.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

        else:
            # Default to PNG bytes
            from io import BytesIO
            buffer = BytesIO()
            canvas.save(buffer, format='PNG')
            return buffer.getvalue()

    async def _add_organic_texture(self, canvas: Image.Image,
                                 expression: VisualExpression) -> Image.Image:
        """Add organic texture to background."""
        draw = ImageDraw.Draw(canvas, 'RGBA')

        # Add subtle organic patterns
        intensity = expression.intensity
        num_blobs = int(intensity * 20) + 5

        for _ in range(num_blobs):
            # Random organic blob
            x = np.random.randint(0, canvas.width)
            y = np.random.randint(0, canvas.height)
            radius = np.random.uniform(10, 50)

            # Soft color variation
            alpha = np.random.uniform(10, 30)
            color = (200, 200, 200, int(alpha))

            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)

        return canvas

    async def _add_dynamic_texture(self, canvas: Image.Image,
                                 expression: VisualExpression) -> Image.Image:
        """Add dynamic texture with energy lines."""
        draw = ImageDraw.Draw(canvas, 'RGBA')

        arousal = expression.perception.arousal_level
        num_lines = int(arousal * 15) + 5

        for _ in range(num_lines):
            # Energy lines
            start_x = np.random.randint(0, canvas.width)
            start_y = np.random.randint(0, canvas.height)
            end_x = start_x + np.random.uniform(-50, 50)
            end_y = start_y + np.random.uniform(-50, 50)

            alpha = np.random.uniform(20, 60)
            color = (255, 255, 255, int(alpha))

            draw.line([start_x, start_y, end_x, end_y], fill=color, width=1)

        return canvas

    async def _add_geometric_pattern(self, canvas: Image.Image,
                                   expression: VisualExpression) -> Image.Image:
        """Add geometric pattern to background."""
        draw = ImageDraw.Draw(canvas, 'RGBA')

        # Grid pattern
        spacing = 40
        alpha = 15

        for x in range(0, canvas.width, spacing):
            color = (255, 255, 255, alpha)
            draw.line([x, 0, x, canvas.height], fill=color, width=1)

        for y in range(0, canvas.height, spacing):
            color = (255, 255, 255, alpha)
            draw.line([0, y, canvas.width, y], fill=color, width=1)

        return canvas

    async def _render_wave(self, draw: ImageDraw.ImageDraw, x: int, y: int,
                          shape: Dict[str, Any], color: Tuple[int, int, int, int],
                          canvas_width: int) -> None:
        """Render wave shape."""
        amplitude = shape.get('amplitude', 20)
        frequency = shape.get('frequency', 2)
        length = 100

        # Draw sinusoidal wave
        points = []
        for i in range(length):
            wave_x = x + i - length // 2
            if 0 <= wave_x < canvas_width:
                wave_y = y + amplitude * np.sin(2 * np.pi * frequency * i / length)
                points.extend([wave_x, wave_y])

        if len(points) >= 4:
            draw.line(points, fill=color, width=2)

    async def _render_burst(self, draw: ImageDraw.ImageDraw, x: int, y: int,
                           shape: Dict[str, Any], color: Tuple[int, int, int, int]) -> None:
        """Render burst shape."""
        num_rays = shape.get('num_rays', 8)
        ray_length = shape.get('ray_length', 30)

        # Draw radial burst
        for i in range(num_rays):
            angle = 2 * np.pi * i / num_rays
            end_x = x + ray_length * np.cos(angle)
            end_y = y + ray_length * np.sin(angle)

            draw.line([x, y, end_x, end_y], fill=color, width=3)

    async def _render_flow(self, draw: ImageDraw.ImageDraw, x: int, y: int,
                          shape: Dict[str, Any], color: Tuple[int, int, int, int],
                          canvas_width: int, canvas_height: int) -> None:
        """Render flow shape."""
        flow_direction = shape.get('flow_direction', 0)
        flow_speed = shape.get('flow_speed', 1)
        length = 80

        # Draw curved flow line
        points = []
        for i in range(length):
            offset = i * 2
            curve_x = x + offset * np.cos(flow_direction)
            curve_y = y + offset * np.sin(flow_direction) + 10 * np.sin(i * 0.2)

            if 0 <= curve_x < canvas_width and 0 <= curve_y < canvas_height:
                points.extend([curve_x, curve_y])

        if len(points) >= 4:
            draw.line(points, fill=color, width=4)

    async def _render_spark(self, draw: ImageDraw.ImageDraw, x: int, y: int,
                           shape: Dict[str, Any], color: Tuple[int, int, int, int]) -> None:
        """Render spark shape."""
        sparkle_intensity = shape.get('sparkle_intensity', 1)
        trail_length = shape.get('trail_length', 10)

        # Draw spark with trail
        for i in range(trail_length):
            alpha = int(color[3] * (1 - i / trail_length))
            spark_color = (*color[:3], alpha)
            radius = 2 + i * 0.5

            bbox = [x - radius, y - radius, x + radius, y + radius]
            draw.ellipse(bbox, fill=spark_color)

    async def _render_radial(self, draw: ImageDraw.ImageDraw, x: int, y: int,
                            shape: Dict[str, Any], color: Tuple[int, int, int, int]) -> None:
        """Render radial shape."""
        # Concentric circles
        max_radius = shape.get('size', 1) * 30
        num_circles = 5

        for i in range(num_circles):
            radius = max_radius * (i + 1) / num_circles
            alpha = int(color[3] * (1 - i / num_circles))
            circle_color = (*color[:3], alpha)

            bbox = [x - radius, y - radius, x + radius, y + radius]
            draw.ellipse(bbox, outline=circle_color, width=2)

    async def _apply_symbol_animation(self, canvas: Image.Image,
                                    symbol: VisualSymbol, x: int, y: int) -> Image.Image:
        """Apply animation effects to symbols."""
        animation = symbol.animation

        if animation == 'glow':
            # Add glow effect
            draw = ImageDraw.Draw(canvas, 'RGBA')
            glow_color = (*symbol.color[:2], min(1.0, symbol.color[2] + 0.3), 0.3)
            glow_rgb = self._hsv_to_rgb(glow_color)

            for radius in range(5, 15, 2):
                alpha = int(50 * (1 - radius / 15))
                glow_rgba = (*glow_rgb, alpha)
                bbox = [x - radius, y - radius, x + radius, y + radius]
                draw.ellipse(bbox, fill=glow_rgba)

        elif animation == 'sparkle':
            # Add sparkle particles
            draw = ImageDraw.Draw(canvas, 'RGBA')
            for _ in range(5):
                sparkle_x = x + np.random.uniform(-10, 10)
                sparkle_y = y + np.random.uniform(-10, 10)
                sparkle_color = (*self._hsv_to_rgb(symbol.color), 150)

                draw.ellipse([sparkle_x-1, sparkle_y-1, sparkle_x+1, sparkle_y+1],
                           fill=sparkle_color)

        return canvas

    async def _apply_motion_blur(self, canvas: Image.Image, speed: float) -> Image.Image:
        """Apply motion blur effect."""
        # Convert to numpy array for processing
        img_array = np.array(canvas)

        # Simple motion blur using OpenCV
        kernel_size = min(5, max(1, int(speed / 2)))
        if kernel_size > 1:
            # Horizontal blur for motion effect
            kernel = np.ones((1, kernel_size), np.float32) / kernel_size
            img_array = cv2.filter2D(img_array, -1, kernel)

        return Image.fromarray(img_array)

    async def _apply_glow_effect(self, canvas: Image.Image, intensity: float) -> Image.Image:
        """Apply glow effect to entire canvas."""
        # Create glow layer
        glow_layer = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(glow_layer, 'RGBA')

        # Add soft glow circles
        num_glows = int(intensity * 10) + 3
        for _ in range(num_glows):
            x = np.random.randint(0, canvas.width)
            y = np.random.randint(0, canvas.height)
            radius = np.random.uniform(20, 80)
            alpha = np.random.uniform(10, 30)

            glow_color = (255, 255, 255, int(alpha))
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=glow_color)

        # Composite with original
        canvas = Image.alpha_composite(canvas, glow_layer)

        return canvas

    async def _apply_interaction_effect(self, canvas: Image.Image,
                                      effect: Dict[str, Any],
                                      expression: VisualExpression) -> Image.Image:
        """Apply interaction effects between elements."""
        effect_type = effect['type']
        strength = effect.get('strength', 0.5)

        if effect_type == 'attraction':
            # Elements pull toward each other
            pass  # Would need element positions to implement

        elif effect_type == 'merge':
            # Elements blend together
            pass  # Complex effect requiring element tracking

        # For now, just add subtle connecting lines
        draw = ImageDraw.Draw(canvas, 'RGBA')

        num_connections = int(strength * 5)
        for _ in range(num_connections):
            x1 = np.random.randint(0, canvas.width)
            y1 = np.random.randint(0, canvas.height)
            x2 = np.random.randint(0, canvas.width)
            y2 = np.random.randint(0, canvas.height)

            alpha = int(strength * 50)
            color = (255, 255, 255, alpha)
            draw.line([x1, y1, x2, y2], fill=color, width=1)

        return canvas

    def _get_font(self, size: float) -> ImageFont.FreeTypeFont:
        """Get font for symbol rendering."""
        size_key = int(size)
        if size_key not in self._font_cache:
            try:
                # Try to use a nice Unicode font
                self._font_cache[size_key] = ImageFont.truetype("DejaVuSans.ttf", size_key)
            except:
                # Fallback to default
                self._font_cache[size_key] = ImageFont.load_default()

        return self._font_cache[size_key]

    def _hsv_to_rgb(self, hsv: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert HSV to RGB."""
        r, g, b = colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])
        return (int(r * 255), int(g * 255), int(b * 255))

    def _get_quality_settings(self) -> Dict[str, Any]:
        """Get current rendering quality settings."""
        return {
            'antialiasing': self._antialiasing,
            'color_depth': self._color_depth,
            'frame_rate': self._frame_rate,
            'canvas_size': (self._canvas_width, self._canvas_height)
        }

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the rendering module."""
        # Update canvas size if specified
        if 'canvas_width' in config:
            self._canvas_width = config['canvas_width']
        if 'canvas_height' in config:
            self._canvas_height = config['canvas_height']

        # Update quality settings
        if 'antialiasing' in config:
            self._antialiasing = config['antialiasing']
        if 'color_depth' in config:
            self._color_depth = config['color_depth']

        logging.info(f"Initialized ArtisticVisualRenderer with canvas {self._canvas_width}x{self._canvas_height}")

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._font_cache.clear()
