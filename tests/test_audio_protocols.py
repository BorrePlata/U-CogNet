"""Simplified audio protocols tests for U-CogNet."""

import pytest
import inspect
import sys
sys.path.insert(0, 'src')
from ucognet.modules.audio.audio_protocols import (
    AudioInputProtocol, AudioFeatureExtractionProtocol, AudioPerceptionProtocol,
    VisualExpressionProtocol, VisualRenderingProtocol, AudioVisualSynthesisProtocol,
    AudioVisualEvaluationProtocol
)

class TestAudioProtocols:
    """Test audio-visual protocol interfaces."""
    
    def test_audio_input_protocol_structure(self):
        """Test AudioInputProtocol has required methods."""
        assert hasattr(AudioInputProtocol, 'sample_rate')
        assert hasattr(AudioInputProtocol, 'channels')
        assert hasattr(AudioInputProtocol, 'capture_audio')
        assert hasattr(AudioInputProtocol, 'initialize')
        assert hasattr(AudioInputProtocol, 'cleanup')
        
    def test_audio_feature_extraction_protocol_structure(self):
        """Test AudioFeatureExtractionProtocol has required methods."""
        assert hasattr(AudioFeatureExtractionProtocol, 'extract_features')
        assert hasattr(AudioFeatureExtractionProtocol, 'initialize')
        assert hasattr(AudioFeatureExtractionProtocol, 'cleanup')
        
        # Check that extract_features is a coroutine function
        assert inspect.iscoroutinefunction(AudioFeatureExtractionProtocol.extract_features)
        
    def test_audio_perception_protocol_structure(self):
        """Test AudioPerceptionProtocol has required methods."""
        assert hasattr(AudioPerceptionProtocol, 'perceive_audio')
        assert hasattr(AudioPerceptionProtocol, 'initialize')
        assert hasattr(AudioPerceptionProtocol, 'cleanup')
        
        # Check that perceive_audio is a coroutine function
        assert inspect.iscoroutinefunction(AudioPerceptionProtocol.perceive_audio)
        
    def test_visual_expression_protocol_structure(self):
        """Test VisualExpressionProtocol has required methods."""
        assert hasattr(VisualExpressionProtocol, 'express_visually')
        assert hasattr(VisualExpressionProtocol, 'initialize')
        assert hasattr(VisualExpressionProtocol, 'cleanup')
        
        # Check that express_visually is a coroutine function
        assert inspect.iscoroutinefunction(VisualExpressionProtocol.express_visually)
        
    def test_visual_rendering_protocol_structure(self):
        """Test VisualRenderingProtocol has required methods."""
        assert hasattr(VisualRenderingProtocol, 'render_visual')
        assert hasattr(VisualRenderingProtocol, 'initialize')
        assert hasattr(VisualRenderingProtocol, 'cleanup')
        
        # Check that render_visual is a coroutine function
        assert inspect.iscoroutinefunction(VisualRenderingProtocol.render_visual)
        
    def test_audio_visual_synthesis_protocol_structure(self):
        """Test AudioVisualSynthesisProtocol has required methods."""
        assert hasattr(AudioVisualSynthesisProtocol, 'synthesize_audio_visual')
        assert hasattr(AudioVisualSynthesisProtocol, 'initialize')
        assert hasattr(AudioVisualSynthesisProtocol, 'cleanup')
        
        # Check that synthesize_audio_visual is a coroutine function
        assert inspect.iscoroutinefunction(AudioVisualSynthesisProtocol.synthesize_audio_visual)
        
    def test_audio_visual_evaluation_protocol_structure(self):
        """Test AudioVisualEvaluationProtocol has required methods."""
        assert hasattr(AudioVisualEvaluationProtocol, 'evaluate_synthesis')
        assert hasattr(AudioVisualEvaluationProtocol, 'initialize')
        assert hasattr(AudioVisualEvaluationProtocol, 'cleanup')
        
        # Check that evaluate_synthesis is a coroutine function
        assert inspect.iscoroutinefunction(AudioVisualEvaluationProtocol.evaluate_synthesis)
        
    def test_protocol_inheritance(self):
        """Test that protocols are properly defined as runtime checkable."""
        # These should be protocol classes
        assert hasattr(AudioInputProtocol, '__protocol_attrs__') or hasattr(AudioInputProtocol, '__annotations__')
        assert hasattr(AudioFeatureExtractionProtocol, '__protocol_attrs__') or hasattr(AudioFeatureExtractionProtocol, '__annotations__')
