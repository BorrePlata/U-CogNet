
"""
U-CogNet: Sistema Cognitivo Universal
Versión consolidada con arquitectura modular
"""

from .common.types import *
from .common.logging import logger
from .input_handler import InputHandler
from .vision_detector import VisionDetector
from .cognitive_core import CognitiveCore
from .semantic_feedback import SemanticFeedback
from .evaluator import Evaluator
from .trainer_loop import TrainerLoop
from .tda_manager import TDAManager
from .visual_interface import VisualInterface

# Nuevos módulos consolidados
try:
    from .audio.feature_extractor import AudioFeatureExtractor
    from .audio.video_audio_extractor import VideoAudioExtractor
    from .security.perception_sanitizer import PerceptionSanitizer
    from .security.universal_ethics_engine import UniversalEthicsEngine
    from .security.cognitive_security_architecture import CognitiveSecurityArchitecture
except ImportError:
    pass

__version__ = "2.0.0"
__author__ = "AGI U-CogNet"
