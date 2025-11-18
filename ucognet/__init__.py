"""
U-CogNet: Unified Cognitive Network
Módulo principal del sistema de inteligencia modular adaptativa.

Este paquete implementa el marco ADN del Agente con énfasis en:
- Modularidad contractual
- Aprendizaje continuo
- Meta-cognición
- Topología dinámica adaptativa
- Ética funcional

Versión: 0.1.0 (Fase inicial de integración)
"""

__version__ = "0.1.0"
__author__ = "U-CogNet Team"
__description__ = "Sistema de inteligencia artificial modular y adaptativa"

# Imports principales para facilitar el uso
from .common import types, utils, logging
from .input_handler import InputHandler
from .vision_detector import VisionDetector
from .cognitive_core import CognitiveCore
from .semantic_feedback import SemanticFeedback
from .evaluator import Evaluator
from .trainer_loop import TrainerLoop
from .tda_manager import TDAManager
from .visual_interface import VisualInterface
from .audio import AudioCognitiveProcessor