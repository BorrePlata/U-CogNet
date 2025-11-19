# Core module for U-CogNet

# Imports seguros que no tienen dependencias externas
from .types import (
    Frame, Detection, Event, Metrics, SystemState, TopologyConfig,
    TrainingData, ModelConfig, OptimizationResult, MycelialNode, MycelialEdge,
    CognitiveTrace, ExperimentConfig, ExperimentResult
)
from .interfaces import (
    CognitiveModule, MemorySystem, EvaluatorInterface, TrainerInterface,
    TDAManagerInterface, TraceManager, InputHandler, VisionDetector, SemanticFeedbackGenerator
)
from .utils import (
    load_config, setup_logging, calculate_metrics, save_checkpoint,
    load_checkpoint, build_event, build_system_state
)

# Imports de clases principales - con manejo de errores
try:
    from .cognitive_core import CognitiveCore
except ImportError:
    CognitiveCore = None

try:
    from .tda_manager import TDAManager
except ImportError:
    TDAManager = None

try:
    from .evaluator import Evaluator
except ImportError:
    Evaluator = None

try:
    from .trainer_loop import TrainerLoop
except ImportError:
    TrainerLoop = None

try:
    from .mycelial_optimizer import MycelialOptimizer
except ImportError:
    MycelialOptimizer = None