# U-CogNet API Documentation
## Complete Reference for Cognitive Architecture Components

**Version:** 0.1.0 | **Date:** November 15, 2025  
**Framework:** Python 3.11+ | **License:** MIT  

---

## Table of Contents

1. [Core Types and Interfaces](#1-core-types-and-interfaces)
2. [Perception Module](#2-perception-module)
3. [Cognitive Core](#3-cognitive-core)
4. [Semantic Feedback](#4-semantic-feedback)
5. [Evaluator](#5-evaluator)
6. [TDA Manager](#6-tda-manager)
7. [Runtime Engine](#7-runtime-engine)
8. [Configuration](#8-configuration)
9. [Error Handling](#9-error-handling)

---

## 1. Core Types and Interfaces

### 1.1 Base Types

#### `DetectionResult`
Represents the result of object detection operations.

```python
@dataclass
class DetectionResult:
    """Result of object detection with confidence and metadata."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### `CognitiveEvent`
Represents events in the cognitive processing pipeline.

```python
@dataclass
class CognitiveEvent:
    """Event in the cognitive processing stream."""
    event_type: str
    timestamp: datetime
    data: dict[str, Any]
    source: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### `PerformanceMetrics`
Comprehensive performance evaluation metrics.

```python
@dataclass
class PerformanceMetrics:
    """Comprehensive performance evaluation."""
    precision: float
    recall: float
    f1_score: float
    mcc: float  # Matthews Correlation Coefficient
    map_score: float  # Mean Average Precision
    temporal_consistency: float
    processing_time: float
    resource_usage: dict[str, float]
    timestamp: datetime
```

### 1.2 Protocol Interfaces

#### `PerceptionProtocol`
Interface for perception modules.

```python
class PerceptionProtocol(Protocol):
    """Protocol for perception modules."""

    async def process_frame(self, frame: np.ndarray) -> list[DetectionResult]:
        """Process a single frame and return detections."""
        ...

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the perception module."""
        ...

    async def cleanup(self) -> None:
        """Clean up resources."""
        ...

    @property
    def capabilities(self) -> list[str]:
        """Return list of supported capabilities."""
        ...
```

#### `CognitiveCoreProtocol`
Interface for cognitive core implementations.

```python
class CognitiveCoreProtocol(Protocol):
    """Protocol for cognitive core modules."""

    async def process_events(self, events: list[CognitiveEvent]) -> list[CognitiveEvent]:
        """Process incoming cognitive events."""
        ...

    async def get_context(self, query: str) -> dict[str, Any]:
        """Retrieve contextual information."""
        ...

    async def store_memory(self, event: CognitiveEvent) -> None:
        """Store event in cognitive memory."""
        ...

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the cognitive core."""
        ...

    async def cleanup(self) -> None:
        """Clean up resources."""
        ...
```

#### `SemanticFeedbackProtocol`
Interface for semantic analysis modules.

```python
class SemanticFeedbackProtocol(Protocol):
    """Protocol for semantic feedback modules."""

    async def analyze_scene(self, detections: list[DetectionResult],
                          context: dict[str, Any]) -> str:
        """Analyze scene and provide semantic interpretation."""
        ...

    async def assess_threat(self, detections: list[DetectionResult]) -> dict[str, Any]:
        """Assess threat level from detections."""
        ...

    async def generate_explanation(self, analysis: dict[str, Any]) -> str:
        """Generate natural language explanation."""
        ...

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the semantic module."""
        ...

    async def cleanup(self) -> None:
        """Clean up resources."""
        ...
```

#### `EvaluatorProtocol`
Interface for evaluation modules.

```python
class EvaluatorProtocol(Protocol):
    """Protocol for evaluation modules."""

    async def evaluate_performance(self, detections: list[DetectionResult],
                                 ground_truth: list[DetectionResult] = None) -> PerformanceMetrics:
        """Evaluate system performance."""
        ...

    async def update_baseline(self, metrics: PerformanceMetrics) -> None:
        """Update performance baseline."""
        ...

    async def get_trends(self, window: int = 100) -> dict[str, list[float]]:
        """Get performance trends over time."""
        ...

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the evaluator."""
        ...

    async def cleanup(self) -> None:
        """Clean up resources."""
        ...
```

#### `TDAManagerProtocol`
Interface for topology adaptation managers.

```python
class TDAManagerProtocol(Protocol):
    """Protocol for TDA managers."""

    async def assess_performance(self, metrics: PerformanceMetrics) -> dict[str, Any]:
        """Assess current system performance."""
        ...

    async def optimize_resources(self, assessment: dict[str, Any]) -> dict[str, Any]:
        """Optimize resource allocation."""
        ...

    async def adapt_topology(self, optimization: dict[str, Any]) -> None:
        """Adapt system topology based on optimization."""
        ...

    async def get_status(self) -> dict[str, Any]:
        """Get current TDA status."""
        ...

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the TDA manager."""
        ...

    async def cleanup(self) -> None:
        """Clean up resources."""
        ...
```

---

## 2. Perception Module

### 2.1 YOLOv8Detector

Enhanced YOLOv8 implementation with weapon detection and MediaPipe integration.

#### Constructor
```python
def __init__(self, model_path: str = "yolov8n.pt",
             conf_threshold: float = 0.5,
             weapon_classes: list[str] = None,
             enable_mediapipe: bool = True)
```

#### Key Methods

##### `process_frame(frame: np.ndarray) -> list[DetectionResult]`
Process a single video frame.

**Parameters:**
- `frame`: Input frame as numpy array (H, W, C)

**Returns:**
- List of DetectionResult objects

**Example:**
```python
detector = YOLOv8Detector()
results = await detector.process_frame(frame)
for result in results:
    print(f"Detected {result.class_name} with {result.confidence:.2f} confidence")
```

##### `detect_weapons(detections: list[DetectionResult]) -> list[DetectionResult]`
Enhanced weapon detection with proximity logic.

**Parameters:**
- `detections`: Raw detection results

**Returns:**
- Filtered detections with weapon-person associations

##### `get_pose_landmarks(frame: np.ndarray) -> dict[str, Any]`
Extract pose landmarks using MediaPipe.

**Parameters:**
- `frame`: Input frame

**Returns:**
- Dictionary containing pose, hands, and face landmarks

### 2.2 Configuration Options

```python
config = {
    "model_path": "yolov8n.pt",
    "conf_threshold": 0.5,
    "weapon_classes": ["knife", "scissors", "baseball bat", "bottle", "fork"],
    "enable_mediapipe": True,
    "pose_confidence": 0.5,
    "hand_confidence": 0.5,
    "face_confidence": 0.5
}
```

---

## 3. Cognitive Core

### 3.1 CognitiveCoreImpl

Dual-memory system implementation with episodic and working memory.

#### Constructor
```python
def __init__(self, max_episodic_memory: int = 1000,
             max_working_memory: int = 100,
             context_window: int = 50)
```

#### Key Methods

##### `process_events(events: list[CognitiveEvent]) -> list[CognitiveEvent]`
Process and integrate cognitive events.

**Parameters:**
- `events`: List of incoming cognitive events

**Returns:**
- Processed events with context enrichment

##### `store_memory(event: CognitiveEvent) -> None`
Store event in appropriate memory system.

**Parameters:**
- `event`: Cognitive event to store

##### `retrieve_context(query: str, k: int = 5) -> list[CognitiveEvent]`
Retrieve relevant context from memory.

**Parameters:**
- `query`: Context query string
- `k`: Number of results to return

**Returns:**
- List of relevant cognitive events

##### `consolidate_memory() -> None`
Consolidate working memory into episodic memory.

### 3.2 Memory Architecture

```
Working Memory (100 events)
├── Recent Events Buffer
├── Active Context
└── Pattern Recognition

Episodic Memory (1000 events)
├── Temporal Indexing
├── Semantic Clustering
└── Pattern Consolidation
```

---

## 4. Semantic Feedback

### 4.1 RuleBasedSemanticFeedback

Rule-based semantic analysis with tactical scene interpretation.

#### Constructor
```python
def __init__(self, rules_path: str = None,
             threat_threshold: float = 0.7,
             context_window: int = 10)
```

#### Key Methods

##### `analyze_scene(detections: list[DetectionResult], context: dict) -> str`
Provide semantic interpretation of current scene.

**Parameters:**
- `detections`: Current detection results
- `context`: Cognitive context dictionary

**Returns:**
- Natural language scene description

##### `assess_threat(detections: list[DetectionResult]) -> dict[str, Any]`
Assess threat level and provide detailed analysis.

**Parameters:**
- `detections`: Detection results

**Returns:**
- Dictionary with threat assessment details

##### `get_scene_type(detections: list[DetectionResult]) -> str`
Classify scene type (convoy, crowd, armed_person, etc.).

**Parameters:**
- `detections`: Detection results

**Returns:**
- Scene classification string

### 4.2 Rule Engine

#### Predefined Rules
- **Convoy Detection**: Multiple vehicles in formation
- **Crowd Analysis**: High person density with movement patterns
- **Armed Person Detection**: Person-weapon proximity analysis
- **Threat Assessment**: Risk level calculation based on context

#### Custom Rule Format
```python
rule = {
    "name": "armed_person_proximity",
    "conditions": [
        {"class": "person", "confidence": ">0.8"},
        {"class": "weapon", "distance_to_person": "<50"}
    ],
    "actions": ["alert_security", "increase_monitoring"],
    "priority": 9
}
```

---

## 5. Evaluator

### 5.1 BasicEvaluator

Real-time performance evaluation with temporal consistency analysis.

#### Constructor
```python
def __init__(self, evaluation_window: int = 100,
             baseline_update_interval: int = 1000,
             temporal_window: int = 10)
```

#### Key Methods

##### `evaluate_performance(detections: list[DetectionResult],
                        ground_truth: list[DetectionResult] = None) -> PerformanceMetrics`
Calculate comprehensive performance metrics.

**Parameters:**
- `detections`: System detection results
- `ground_truth`: Ground truth annotations (optional)

**Returns:**
- PerformanceMetrics object with all metrics

##### `calculate_temporal_consistency(detections_history: list[list[DetectionResult]]) -> float`
Analyze detection stability over time.

**Parameters:**
- `detections_history`: Historical detection results

**Returns:**
- Temporal consistency score (0-1)

##### `get_performance_trends(metric: str, window: int = 50) -> list[float]`
Get performance trends for specific metric.

**Parameters:**
- `metric`: Metric name ("precision", "recall", "f1_score", etc.)
- `window`: Number of recent measurements

**Returns:**
- List of metric values over time

### 5.2 Metrics Calculation

#### Precision, Recall, F1-Score
Standard classification metrics calculated per class and averaged.

#### Matthews Correlation Coefficient (MCC)
```python
MCC = (TP × TN - FP × FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))
```

#### Mean Average Precision (mAP)
Calculated using 11-point interpolation method.

#### Temporal Consistency
Measures stability of detections across frames using IoU (Intersection over Union).

---

## 6. TDA Manager

### 6.1 BasicTDAManager

Performance-driven resource allocation with hysteresis.

#### Constructor
```python
def __init__(self, adaptation_threshold: float = 0.8,
             hysteresis_band: float = 0.05,
             resource_budget: dict[str, float] = None)
```

#### Key Methods

##### `assess_performance(metrics: PerformanceMetrics) -> dict[str, Any]`
Assess system performance and identify bottlenecks.

**Parameters:**
- `metrics`: Current performance metrics

**Returns:**
- Assessment dictionary with bottleneck analysis

##### `optimize_resources(assessment: dict[str, Any]) -> dict[str, Any]`
Calculate optimal resource allocation.

**Parameters:**
- `assessment`: Performance assessment results

**Returns:**
- Optimization recommendations

##### `adapt_topology(optimization: dict[str, Any]) -> None`
Apply topology changes to the system.

**Parameters:**
- `optimization`: Resource optimization plan

### 6.2 Adaptation Strategies

#### Resource Reallocation
- **GPU Memory**: Adjust model batch sizes and resolutions
- **CPU Threads**: Modify parallel processing levels
- **Memory Buffers**: Resize working/episodic memory
- **Module Activation**: Enable/disable optional components

#### Hysteresis Control
Prevents oscillation by maintaining state until performance crosses threshold bands.

```python
if performance < (threshold - hysteresis):
    activate_adaptation()
elif performance > (threshold + hysteresis):
    deactivate_adaptation()
```

---

## 7. Runtime Engine

### 7.1 Engine

Orchestrates the entire cognitive pipeline.

#### Constructor
```python
def __init__(self, config: dict[str, Any])
```

#### Key Methods

##### `initialize() -> None`
Initialize all modules and establish connections.

##### `process_frame(frame: np.ndarray) -> dict[str, Any]`
Process a single frame through the entire pipeline.

**Parameters:**
- `frame`: Input video frame

**Returns:**
- Processing results dictionary

##### `run_pipeline(video_path: str, max_frames: int = None) -> dict[str, Any]`
Run complete processing pipeline on video.

**Parameters:**
- `video_path`: Path to video file
- `max_frames`: Maximum frames to process

**Returns:**
- Complete processing statistics

##### `shutdown() -> None`
Gracefully shutdown all modules.

### 7.2 Pipeline Flow

```
Input Frame → Perception → Cognitive Core → Semantic Analysis → Evaluator → TDA → Output
                      ↑                                                            ↓
                      └─────────────────────── Adaptation Loop ─────────────────────┘
```

---

## 8. Configuration

### 8.1 Main Configuration Structure

```python
config = {
    "perception": {
        "model_path": "yolov8n.pt",
        "conf_threshold": 0.5,
        "weapon_classes": ["knife", "scissors", "baseball bat", "bottle", "fork"],
        "enable_mediapipe": True
    },
    "cognitive_core": {
        "max_episodic_memory": 1000,
        "max_working_memory": 100,
        "context_window": 50
    },
    "semantic": {
        "rules_path": "rules/tactical_rules.json",
        "threat_threshold": 0.7,
        "context_window": 10
    },
    "evaluator": {
        "evaluation_window": 100,
        "baseline_update_interval": 1000,
        "temporal_window": 10
    },
    "tda": {
        "adaptation_threshold": 0.8,
        "hysteresis_band": 0.05,
        "resource_budget": {
            "gpu_memory": 4.0,
            "cpu_threads": 8,
            "memory_buffers": 1000
        }
    },
    "ui": {
        "enable_tactical_display": True,
        "alert_threshold": 0.8,
        "auto_recording": True
    }
}
```

### 8.2 Configuration Validation

All configurations are validated using Pydantic models ensuring type safety and constraint checking.

---

## 9. Error Handling

### 9.1 Exception Hierarchy

```python
class UCogNetError(Exception):
    """Base exception for U-CogNet errors."""
    pass

class PerceptionError(UCogNetError):
    """Errors in perception modules."""
    pass

class CognitiveError(UCogNetError):
    """Errors in cognitive processing."""
    pass

class SemanticError(UCogNetError):
    """Errors in semantic analysis."""
    pass

class EvaluationError(UCogNetError):
    """Errors in performance evaluation."""
    pass

class TDAError(UCogNetError):
    """Errors in topology adaptation."""
    pass
```

### 9.2 Error Recovery

The system implements graceful degradation:
- **Module Failure**: Automatic fallback to simpler implementations
- **Resource Exhaustion**: TDA-triggered resource reallocation
- **Processing Errors**: Logging and continuation with degraded performance

### 9.3 Logging

Comprehensive logging with multiple levels:
- **DEBUG**: Detailed internal operations
- **INFO**: Normal operations and milestones
- **WARNING**: Recoverable issues
- **ERROR**: Critical failures requiring attention
- **CRITICAL**: System-threatening failures

---

## Usage Examples

### Basic Setup
```python
from ucognet import Engine

# Load configuration
config = load_config("config/default.json")

# Initialize engine
engine = Engine(config)
await engine.initialize()

# Process video
results = await engine.run_pipeline("video.mp4", max_frames=150)
print(f"Processed {results['frames_processed']} frames at {results['avg_fps']:.1f} FPS")
```

### Custom Module Implementation
```python
from ucognet.core.protocols import PerceptionProtocol

class CustomDetector(PerceptionProtocol):
    async def process_frame(self, frame: np.ndarray) -> list[DetectionResult]:
        # Custom detection logic
        return detections

# Register custom module
engine.register_module("perception", CustomDetector())
```

---

**For more examples and tutorials, see the `/examples` directory.**  
**API compatibility guaranteed for major versions. Minor versions may add features.**</content>
<parameter name="filePath">/mnt/c/Users/desar/Documents/Science/UCogNet/API_DOCUMENTATION.md