"""
Pruebas unitarias para módulos de U-CogNet.
Genera métricas de cobertura y rendimiento para análisis científico.
"""

import pytest
import numpy as np
import time
from ucognet.input_handler import InputHandler
from ucognet.vision_detector import VisionDetector
from ucognet.cognitive_core import CognitiveCore
from ucognet.semantic_feedback import SemanticFeedback
from ucognet.evaluator import Evaluator
from ucognet.common.types import Detection, Event

class TestInputHandler:
    def test_get_frame(self):
        handler = InputHandler(width=320, height=240)
        frame = handler.get_frame()

        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8
        assert np.all(frame >= 0) and np.all(frame <= 255)

    def test_frame_metadata(self):
        handler = InputHandler()
        frame, metadata = handler.get_frame_with_metadata()

        assert 'frame_id' in metadata
        assert 'timestamp' in metadata
        assert 'source' in metadata
        assert metadata['source'] == 'simulated'

class TestVisionDetector:
    def test_detect_empty_frame(self):
        detector = VisionDetector(confidence_threshold=0.9)  # Alto threshold
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame)
        assert isinstance(detections, list)

    def test_detect_with_detections(self):
        detector = VisionDetector(confidence_threshold=0.1)  # Bajo threshold
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame)

        for detection in detections:
            assert isinstance(detection, Detection)
            assert detection.confidence >= 0.1
            assert detection.class_name in detector.classes

class TestCognitiveCore:
    def test_store_and_retrieve(self):
        core = CognitiveCore(buffer_size=50)

        # Crear evento de prueba
        detection = Detection('person', 0.8, (10, 10, 50, 50), time.time())
        event = Event(time.time(), [detection])

        core.store(event)
        context = core.get_context()

        assert len(context.recent_events) > 0
        assert context.recent_events[-1] == event

    def test_buffer_limits(self):
        core = CognitiveCore(buffer_size=5)

        for i in range(10):
            event = Event(time.time(), [])
            core.store(event)

        assert len(core.event_buffer) == 5

class TestSemanticFeedback:
    def test_generate_feedback(self):
        feedback = SemanticFeedback()

        # Contexto vacío
        context = type('Context', (), {'recent_events': [], 'window_size': 10, 'current_state': {}})()
        detections = []

        text = feedback.generate(context, detections)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_convoy_detection(self):
        feedback = SemanticFeedback()

        context = type('Context', (), {'recent_events': [], 'window_size': 10, 'current_state': {}})()
        detections = [
            Detection('vehicle', 0.8, (0, 0, 10, 10), time.time()),
            Detection('tank', 0.9, (20, 20, 30, 30), time.time())
        ]

        text = feedback.generate(context, detections)
        assert 'convoy' in text.lower()

class TestEvaluator:
    def test_metrics_calculation(self):
        evaluator = Evaluator()

        # Simular algunas predicciones
        for _ in range(10):
            metrics = evaluator.update()

        assert metrics.precision >= 0
        assert metrics.recall >= 0
        assert metrics.f1_score >= 0
        assert metrics.mcc >= -1 and metrics.mcc <= 1

    def test_latency_tracking(self):
        evaluator = Evaluator()

        metrics = evaluator.update(latency=0.1)
        assert metrics.latency_ms == 100.0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])