"""
Script de integración inicial para U-CogNet.
Prueba básica del loop cognitivo con módulos simulados.
Genera métricas para análisis científico.
"""

import time
import json
from pathlib import Path
from ucognet import (
    InputHandler, VisionDetector, CognitiveCore,
    SemanticFeedback, Evaluator, TrainerLoop,
    TDAManager, VisualInterface
)
from ucognet.common.types import Event, SystemState
from ucognet.common.logging import logger

def build_event(frame, detections):
    """Construye un evento cognitivo."""
    return Event(
        timestamp=time.time(),
        detections=detections,
        frame=frame
    )

def build_system_state(metrics, topology, load):
    """Construye el estado del sistema."""
    return SystemState(
        active_modules=topology.active_modules,
        resource_usage={'cpu': load, 'memory': load * 0.8},
        performance_metrics=metrics,
        topology_config=topology.__dict__
    )

def run_integration_test(num_iterations: int = 100):
    """
    Ejecuta prueba de integración del sistema.
    Genera métricas para análisis científico.
    """
    logger.info("Iniciando prueba de integración de U-CogNet")

    # Inicializar módulos
    input_handler = InputHandler()
    vision_detector = VisionDetector()
    cognitive_core = CognitiveCore()
    semantic_feedback = SemanticFeedback()
    evaluator = Evaluator()
    trainer_loop = TrainerLoop()
    tda_manager = TDAManager()
    visual_interface = VisualInterface()

    # Métricas de la prueba
    test_metrics = {
        'iterations': num_iterations,
        'start_time': time.time(),
        'latencies': [],
        'detections_per_frame': [],
        'f1_scores': [],
        'throughput_history': []
    }

    # Loop principal (versión simplificada del ADN)
    for i in range(num_iterations):
        iteration_start = time.time()

        # Ciclo cognitivo
        frame = input_handler.get_frame()
        detections = vision_detector.detect(frame)
        event = build_event(frame, detections)
        cognitive_core.store(event)
        context = cognitive_core.get_context()
        text = semantic_feedback.generate(context, detections)

        # Evaluación
        latency = time.time() - iteration_start
        metrics = evaluator.update(latency=latency)

        # Entrenamiento condicional
        recent_events = cognitive_core.get_recent_events(5)
        trainer_loop.step(recent_events)
        trainer_loop.maybe_train()

        # TDA
        load = 0.5 + 0.3 * (i / num_iterations)  # Simular carga creciente
        state = build_system_state(metrics, tda_manager.update_topology(state) if i > 0 else tda_manager.update_topology(SystemState([], {}, metrics, {})), load)
        topology = tda_manager.update_topology(state)

        # Visualización
        visual_interface.render(frame, detections, text, state.__dict__)

        # Registrar métricas
        test_metrics['latencies'].append(latency)
        test_metrics['detections_per_frame'].append(len(detections))
        test_metrics['f1_scores'].append(metrics.f1_score)
        test_metrics['throughput_history'].append(metrics.throughput_fps)

        if (i + 1) % 20 == 0:
            logger.info(f"Iteración {i+1}/{num_iterations}: F1={metrics.f1_score:.3f}, Latency={latency:.4f}s")

    # Calcular métricas finales
    test_metrics['end_time'] = time.time()
    test_metrics['total_time'] = test_metrics['end_time'] - test_metrics['start_time']
    test_metrics['avg_latency'] = sum(test_metrics['latencies']) / len(test_metrics['latencies'])
    test_metrics['avg_detections'] = sum(test_metrics['detections_per_frame']) / len(test_metrics['detections_per_frame'])
    test_metrics['final_f1'] = test_metrics['f1_scores'][-1]
    test_metrics['avg_throughput'] = sum(test_metrics['throughput_history']) / len(test_metrics['throughput_history'])

    # Guardar resultados
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "integration_test_results.json", 'w') as f:
        json.dump(test_metrics, f, indent=2)

    logger.info("Prueba de integración completada")
    logger.info(f"Resultados guardados en {results_dir / 'integration_test_results.json'}")

    # Resumen para paper
    print("\n" + "="*60)
    print("RESUMEN DE PRUEBA DE INTEGRACIÓN - U-CogNet v0.1.0")
    print("="*60)
    print(f"Iteraciones: {num_iterations}")
    print(".2f")
    print(".4f")
    print(".2f")
    print(".3f")
    print(".2f")
    print(".2f")
    print("="*60)

    return test_metrics

if __name__ == "__main__":
    run_integration_test(100)