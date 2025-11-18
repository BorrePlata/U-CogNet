"""
Prueba de integración con YOLOv8 real.
Compara rendimiento entre simulación y modelo real.
"""

import time
import json
import numpy as np
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

def run_yolo_integration_test(num_iterations: int = 50, use_real_yolo: bool = True):
    """
    Prueba de integración con YOLOv8 real vs simulado.
    """
    logger.info(f"Iniciando prueba YOLOv8 {'REAL' if use_real_yolo else 'SIMULADO'}")

    # Configurar detector
    if use_real_yolo:
        detector = VisionDetector(model_path='yolov8n.pt', confidence_threshold=0.3)
    else:
        detector = VisionDetector(confidence_threshold=0.5)  # Simulado

    # Inicializar otros módulos
    input_handler = InputHandler(width=640, height=480)
    cognitive_core = CognitiveCore()
    semantic_feedback = SemanticFeedback()
    evaluator = Evaluator()
    trainer_loop = TrainerLoop()
    tda_manager = TDAManager()
    visual_interface = VisualInterface()

    # Métricas de la prueba
    test_metrics = {
        'test_type': 'YOLOv8_REAL' if use_real_yolo else 'SIMULATED',
        'iterations': num_iterations,
        'start_time': time.time(),
        'latencies': [],
        'detections_per_frame': [],
        'f1_scores': [],
        'throughput_history': [],
        'yolo_inference_times': [],
        'detected_classes': set()
    }

    # Loop principal
    for i in range(num_iterations):
        iteration_start = time.time()

        # Captura de frame
        frame = input_handler.get_frame()

        # Detección con medición de tiempo
        detect_start = time.time()
        detections = detector.detect(frame)
        detect_time = time.time() - detect_start

        # Actualizar clases detectadas
        for det in detections:
            test_metrics['detected_classes'].add(det.class_name)

        # Continuar con el loop cognitivo
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
        load = 0.5 + 0.3 * (i / num_iterations)
        state = build_system_state(metrics, tda_manager.update_topology(state) if i > 0 else tda_manager.update_topology(SystemState([], {}, metrics, {})), load)
        topology = tda_manager.update_topology(state)

        # Visualización
        visual_interface.render(frame, detections, text, state.__dict__)

        # Registrar métricas
        test_metrics['latencies'].append(latency)
        test_metrics['detections_per_frame'].append(len(detections))
        test_metrics['f1_scores'].append(metrics.f1_score)
        test_metrics['throughput_history'].append(metrics.throughput_fps)
        test_metrics['yolo_inference_times'].append(detect_time)

        if (i + 1) % 10 == 0:
            logger.info(f"Iteración {i+1}/{num_iterations}: F1={metrics.f1_score:.3f}, Detecciones={len(detections)}, YOLO_time={detect_time:.4f}s")

    # Calcular métricas finales
    test_metrics['end_time'] = time.time()
    test_metrics['total_time'] = test_metrics['end_time'] - test_metrics['start_time']
    test_metrics['avg_latency'] = sum(test_metrics['latencies']) / len(test_metrics['latencies'])
    test_metrics['avg_detections'] = sum(test_metrics['detections_per_frame']) / len(test_metrics['detections_per_frame'])
    test_metrics['final_f1'] = test_metrics['f1_scores'][-1]
    test_metrics['avg_throughput'] = sum(test_metrics['throughput_history']) / len(test_metrics['throughput_history'])
    test_metrics['avg_yolo_time'] = sum(test_metrics['yolo_inference_times']) / len(test_metrics['yolo_inference_times'])
    test_metrics['detected_classes'] = list(test_metrics['detected_classes'])

    # Guardar resultados
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    filename = f"yolo_real_test_results.json" if use_real_yolo else f"simulated_test_results.json"
    with open(results_dir / filename, 'w') as f:
        json.dump(test_metrics, f, indent=2)

    logger.info("Prueba YOLOv8 completada")
    logger.info(f"Resultados guardados en {results_dir / filename}")

    # Resumen
    print(f"\n{'='*60}")
    print(f"RESULTADOS PRUEBA {'YOLOv8 REAL' if use_real_yolo else 'SIMULACIÓN'}")
    print(f"{'='*60}")
    print(f"Iteraciones: {num_iterations}")
    print(".2f")
    print(".4f")
    print(".4f")
    print(".2f")
    print(".3f")
    print(".2f")
    print(f"Clases detectadas: {len(test_metrics['detected_classes'])}")
    print(f"Top clases: {test_metrics['detected_classes'][:5]}")
    print(f"{'='*60}")

    return test_metrics

def compare_simulated_vs_real():
    """Compara rendimiento entre simulación y YOLOv8 real."""
    print("\n🔬 COMPARACIÓN: SIMULADO vs YOLOv8 REAL")
    print("="*50)

    # Ejecutar simulación
    print("\n📊 Ejecutando prueba SIMULADA...")
    sim_results = run_yolo_integration_test(num_iterations=30, use_real_yolo=False)

    # Ejecutar YOLO real
    print("\n🤖 Ejecutando prueba YOLOv8 REAL...")
    try:
        real_results = run_yolo_integration_test(num_iterations=30, use_real_yolo=True)
    except Exception as e:
        print(f"❌ Error con YOLOv8 real: {e}")
        print("Usando solo resultados simulados")
        real_results = None

    # Comparación
    if real_results:
        print("\n📈 COMPARACIÓN DE RENDIMIENTO:")
        print(f"F1 Score - Simulado: {sim_results['final_f1']:.3f} vs Real: {real_results['final_f1']:.3f}")
        print(f"Latencia Promedio - Simulado: {sim_results['avg_latency']:.4f}s vs Real: {real_results['avg_latency']:.4f}s")
        print(f"Throughput - Simulado: {sim_results['avg_throughput']:.2f} FPS vs Real: {real_results['avg_throughput']:.2f} FPS")
        print(f"Detecciones/frame - Simulado: {sim_results['avg_detections']:.2f} vs Real: {real_results['avg_detections']:.2f}")
        print(f"Clases detectadas - Simulado: {len(sim_results['detected_classes'])} vs Real: {len(real_results['detected_classes'])}")

        # Análisis
        latency_improvement = (sim_results['avg_latency'] - real_results['avg_latency']) / sim_results['avg_latency'] * 100
        print(f"Mejora de latencia: {latency_improvement:.1f}%")

        if real_results['avg_detections'] > sim_results['avg_detections']:
            print("✅ YOLOv8 detecta más objetos que la simulación")
        else:
            print("⚠️  YOLOv8 detecta menos objetos (posiblemente más preciso)")

if __name__ == "__main__":
    compare_simulated_vs_real()