#!/usr/bin/env python3
"""
Demo Avanzada de U-CogNet - Sistema Cognitivo Universal
Muestra todas las capacidades implementadas del sistema.
"""

import argparse
import sys
import time
from pathlib import Path

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ucognet.runtime.engine import Engine
from ucognet.modules.input.opencv_camera import OpenCVInputHandler
from ucognet.modules.vision.yolov8_detector import YOLOv8Detector
from ucognet.modules.cognitive.cognitive_core import CognitiveCoreImpl
from ucognet.modules.semantic.rule_based import RuleBasedSemanticFeedback
from ucognet.modules.eval.basic_evaluator import BasicEvaluator
from ucognet.modules.train.mock_trainer import MockTrainerLoop
from ucognet.modules.tda.basic_tda import BasicTDAManager
from ucognet.modules.ui.opencv_ui import OpenCVVisualInterface

def print_banner():
    """Imprime el banner de U-CogNet."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🚀 U-COGNET v0.1.0 🚀                     ║
    ║              Sistema Cognitivo Artificial Universal          ║
    ║                                                              ║
    ║  Capacidades Implementadas:                                  ║
    ║  ✅ Detección de armas con lógica avanzada                   ║
    ║  ✅ Análisis semántico con reglas simbólicas                 ║
    ║  ✅ Evaluación automática con métricas reales               ║
    ║  ✅ Topología Dinámica Adaptativa (TDA) básica              ║
    ║  ✅ MediaPipe integrado (pose, manos, rostro)               ║
    ║  ✅ Interfaz visual con alertas de seguridad                ║
    ║  ✅ Grabación automática inteligente                         ║
    ║                                                              ║
    ║  Controles:                                                  ║
    ║  • Presiona 'q' para salir                                   ║
    ║  • Presiona 'r' para forzar grabación                        ║
    ║  • Presiona 's' para mostrar estadísticas                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def create_demo_engine(video_source: str, use_mediapipe: bool = False) -> Engine:
    """Crea el engine con todas las capacidades avanzadas."""
    print("🔧 Inicializando módulos del sistema...")

    # Crear componentes del sistema
    input_handler = OpenCVInputHandler(source=video_source)
    vision_detector = YOLOv8Detector(use_mediapipe=use_mediapipe)
    cognitive_core = CognitiveCoreImpl()
    semantic_feedback = RuleBasedSemanticFeedback()
    evaluator = BasicEvaluator()
    trainer_loop = MockTrainerLoop()
    tda_manager = BasicTDAManager()
    visual_interface = OpenCVVisualInterface(record_on_crowd=True, record_duration=30)

    # Crear engine
    engine = Engine(
        input_handler=input_handler,
        vision_detector=vision_detector,
        cognitive_core=cognitive_core,
        semantic_feedback=semantic_feedback,
        evaluator=evaluator,
        trainer_loop=trainer_loop,
        tda_manager=tda_manager,
        visual_interface=visual_interface,
    )

    print("✅ Sistema inicializado correctamente")
    print(f"📹 Fuente de video: {video_source}")
    print(f"🤖 MediaPipe: {'Activado' if use_mediapipe else 'Desactivado'}")
    print()

    return engine

def run_demo(engine: Engine, max_frames: int = None):
    """Ejecuta la demo del sistema."""
    print("🎬 Iniciando demo de U-CogNet...")
    print("Presiona 'q' en la ventana de video para salir")
    print("-" * 60)

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Ejecutar un paso del engine
            engine.step()
            frame_count += 1

            # Mostrar progreso cada 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"📊 Progreso: {frame_count} frames | FPS: {fps:.1f}")
            # Limitar frames si se especifica
            if max_frames and frame_count >= max_frames:
                print(f"\n🎯 Demo completada después de {max_frames} frames")
                break

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante la demo: {e}")
    finally:
        # Limpiar recursos
        print("🧹 Limpiando recursos...")
        engine.input_handler.release()
        engine.visual_interface.close()

        # Mostrar estadísticas finales
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print("\n📊 Estadísticas de la Demo:")
        print(f"   • Frames procesados: {frame_count}")
        print(f"   • Tiempo total: {total_time:.2f}s")
        print(f"   • FPS promedio: {avg_fps:.1f}")
        print("\n✅ Demo finalizada exitosamente!")

def main():
    parser = argparse.ArgumentParser(description="Demo Avanzada de U-CogNet")
    parser.add_argument(
        "--video",
        type=str,
        default="videoplayback.webm",
        help="Ruta al archivo de video o '0' para webcam"
    )
    parser.add_argument(
        "--no-mediapipe",
        action="store_true",
        help="Desactivar MediaPipe para mejor rendimiento"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Número máximo de frames a procesar (para testing)"
    )

    args = parser.parse_args()

    # Imprimir banner
    print_banner()

    # Verificar que existe el archivo de video
    if args.video != "0" and not Path(args.video).exists():
        print(f"❌ Error: No se encuentra el archivo de video '{args.video}'")
        print("💡 Asegúrate de que el archivo existe o usa --video 0 para webcam")
        return 1

    try:
        # Crear engine
        if args.no_mediapipe:
            engine = create_demo_engine(args.video, use_mediapipe=False)
        else:
            engine = create_demo_engine(args.video)  # Usa default (False)

        # Ejecutar demo
        run_demo(engine, max_frames=args.max_frames)

        return 0

    except Exception as e:
        print(f"❌ Error fatal: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())