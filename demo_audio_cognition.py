#!/usr/bin/env python3
"""
Demostración del Procesador Cognitivo de Audio
Muestra extracción, razonamiento, interiorización e imaginación
"""

import asyncio
import sys
import os
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ucognet import AudioCognitiveProcessor, CognitiveCore, SemanticFeedback

async def demo_audio_cognition():
    """Demostración completa del procesamiento cognitivo de audio."""

    print("🎵 U-CogNet: Demostración de Procesamiento Cognitivo de Audio")
    print("=" * 60)

    # Inicializar componentes cognitivos
    cognitive_core = CognitiveCore(buffer_size=50)
    semantic_feedback = SemanticFeedback()

    # Crear procesador cognitivo de audio
    audio_processor = AudioCognitiveProcessor(
        cognitive_core=cognitive_core,
        semantic_feedback=semantic_feedback
    )

    print("✅ Componentes cognitivos inicializados")

    # Buscar un video de prueba
    video_paths = [
        "test_video.mp4",
        "sample_video.mp4",
        "demo_video.mp4"
    ]

    video_path = None
    for path in video_paths:
        if os.path.exists(path):
            video_path = path
            break

    if not video_path:
        print("⚠️  No se encontró video de prueba. Creando audio sintético para demostración...")

        # Crear audio sintético para demostración
        import numpy as np
        sample_rate = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Generar tono musical
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # La 440Hz

        # Simular procesamiento con datos sintéticos
        from ucognet.common.audio_types import AudioData
        synthetic_audio = AudioData(
            waveform=audio_data,
            sample_rate=sample_rate,
            duration=duration,
            source="synthetic_tone",
            timestamp=0.0
        )

        print("🎛️  Procesando audio sintético...")

        # Procesar razonamiento
        reasoning = await audio_processor._reason_about_audio(synthetic_audio)
        print(f"🧠 Razonamiento: {reasoning.event_type} (confianza: {reasoning.confidence:.2f})")
        print(f"📝 Descripción semántica: {reasoning.semantic_description}")

        # Interiorizar
        await audio_processor._interiorize_audio(synthetic_audio, reasoning)
        print("🧠 Audio interiorizado en memoria cognitiva")

        # Generar imaginación
        imagination = await audio_processor._generate_imagination(synthetic_audio, reasoning)
        print(f"🎨 Imaginación generada - Novedad: {imagination.novelty_score:.2f}, Coherencia: {imagination.coherence_score:.2f}")

        # Calcular métricas
        metrics = audio_processor._calculate_metrics(synthetic_audio, reasoning, imagination, 0.5)
        print("📊 Métricas calculadas:")
        print(f"   Calidad de extracción: {metrics.extraction_quality:.2f}")
        print(f"   Precisión de razonamiento: {metrics.reasoning_accuracy:.2f}")
        print(f"   Profundidad de interiorización: {metrics.interiorization_depth:.2f}")
        print(f"   Creatividad de imaginación: {metrics.imagination_creativity:.2f}")
        print(f"   Latencia de procesamiento: {metrics.processing_latency:.2f}s")
        print(f"   Utilización de memoria: {metrics.memory_utilization:.2f}")

        # Mostrar estado cognitivo
        status = audio_processor.get_cognitive_status()
        print("\n🧠 Estado Cognitivo:")
        print(f"   Memoria de audio: {status['audio_memory_size']} patrones")
        print(f"   Patrones de razonamiento: {status['reasoning_patterns']}")
        print(f"   Historial de métricas: {status['metrics_history_length']}")
        print(f"   Creatividad promedio: {status['average_creativity']:.2f}")
        print(f"   Precisión de razonamiento promedio: {status['average_reasoning_accuracy']:.2f}")
    else:
        print(f"🎬 Procesando video real: {video_path}")

        try:
            # Procesar video completo
            result = await audio_processor.process_video_audio(video_path)

            print("✅ Procesamiento completado exitosamente!")
            print(f"⏱️  Tiempo total: {result['processing_time']:.2f}s")

            # Mostrar resultados detallados
            reasoning = result['reasoning']
            imagination = result['imagination']
            metrics = result['metrics']

            print("\n🧠 RAZONAMIENTO:")
            print(f"   Tipo de evento: {reasoning.event_type}")
            print(f"   Confianza: {reasoning.confidence:.2f}")
            print(f"   Descripción: {reasoning.semantic_description}")
            print(f"   Insights cognitivos: {len(reasoning.cognitive_insights)}")

            print("\n🎨 IMAGINACIÓN:")
            print(f"   Escenarios imaginados: {len(imagination.imagined_scenarios)}")
            print(f"   Novedad: {imagination.novelty_score:.2f}")
            print(f"   Coherencia: {imagination.coherence_score:.2f}")
            print(f"   Asociaciones creativas: {len(imagination.creative_associations)}")

            print("\n📊 MÉTRICAS COGNITIVAS:")
            print(f"   Calidad de extracción: {metrics.extraction_quality:.2f}")
            print(f"   Precisión de razonamiento: {metrics.reasoning_accuracy:.2f}")
            print(f"   Profundidad de interiorización: {metrics.interiorization_depth:.2f}")
            print(f"   Creatividad de imaginación: {metrics.imagination_creativity:.2f}")
            print(f"   Latencia de procesamiento: {metrics.processing_latency:.2f}s")
            print(f"   Utilización de memoria: {metrics.memory_utilization:.2f}")
        except Exception as e:
            print(f"❌ Error procesando video: {e}")
            print("💡 Asegúrate de que MoviePy esté instalado: pip install moviepy")

    print("\n🎯 Demostración completada!")
    print("El sistema ha demostrado capacidad para:")
    print("  • Extraer audio de fuentes diversas")
    print("  • Razonar sobre el contenido semántico")
    print("  • Interiorizar patrones en memoria cognitiva")
    print("  • Generar imaginación creativa")
    print("  • Medir todas las capacidades cognitivas")

if __name__ == "__main__":
    asyncio.run(demo_audio_cognition())