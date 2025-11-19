#!/usr/bin/env python3
"""
Demostración del Sistema de Trazabilidad Cognitiva
Muestra cómo funciona el MTC (Meta-módulo de Trazabilidad Cognitiva)
"""

import asyncio
import time
from pathlib import Path
import sys

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from ucognet.core.tracing import (
    CognitiveTraceCore, CognitiveEventBus, CausalGraphBuilder,
    CoherenceEthicsEvaluator, TraceQueryAPI, TraceStorageManager,
    StorageConfig, EventType, LogLevel, emit_event
)

async def demo_trazabilidad_basica():
    """Demostración básica del sistema de trazabilidad"""
    print("🧠 Demo: Sistema de Trazabilidad Cognitiva Básica")
    print("=" * 60)

    # 1. Inicializar componentes
    print("1. Inicializando componentes...")

    trace_core = CognitiveTraceCore(buffer_size=1000)
    event_bus = CognitiveEventBus(trace_core)
    causal_builder = CausalGraphBuilder()
    coherence_evaluator = CoherenceEthicsEvaluator()
    storage_config = StorageConfig(
        base_path="demo_traces",
        compression_enabled=True,
        retention_days=7
    )
    storage_manager = TraceStorageManager(storage_config)
    query_api = TraceQueryAPI(trace_core, causal_builder, coherence_evaluator)

    print("✅ Componentes inicializados")

    # 2. Simular actividad del sistema
    print("\n2. Simulando actividad del sistema...")

    episode_id = "demo_episode_001"

    # Simular un episodio de juego
    for step in range(10):
        # Evento de decisión
        decision_id = event_bus.emit(
            EventType.DECISION,
            "MockAgent",
            inputs={"game_state": f"step_{step}"},
            outputs={"action": "move_right" if step % 2 == 0 else "move_left"},
            episode_id=episode_id,
            step_id=step,
            explanation=f"Decisión en paso {step}"
        )

        # Simular reward
        reward = 1.0 if step % 3 == 0 else 0.0
        event_bus.emit(
            EventType.REWARD,
            "Environment",
            outputs={"reward": reward},
            episode_id=episode_id,
            step_id=step,
            trace_id=decision_id
        )

        # Evento de aprendizaje
        if step % 2 == 0:
            event_bus.emit(
                EventType.LEARNING_STEP,
                "MockAgent",
                metrics={"loss": 0.5 - step * 0.01},
                episode_id=episode_id,
                step_id=step
            )

        await asyncio.sleep(0.01)  # Simular tiempo de procesamiento

    print("✅ Actividad simulada completada")

    # 3. Consultas y análisis
    print("\n3. Realizando consultas y análisis...")

    # Obtener resumen del episodio
    summary = query_api.get_episode_summary(episode_id)
    print(f"📊 Resumen del episodio: {len(summary.get('event_counts', {}))} tipos de eventos")
    print(f"🎯 Score de coherencia: {summary.get('coherence_score', 0):.3f}")
    print(f"🤖 Score ético: {summary.get('ethics_score', 0):.3f}")

    # Análisis causal
    causal_analysis = query_api.get_causal_analysis(episode_id)
    print(f"🔗 Enlaces causales encontrados: {causal_analysis.get('causal_links', 0)}")

    # Reporte de salud del sistema
    health_report = query_api.get_system_health_report(hours=1)
    print(f"💚 Eventos en la última hora: {health_report.get('total_events', 0)}")

    print("✅ Consultas completadas")

    # 4. Estadísticas finales
    print("\n4. Estadísticas finales...")
    core_stats = trace_core.get_stats()
    storage_stats = storage_manager.get_storage_stats()

    print(f"📈 Eventos procesados: {core_stats.get('events_processed', 0)}")
    print(f"🎬 Episodios rastreados: {core_stats.get('episodes_tracked', 0)}")
    print(f"💾 Archivos de almacenamiento: {storage_stats.get('active_files', 0)}")

    print("\n🎉 Demo completada exitosamente!")

async def demo_trazabilidad_avanzada():
    """Demostración avanzada con múltiples episodios y análisis complejo"""
    print("\n🚀 Demo: Sistema de Trazabilidad Cognitiva Avanzada")
    print("=" * 60)

    # Configurar sistema
    trace_core = CognitiveTraceCore(buffer_size=5000)
    event_bus = CognitiveEventBus(trace_core)
    causal_builder = CausalGraphBuilder()
    query_api = TraceQueryAPI(trace_core, causal_builder)

    # Simular múltiples episodios
    print("Simulando múltiples episodios de aprendizaje...")

    for episode in range(3):
        episode_id = f"learning_episode_{episode:03d}"

        print(f"  📝 Episodio {episode + 1}/3: {episode_id}")

        # Simular aprendizaje progresivo
        for step in range(20):
            # Decisiones que mejoran con el tiempo
            decision_quality = min(0.9, 0.3 + (episode * 0.2) + (step * 0.02))

            decision_id = event_bus.emit(
                EventType.DECISION,
                "LearningAgent",
                outputs={"action": "optimal_choice", "confidence": decision_quality},
                metrics={"decision_quality": decision_quality},
                episode_id=episode_id,
                step_id=step
            )

            # Reward basado en calidad de decisión
            reward = decision_quality * 2 - 1  # Escala de -1 a 1
            event_bus.emit(
                EventType.REWARD,
                "Environment",
                outputs={"reward": reward},
                episode_id=episode_id,
                step_id=step,
                trace_id=decision_id
            )

            # Evento de aprendizaje cada pocos pasos
            if step % 5 == 0:
                event_bus.emit(
                    EventType.LEARNING_STEP,
                    "LearningAgent",
                    metrics={"loss": 1.0 - decision_quality, "learning_rate": 0.01},
                    episode_id=episode_id,
                    step_id=step
                )

    print("✅ Múltiples episodios simulados")

    # Análisis comparativo
    print("\n📊 Análisis comparativo de episodios:")

    for episode in range(3):
        episode_id = f"learning_episode_{episode:03d}"
        summary = query_api.get_episode_summary(episode_id)

        coherence = summary.get('coherence_score', 0)
        ethics = summary.get('ethics_score', 0)

        print(f"  🎯 Episodio {episode + 1}: Coherencia={coherence:.3f}, Ética={ethics:.3f}")

    # Análisis causal del mejor episodio
    best_episode = "learning_episode_002"  # El último debería ser el mejor
    causal_analysis = query_api.get_causal_analysis(best_episode)

    print(f"\n🔍 Análisis causal del mejor episodio:")
    print(f"  📈 Eventos totales: {causal_analysis.get('total_events', 0)}")
    print(f"  🔗 Conexiones causales: {causal_analysis.get('causal_links', 0)}")

    if causal_analysis.get('root_causes'):
        print(f"  🏆 Causas raíz identificadas: {len(causal_analysis['root_causes'])}")

    print("\n🎉 Demo avanzada completada!")

async def main():
    """Función principal de la demo"""
    print("🧠 U-CogNet - Demostración del Sistema de Trazabilidad Cognitiva")
    print("Meta-módulo de Trazabilidad Cognitiva (MTC)")
    print("=" * 80)

    try:
        # Demo básica
        await demo_trazabilidad_basica()

        # Demo avanzada
        await demo_trazabilidad_avanzada()

        print("\n" + "=" * 80)
        print("✅ Todas las demos completadas exitosamente!")
        print("📚 El sistema de trazabilidad está listo para producción.")
        print("🔍 Usa TraceQueryAPI para explorar trazas en tiempo real.")
        print("💾 Los datos se almacenan automáticamente en 'cognitive_traces/'")

    except Exception as e:
        print(f"❌ Error en la demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())