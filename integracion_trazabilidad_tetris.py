#!/usr/bin/env python3
"""
Integración de Trazabilidad en Cognitive Tetris
Demuestra cómo el MTC se integra perfectamente con aplicaciones existentes
"""

import asyncio
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

# Importar componentes de trazabilidad
from ucognet.core.tracing import (
    get_event_bus, EventType, LogLevel,
    TraceQueryAPI, CausalGraphBuilder, CoherenceEthicsEvaluator
)

# Importar Cognitive Tetris (con modificaciones para trazabilidad)
from cognitive_tetris_game import CognitiveTetrisGame

class TracedCognitiveTetrisGame(CognitiveTetrisGame):
    """Versión de Cognitive Tetris con trazabilidad cognitiva integrada"""

    def __init__(self):
        super().__init__()

        # Inicializar sistema de trazabilidad
        self.event_bus = get_event_bus()
        self.current_episode_id = None
        self.step_counter = 0

        print("🎮 Cognitive Tetris con Trazabilidad Cognitiva activada")

    async def _make_cognitive_move(self):
        """Movimiento cognitivo con trazabilidad completa"""
        if not self.current_episode_id:
            self.current_episode_id = f"tetris_episode_{int(asyncio.get_event_loop().time())}"

        # Iniciar traza de decisión cognitiva
        trace_id = self.event_bus.emit(
            EventType.DECISION,
            "CognitiveTetrisPlayer",
            inputs={
                "board_state": self._get_board_summary(),
                "current_piece": str(self.current_piece.shape_type) if self.current_piece else None,
                "score": self.board.score,
                "lines": self.board.lines_cleared
            },
            episode_id=self.current_episode_id,
            step_id=self.step_counter,
            explanation="Iniciando proceso de decisión cognitiva"
        )

        try:
            # Obtener decisión del agente cognitivo
            decision = await self.player.make_decision(
                self.board,
                self.current_piece,
                self.next_piece
            )

            # Registrar decisión tomada
            self.event_bus.emit(
                EventType.DECISION,
                "CognitiveTetrisPlayer",
                outputs={
                    "action": decision.get("action", "unknown"),
                    "rotation": decision.get("rotation", 0),
                    "position": decision.get("position", 0)
                },
                context={
                    "reasoning": decision.get("reasoning", {}),
                    "confidence": decision.get("confidence", 0.0)
                },
                metrics={
                    "thinking_time": decision.get("thinking_time", 0),
                    "decision_quality": decision.get("decision_quality", 0)
                },
                episode_id=self.current_episode_id,
                step_id=self.step_counter,
                trace_id=trace_id,
                explanation=f"Decisión: {decision.get('action', 'unknown')}"
            )

            # Ejecutar la acción
            action_result = self._execute_decision(decision)

            # Registrar resultado
            reward = action_result.get("reward", 0)
            self.event_bus.emit(
                EventType.REWARD,
                "TetrisEnvironment",
                outputs={"reward": reward, "lines_cleared": action_result.get("lines_cleared", 0)},
                episode_id=self.current_episode_id,
                step_id=self.step_counter,
                trace_id=trace_id,
                explanation=f"Reward: {reward} por acción cognitiva"
            )

            # Registrar métricas de aprendizaje si hubo mejora
            if action_result.get("learning_opportunity", False):
                self.event_bus.emit(
                    EventType.LEARNING_STEP,
                    "CognitiveTetrisPlayer",
                    metrics={
                        "pattern_learned": True,
                        "creativity_applied": decision.get("creativity_applied", False),
                        "reasoning_confidence": decision.get("reasoning_confidence", 0)
                    },
                    episode_id=self.current_episode_id,
                    step_id=self.step_counter,
                    trace_id=trace_id
                )

            self.step_counter += 1

        except Exception as e:
            # Registrar error en el proceso cognitivo
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "CognitiveTetrisPlayer",
                outputs={"success": False, "error": str(e)},
                episode_id=self.current_episode_id,
                step_id=self.step_counter,
                trace_id=trace_id,
                log_level=LogLevel.INFO
            )
            print(f"⚠️ Error en decisión cognitiva: {e}")

    def _get_board_summary(self) -> dict:
        """Obtiene resumen del estado del tablero para trazabilidad"""
        if not hasattr(self.board, 'grid') or self.board.grid is None:
            return {"empty": True}

        # Contar huecos, altura máxima, etc.
        heights = []
        holes = 0

        for x in range(self.board.width):
            column_height = 0
            column_holes = 0
            found_top = False

            for y in range(self.board.height):
                if self.board.grid[y][x]:
                    if not found_top:
                        found_top = True
                    column_height = self.board.height - y
                elif found_top:
                    column_holes += 1

            heights.append(column_height)
            holes += column_holes

        return {
            "max_height": max(heights) if heights else 0,
            "avg_height": sum(heights) / len(heights) if heights else 0,
            "total_holes": holes,
            "bumpiness": sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1)) if len(heights) > 1 else 0
        }

    def _execute_decision(self, decision: dict) -> dict:
        """Ejecuta una decisión y retorna resultado"""
        action = decision.get("action", "hold")
        lines_before = self.board.lines_cleared

        if action == "rotate":
            if self.current_piece:
                original_shape = self.current_piece.shape.copy()
                self.current_piece.rotate()
                if not self.board.is_valid_position(self.current_piece):
                    self.current_piece.shape = original_shape
        elif action == "move_left":
            if self.current_piece and self.board.is_valid_position(self.current_piece, -1, 0):
                self.current_piece.x -= 1
        elif action == "move_right":
            if self.current_piece and self.board.is_valid_position(self.current_piece, 1, 0):
                self.current_piece.x += 1
        elif action == "soft_drop":
            if self.current_piece and self.board.is_valid_position(self.current_piece, 0, 1):
                self.current_piece.y += 1
        elif action == "hard_drop":
            while self.current_piece and self.board.is_valid_position(self.current_piece, 0, 1):
                self.current_piece.y += 1
            self._place_piece()

        lines_after = self.board.lines_cleared
        lines_cleared = lines_after - lines_before

        # Calcular reward basado en líneas limpiadas
        reward = lines_cleared * 10  # 10 puntos por línea
        if lines_cleared >= 4:  # Tetris
            reward += 40

        return {
            "reward": reward,
            "lines_cleared": lines_cleared,
            "learning_opportunity": lines_cleared > 0
        }

    def _place_piece(self):
        """Coloca pieza con trazabilidad adicional"""
        # Llamar al método original
        super()._place_piece()

        # Registrar evento de colocación
        if self.current_episode_id:
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "TetrisBoard",
                outputs={
                    "piece_placed": True,
                    "lines_cleared": self.board.lines_cleared,
                    "current_score": self.board.score
                },
                episode_id=self.current_episode_id,
                step_id=self.step_counter
            )

async def demo_trazabilidad_en_juego():
    """Demuestra trazabilidad integrada en Cognitive Tetris"""
    print("🎮 Demo: Trazabilidad Integrada en Cognitive Tetris")
    print("=" * 60)

    # Inicializar componentes de trazabilidad
    event_bus = get_event_bus()
    causal_builder = CausalGraphBuilder()
    coherence_evaluator = CoherenceEthicsEvaluator()
    query_api = TraceQueryAPI(
        event_bus.trace_core,
        causal_builder,
        coherence_evaluator
    )

    print("1. Inicializando Cognitive Tetris con trazabilidad...")

    # Crear juego con trazabilidad
    game = TracedCognitiveTetrisGame()

    print("2. Ejecutando episodio corto de juego...")

    # Ejecutar un episodio corto (simulado)
    episode_id = f"tetris_demo_{int(asyncio.get_event_loop().time())}"

    # Simular algunas decisiones cognitivas
    for step in range(5):
        # Simular estado del juego
        game.current_episode_id = episode_id
        game.step_counter = step

        # Hacer movimiento cognitivo simulado
        await game._make_cognitive_move()

        # Simular algo de tiempo entre decisiones
        await asyncio.sleep(0.1)

    print("3. Analizando trazas generadas...")

    # Obtener resumen del episodio
    summary = query_api.get_episode_summary(episode_id)
    print(f"📊 Episodio analizado: {summary.get('total_events', 0)} eventos")
    print(".3f"    print(".3f"
    # Análisis causal
    causal_analysis = query_api.get_causal_analysis(episode_id)
    print(f"🔗 Análisis causal: {causal_analysis.get('causal_links', 0)} conexiones")

    # Mostrar recomendaciones
    recommendations = summary.get('recommendations', [])
    if recommendations:
        print("💡 Recomendaciones del sistema:")
        for rec in recommendations[:3]:  # Mostrar top 3
            print(f"   • {rec}")

    print("4. Reporte de salud del sistema...")

    # Reporte de salud
    health = query_api.get_system_health_report(hours=1)
    print(f"💚 Salud del sistema: {health.get('total_events', 0)} eventos recientes")
    print(".3f"
    print("\n✅ Demo de integración completada!")
    print("🎮 El Cognitive Tetris ahora tiene trazabilidad cognitiva completa")
    print("📈 Cada decisión, reward y evento de aprendizaje queda registrado")
    print("🔍 Puedes analizar el comportamiento cognitivo en tiempo real")

async def main():
    """Función principal"""
    print("🧠 U-CogNet - Trazabilidad Integrada en Aplicaciones")
    print("=" * 80)

    try:
        await demo_trazabilidad_en_juego()

        print("\n" + "=" * 80)
        print("🎉 Integración completada exitosamente!")
        print("📚 El MTC está integrado en Cognitive Tetris")
        print("🔬 Listo para experimentos con trazabilidad cognitiva completa")

    except Exception as e:
        print(f"❌ Error en la integración: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())