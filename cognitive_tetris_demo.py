#!/usr/bin/env python3
"""
Demostración de Cognitive Tetris - Versión de Consola
Muestra las capacidades cognitivas de U-CogNet en Tetris sin interfaz gráfica
"""

import asyncio
import time
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import os

# Configurar pygame para modo headless
os.environ['SDL_VIDEODRIVER'] = 'dummy'

sys.path.insert(0, str(Path(__file__).parent))

from cognitive_tetris import TetrisBoard, CognitiveTetrisPlayer, TetrisPiece

class CognitiveTetrisDemo:
    """Demostración del sistema cognitivo de Tetris en modo consola."""

    def __init__(self):
        self.board = TetrisBoard(10, 20)
        self.player = CognitiveTetrisPlayer(self.board)
        self.game_active = True
        self.moves_count = 0
        self.start_time = time.time()

        print("🧠 U-CogNet Cognitive Tetris Demo")
        print("=" * 50)
        print("🎮 Demostración de capacidades AGI en tiempo real")
        print("📊 Métricas cognitivas evaluadas continuamente")
        print()

    async def run_demo(self, max_moves: int = 50):
        """Ejecuta la demostración por un número limitado de movimientos."""

        print(f"🚀 Iniciando demostración con {max_moves} movimientos cognitivos...")
        print()

        while self.game_active and self.moves_count < max_moves:
            # Generar nueva pieza
            self._spawn_new_piece()

            if not self.board.is_valid_position(self.board.current_piece):
                print("💀 Game Over - Tablero lleno")
                break

            # Mostrar estado actual
            self._display_game_state()

            # Hacer movimiento cognitivo
            print(f"\n🤔 Movimiento {self.moves_count + 1}: Pensando...")
            decision = await self.player.make_move(self.board.current_piece)

            # Aplicar decisión
            action = decision['action']
            if action['type'] == 'no_valid_moves':
                print("❌ No hay movimientos válidos disponibles")
                break

            # Aplicar rotación y posición
            self.board.current_piece = TetrisPiece(self.board.current_piece.shape_type)
            for _ in range(action.get('rotation', 0)):
                self.board.current_piece.rotate()

            self.board.current_piece.x = action.get('position', self.board.width // 2)

            # Encontrar posición final
            while self.board.is_valid_position(self.board.current_piece, 0, 1):
                self.board.current_piece.y += 1

            # Colocar pieza
            self.board.place_piece(self.board.current_piece)
            lines_cleared = self.board.clear_lines()

            # Mostrar resultados
            self._display_move_results(decision, lines_cleared)

            self.moves_count += 1

            # Pequeña pausa para observar
            await asyncio.sleep(0.5)

        # Mostrar resumen final
        self._display_final_summary()

    def _spawn_new_piece(self):
        """Genera una nueva pieza."""
        piece_type = np.random.choice(list(TetrisPiece.SHAPES.keys()))
        self.board.current_piece = self.board.spawn_piece(piece_type)
        self.board.current_piece.x = self.board.width // 2 - len(self.board.current_piece.shape[0]) // 2
        self.board.current_piece.y = 0

    def _display_game_state(self):
        """Muestra el estado actual del tablero."""

        print(f"\n📋 Estado del Juego - Movimiento {self.moves_count + 1}")
        print(f"   Score: {self.board.score} | Líneas: {self.board.lines_cleared} | Nivel: {self.board.level}")

        # Mostrar tablero simplificado
        print("   Tablero actual:")
        for y in range(min(10, self.board.height)):  # Mostrar solo las primeras 10 filas
            row_str = "   "
            for x in range(self.board.width):
                if self.board.board[y][x]:
                    row_str += "█"
                else:
                    row_str += " "
            print(row_str)

        # Pieza actual
        piece_name = self.board.current_piece.shape_type
        print(f"   Pieza actual: {piece_name}")

    def _display_move_results(self, decision: dict, lines_cleared: int):
        """Muestra los resultados del movimiento cognitivo."""

        action = decision['action']
        metrics = decision['metrics']

        print("✅ Movimiento completado:")
        print(f"   Rotación: {action.get('rotation', 0)} | Posición: {action.get('position', 'N/A')}")
        print(f"   Líneas limpiadas: {lines_cleared}")

        print("🧠 Métricas cognitivas:")
        print(".3f")
        print(".3f")
        print(".3f")
        print(f"   Creatividad aplicada: {'Sí' if metrics.get('creativity_applied', False) else 'No'}")
        print(f"   Confianza razonamiento: {metrics.get('reasoning_confidence', 0):.3f}")

        # Mostrar razonamiento si está disponible
        reasoning = decision.get('reasoning', {})
        if reasoning.get('board_analysis'):
            analysis = reasoning['board_analysis']
            print("📊 Análisis del tablero:")
            print(".3f")
            print(f"   Patrones problemáticos: {len(analysis.get('problematic_patterns', []))}")

        if reasoning.get('creative_insights'):
            insights = reasoning['creative_insights']
            if insights:
                print(f"💡 Insights creativos generados: {len(insights)}")

    def _display_final_summary(self):
        """Muestra el resumen final de la demostración."""

        duration = time.time() - self.start_time
        real_time_metrics = self.player.get_real_time_metrics()

        print("\n" + "=" * 50)
        print("🏁 DEMOSTRACIÓN COMPLETADA")
        print("=" * 50)

        print("📊 Estadísticas Finales:")
        print(f"   Duración: {duration:.1f} segundos")
        print(f"   Movimientos realizados: {self.moves_count}")
        print(f"   Score final: {self.board.score}")
        print(f"   Líneas totales: {self.board.lines_cleared}")
        print(f"   Nivel alcanzado: {self.board.level}")

        print("\n🧠 Estado Cognitivo Final:")
        cog = real_time_metrics['cognitive_metrics']
        print(".3f")
        print(f"   Patrones aprendidos: {cog['patterns_learned']}")
        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")

        print("\n🎯 Métricas AGI:")
        adapt = real_time_metrics['learning_metrics']['adaptability_score']
        reason = real_time_metrics['reasoning_metrics']['avg_confidence']
        learn = len(self.player.adaptive_learning['success_rates']) / 50.0

        print(".3f")
        print(".3f")
        print(".3f")
        overall_agi = (adapt + reason + learn) / 3.0
        print(".3f")

        # Guardar resultados
        self._save_demo_results(real_time_metrics, duration)

        print("\n💾 Resultados guardados en 'cognitive_tetris_demo_results/'")

    def _save_demo_results(self, metrics: dict, duration: float):
        """Guarda los resultados de la demostración."""

        output_dir = Path("cognitive_tetris_demo_results")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"demo_session_{timestamp}.json"

        results = {
            'session_info': {
                'timestamp': timestamp,
                'duration': duration,
                'moves_completed': self.moves_count,
                'final_score': self.board.score,
                'final_lines': self.board.lines_cleared,
                'final_level': self.board.level
            },
            'cognitive_metrics': metrics['cognitive_metrics'],
            'performance_metrics': metrics['performance_metrics'],
            'learning_metrics': metrics['learning_metrics'],
            'creativity_metrics': metrics['creativity_metrics'],
            'reasoning_metrics': metrics['reasoning_metrics'],
            'agi_evaluation': {
                'adaptability': metrics['learning_metrics']['adaptability_score'],
                'reasoning_quality': metrics['reasoning_metrics']['avg_confidence'],
                'learning_efficiency': len(self.player.adaptive_learning['success_rates']) / 50.0,
                'overall_score': (metrics['learning_metrics']['adaptability_score'] +
                                metrics['reasoning_metrics']['avg_confidence'] +
                                len(self.player.adaptive_learning['success_rates']) / 50.0) / 3.0
            }
        }

        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            import json
            json.dump(results, f, indent=2, ensure_ascii=False)

async def main():
    """Función principal de la demostración."""

    print("🎮 Cognitive Tetris Demo - U-CogNet")
    print("Sistema cognitivo completo evaluado en tiempo real")
    print()

    # Preguntar número de movimientos
    try:
        moves = input("¿Cuántos movimientos cognitivos deseas ver? (default: 20): ").strip()
        max_moves = int(moves) if moves else 20
    except ValueError:
        max_moves = 20

    print(f"\n🚀 Ejecutando demostración con {max_moves} movimientos...")
    print("Observa cómo el sistema piensa, aprende y toma decisiones creativas")
    print("-" * 60)

    demo = CognitiveTetrisDemo()
    await demo.run_demo(max_moves)

if __name__ == "__main__":
    asyncio.run(main())