#!/usr/bin/env python3
"""
U-CogNet Tetris Environment
Entorno de Tetris integrado con sistema cognitivo completo
Métricas en tiempo real y evaluación AGI continua
"""

import pygame
import random
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import asyncio
import time
from datetime import datetime
import json
from pathlib import Path
import sys

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from ucognet import AudioCognitiveProcessor, CognitiveCore, SemanticFeedback

class TetrisPiece:
    """Representa una pieza de Tetris."""

    SHAPES = {
        'I': [[1, 1, 1, 1]],
        'O': [[1, 1], [1, 1]],
        'T': [[0, 1, 0], [1, 1, 1]],
        'S': [[0, 1, 1], [1, 1, 0]],
        'Z': [[1, 1, 0], [0, 1, 1]],
        'J': [[1, 0, 0], [1, 1, 1]],
        'L': [[0, 0, 1], [1, 1, 1]]
    }

    COLORS = {
        'I': (0, 255, 255),    # Cyan
        'O': (255, 255, 0),    # Yellow
        'T': (128, 0, 128),    # Purple
        'S': (0, 255, 0),      # Green
        'Z': (255, 0, 0),      # Red
        'J': (0, 0, 255),      # Blue
        'L': (255, 165, 0)     # Orange
    }

    def __init__(self, shape_type: str):
        self.shape_type = shape_type
        self.shape = [row[:] for row in self.SHAPES[shape_type]]
        self.color = self.COLORS[shape_type]
        self.x = 0
        self.y = 0

    def rotate(self):
        """Rota la pieza 90 grados en sentido horario."""
        self.shape = list(zip(*self.shape[::-1]))
        self.shape = [list(row) for row in self.shape]

class TetrisBoard:
    """Tablero de Tetris con lógica de juego."""

    def __init__(self, width: int = 10, height: int = 20):
        self.width = width
        self.height = height
        self.board = [[0 for _ in range(width)] for _ in range(height)]
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.game_over = False

        # Métricas cognitivas
        self.moves_history = []
        self.thinking_time = []
        self.adaptive_decisions = []
        self.pattern_recognition = []

    def spawn_piece(self, piece_type: str = None) -> TetrisPiece:
        """Genera una nueva pieza."""
        if piece_type is None:
            piece_type = random.choice(list(TetrisPiece.SHAPES.keys()))

        piece = TetrisPiece(piece_type)
        piece.x = self.width // 2 - len(piece.shape[0]) // 2
        piece.y = 0

        return piece

    def is_valid_position(self, piece: TetrisPiece, dx: int = 0, dy: int = 0) -> bool:
        """Verifica si una posición es válida para la pieza."""
        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = piece.x + x + dx
                    new_y = piece.y + y + dy

                    if (new_x < 0 or new_x >= self.width or
                        new_y >= self.height or
                        (new_y >= 0 and self.board[new_y][new_x])):
                        return False
        return True

    def place_piece(self, piece: TetrisPiece):
        """Coloca la pieza en el tablero."""
        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    board_y = piece.y + y
                    board_x = piece.x + x
                    if 0 <= board_y < self.height and 0 <= board_x < self.width:
                        self.board[board_y][board_x] = 1

    def clear_lines(self) -> int:
        """Limpia líneas completas y devuelve el número de líneas limpiadas."""
        lines_to_clear = []
        for y in range(self.height):
            if all(self.board[y]):
                lines_to_clear.append(y)

        for y in lines_to_clear[::-1]:
            del self.board[y]
            self.board.insert(0, [0 for _ in range(self.width)])

        lines_cleared = len(lines_to_clear)
        self.lines_cleared += lines_cleared

        # Actualizar score y level
        self.score += lines_cleared * 100 * self.level
        self.level = self.lines_cleared // 10 + 1

        return lines_cleared

    def get_board_state(self) -> np.ndarray:
        """Obtiene el estado actual del tablero como array numpy."""
        state = np.array(self.board, dtype=np.float32)

        # Añadir pieza actual si existe
        if self.current_piece:
            for y, row in enumerate(self.current_piece.shape):
                for x, cell in enumerate(row):
                    if cell:
                        board_y = self.current_piece.y + y
                        board_x = self.current_piece.x + x
                        if 0 <= board_y < self.height and 0 <= board_x < self.width:
                            state[board_y][board_x] = 2  # Pieza actual

        return state

    def get_game_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas actuales del juego."""
        # Calcular métricas del tablero
        holes = 0
        height_variance = []
        column_heights = []

        for x in range(self.width):
            column_height = 0
            column_holes = 0
            found_block = False

            for y in range(self.height):
                if self.board[y][x]:
                    found_block = True
                    column_height = max(column_height, self.height - y)
                elif found_block:
                    column_holes += 1

            holes += column_holes
            column_heights.append(column_height)

        if column_heights:
            height_variance.append(np.var(column_heights))

        return {
            'score': self.score,
            'lines_cleared': self.lines_cleared,
            'level': self.level,
            'holes': holes,
            'height_variance': height_variance[0] if height_variance else 0,
            'avg_column_height': np.mean(column_heights) if column_heights else 0,
            'max_column_height': max(column_heights) if column_heights else 0
        }

class CognitiveTetrisPlayer:
    """
    Jugador de Tetris con capacidades cognitivas U-CogNet.
    Integra razonamiento, aprendizaje adaptativo y toma de decisiones.
    """

    def __init__(self, board: TetrisBoard):
        self.board = board

        # Sistema cognitivo
        self.cognitive_core = CognitiveCore(buffer_size=100)
        self.semantic_feedback = SemanticFeedback()
        self.audio_processor = AudioCognitiveProcessor(
            cognitive_core=self.cognitive_core,
            semantic_feedback=self.semantic_feedback
        )

        # Memoria de juego
        self.game_memory = []
        self.strategy_patterns = {}
        self.adaptive_learning = {
            'rotation_preference': {},
            'position_preference': {},
            'timing_patterns': [],
            'success_rates': {}
        }

        # Estado cognitivo
        self.thinking_mode = False
        self.decision_history = []
        self.creativity_score = 0.0
        self.adaptability_score = 0.0

        # Métricas en tiempo real
        self.real_time_metrics = {
            'thinking_time': [],
            'decision_quality': [],
            'pattern_recognition': [],
            'adaptive_learning': [],
            'creativity_index': [],
            'reasoning_confidence': []
        }

    async def make_move(self, current_piece: TetrisPiece, next_piece: TetrisPiece = None) -> Dict[str, Any]:
        """
        Toma una decisión de movimiento usando capacidades cognitivas.
        Retorna acción y métricas cognitivas.
        """

        start_time = time.time()
        self.thinking_mode = True

        try:
            # 1. ANALIZAR ESTADO ACTUAL
            board_state = self.board.get_board_state()
            game_metrics = self.board.get_game_metrics()

            # 2. RAZONAMIENTO COGNITIVO
            reasoning = await self._cognitive_reasoning(current_piece, next_piece, board_state, game_metrics)

            # 3. GENERAR ACCIONES POSIBLES
            possible_actions = self._generate_possible_actions(current_piece)

            # 4. EVALUAR ACCIONES
            action_evaluation = await self._evaluate_actions(possible_actions, current_piece, board_state, game_metrics)

            # 5. TOMAR DECISIÓN ADAPTATIVA
            best_action = self._select_adaptive_action(action_evaluation, reasoning)

            # 6. APRENDER DE LA DECISIÓN
            await self._learn_from_decision(best_action, reasoning, game_metrics)

            # 7. CALCULAR MÉTRICAS EN TIEMPO REAL
            thinking_time = time.time() - start_time
            decision_metrics = self._calculate_decision_metrics(best_action, reasoning, thinking_time)

            # Actualizar métricas en tiempo real
            self._update_real_time_metrics(decision_metrics)

            self.thinking_mode = False

            return {
                'action': best_action,
                'reasoning': reasoning,
                'metrics': decision_metrics,
                'cognitive_state': self._get_cognitive_state()
            }

        except Exception as e:
            self.thinking_mode = False
            # Fallback a movimiento aleatorio
            return {
                'action': {'type': 'random', 'rotation': 0, 'position': self.board.width // 2},
                'reasoning': {'error': str(e)},
                'metrics': {'thinking_time': time.time() - start_time, 'fallback': True},
                'cognitive_state': self._get_cognitive_state()
            }

    async def _cognitive_reasoning(self, current_piece: TetrisPiece, next_piece: TetrisPiece,
                                 board_state: np.ndarray, game_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza razonamiento cognitivo sobre la situación de juego."""

        # Analizar patrón del tablero
        board_analysis = self._analyze_board_patterns(board_state)

        # Predecir consecuencias
        future_predictions = self._predict_future_states(current_piece, next_piece, board_state)

        # Evaluar riesgos y oportunidades
        risk_assessment = self._assess_risks_and_opportunities(game_metrics, board_analysis)

        # Generar insights creativos
        creative_insights = await self._generate_creative_insights(current_piece, board_analysis)

        return {
            'board_analysis': board_analysis,
            'future_predictions': future_predictions,
            'risk_assessment': risk_assessment,
            'creative_insights': creative_insights,
            'confidence': self._calculate_reasoning_confidence(board_analysis, risk_assessment)
        }

    def _analyze_board_patterns(self, board_state: np.ndarray) -> Dict[str, Any]:
        """Analiza patrones en el estado del tablero."""

        # Calcular características del tablero
        flat_board = board_state.flatten()
        occupied_cells = np.sum(flat_board > 0)
        total_cells = flat_board.size
        density = occupied_cells / total_cells

        # Analizar estructura vertical
        column_densities = []
        for x in range(board_state.shape[1]):
            column = board_state[:, x]
            column_density = np.sum(column > 0) / len(column)
            column_densities.append(column_density)

        # Detectar patrones problemáticos
        problematic_patterns = self._detect_problematic_patterns(board_state)

        return {
            'density': density,
            'column_densities': column_densities,
            'problematic_patterns': problematic_patterns,
            'structural_score': self._calculate_structural_score(column_densities, problematic_patterns)
        }

    def _detect_problematic_patterns(self, board_state: np.ndarray) -> List[str]:
        """Detecta patrones problemáticos en el tablero."""

        patterns = []

        # Huecos altos (difíciles de llenar)
        for x in range(board_state.shape[1]):
            column = board_state[:, x]
            occupied_indices = np.where(column > 0)[0]

            if len(occupied_indices) > 0:
                top_occupied = occupied_indices[0]
                # Buscar huecos en la parte superior
                for y in range(top_occupied, min(top_occupied + 4, board_state.shape[0])):
                    if board_state[y, x] == 0:
                        patterns.append(f'hole_col_{x}_row_{y}')
                        break

        # Paredes irregulares
        heights = []
        for x in range(board_state.shape[1]):
            column = board_state[:, x]
            occupied_indices = np.where(column > 0)[0]
            height = board_state.shape[0] - occupied_indices[0] if len(occupied_indices) > 0 else 0
            heights.append(height)

        if len(heights) > 1:
            height_variance = np.var(heights)
            if height_variance > 2.0:
                patterns.append('irregular_walls')

        return patterns

    def _calculate_structural_score(self, column_densities: List[float],
                                  problematic_patterns: List[str]) -> float:
        """Calcula score de calidad estructural del tablero."""

        # Penalizar densidades irregulares
        density_variance = np.var(column_densities)
        irregularity_penalty = min(1.0, density_variance * 2)

        # Penalizar patrones problemáticos
        pattern_penalty = min(1.0, len(problematic_patterns) * 0.2)

        # Bonus por densidades balanceadas
        balanced_bonus = 1.0 - irregularity_penalty

        return max(0.0, balanced_bonus - pattern_penalty)

    def _predict_future_states(self, current_piece: TetrisPiece, next_piece: TetrisPiece,
                             board_state: np.ndarray) -> Dict[str, Any]:
        """Predice estados futuros del juego."""

        predictions = {
            'immediate_impact': self._predict_piece_impact(current_piece, board_state),
            'next_piece_setup': None,
            'long_term_consequences': []
        }

        if next_piece:
            predictions['next_piece_setup'] = self._analyze_next_piece_setup(next_piece, board_state)

        # Predecir consecuencias a largo plazo
        predictions['long_term_consequences'] = self._predict_long_term_consequences(board_state)

        return predictions

    def _predict_piece_impact(self, piece: TetrisPiece, board_state: np.ndarray) -> Dict[str, Any]:
        """Predice el impacto de colocar una pieza."""

        # Simular colocación en diferentes posiciones
        impact_scores = {}

        for rotation in range(4):
            test_piece = TetrisPiece(piece.shape_type)
            for _ in range(rotation):
                test_piece.rotate()

            for x_pos in range(-len(test_piece.shape[0]) + 1, self.board.width):
                test_piece.x = x_pos
                test_piece.y = 0

                # Encontrar posición de caída
                while self.board.is_valid_position(test_piece, 0, 1):
                    test_piece.y += 1

                if self.board.is_valid_position(test_piece):
                    # Calcular score de esta posición
                    lines_cleared = self._simulate_lines_cleared(test_piece, board_state)
                    structural_impact = self._calculate_structural_impact(test_piece, board_state)

                    impact_scores[f'rot{rotation}_pos{x_pos}'] = {
                        'lines_cleared': lines_cleared,
                        'structural_impact': structural_impact,
                        'total_score': lines_cleared * 10 + structural_impact
                    }

        best_position = max(impact_scores.items(), key=lambda x: x[1]['total_score']) if impact_scores else None

        return {
            'possible_positions': len(impact_scores),
            'best_position': best_position[0] if best_position else None,
            'best_score': best_position[1]['total_score'] if best_position else 0,
            'impact_distribution': list(impact_scores.values())[:5]  # Top 5
        }

    def _simulate_lines_cleared(self, piece: TetrisPiece, board_state: np.ndarray) -> int:
        """Simula cuántas líneas se limpiarían con esta pieza."""

        # Crear copia del tablero
        temp_board = board_state.copy()

        # Colocar pieza
        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    board_y = piece.y + y
                    board_x = piece.x + x
                    if 0 <= board_y < temp_board.shape[0] and 0 <= board_x < temp_board.shape[1]:
                        temp_board[board_y, board_x] = 2

        # Contar líneas completas
        lines_cleared = 0
        for y in range(temp_board.shape[0]):
            if np.all(temp_board[y, :] > 0):
                lines_cleared += 1

        return lines_cleared

    def _calculate_structural_impact(self, piece: TetrisPiece, board_state: np.ndarray) -> float:
        """Calcula el impacto estructural de colocar una pieza."""

        # Simular tablero después de colocar
        temp_board = board_state.copy()

        # Colocar pieza
        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    board_y = piece.y + y
                    board_x = piece.x + x
                    if 0 <= board_y < temp_board.shape[0] and 0 <= board_x < temp_board.shape[1]:
                        temp_board[board_y, board_x] = 2

        # Calcular métricas estructurales
        holes = 0
        column_heights = []

        for x in range(temp_board.shape[1]):
            height = 0
            found_top = False

            for y in range(temp_board.shape[0]):
                if temp_board[y, x] > 0:
                    if not found_top:
                        found_top = True
                    height = temp_board.shape[0] - y
                elif found_top:
                    holes += 1

            column_heights.append(height)

        # Score basado en estructura
        avg_height = np.mean(column_heights)
        height_variance = np.var(column_heights)
        structural_score = 10.0 - (holes * 2) - (height_variance * 0.5) - (avg_height * 0.1)

        return max(-10.0, min(10.0, structural_score))

    def _assess_risks_and_opportunities(self, game_metrics: Dict[str, Any],
                                       board_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Evalúa riesgos y oportunidades."""

        risks = []
        opportunities = []

        # Evaluar riesgos
        if game_metrics['holes'] > 5:
            risks.append({'type': 'many_holes', 'severity': 'high', 'description': 'Múltiples huecos difíciles de llenar'})

        if game_metrics['height_variance'] > 3.0:
            risks.append({'type': 'irregular_heights', 'severity': 'medium', 'description': 'Paredes irregulares'})

        if game_metrics['max_column_height'] > 15:
            risks.append({'type': 'high_stack', 'severity': 'critical', 'description': 'Apilamiento peligroso'})

        # Evaluar oportunidades
        if board_analysis['structural_score'] > 0.7:
            opportunities.append({'type': 'good_structure', 'potential': 'high', 'description': 'Estructura favorable'})

        if len(board_analysis['problematic_patterns']) == 0:
            opportunities.append({'type': 'clean_board', 'potential': 'high', 'description': 'Tablero limpio'})

        return {
            'risks': risks,
            'opportunities': opportunities,
            'net_assessment': len(opportunities) - len(risks),
            'risk_level': 'low' if len(risks) == 0 else 'high' if len(risks) > 2 else 'medium'
        }

    async def _generate_creative_insights(self, piece: TetrisPiece,
                                        board_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera insights creativos sobre estrategias de juego."""

        insights = []

        # Analizar pieza actual
        piece_analysis = self._analyze_piece_characteristics(piece)

        # Generar ideas creativas basadas en el contexto
        if board_analysis['structural_score'] < 0.5:
            insights.append({
                'type': 'structural_fix',
                'creativity': 0.7,
                'description': f'Usar {piece.shape_type} para corregir estructura irregular',
                'action': 'fill_holes'
            })

        if piece_analysis['complexity'] > 0.7:
            insights.append({
                'type': 'complex_manipulation',
                'creativity': 0.8,
                'description': f'Aprovechar complejidad de {piece.shape_type} para maniobras avanzadas',
                'action': 'creative_placement'
            })

        # Insights basados en memoria de juego
        if self.game_memory:
            recent_successes = [m for m in self.game_memory[-5:] if m.get('success', False)]
            if recent_successes:
                last_success = recent_successes[-1]
                insights.append({
                    'type': 'pattern_learning',
                    'creativity': 0.6,
                    'description': f'Aplicar patrón exitoso anterior: {last_success.get("pattern", "unknown")}',
                    'action': 'learned_strategy'
                })

        return insights

    def _analyze_piece_characteristics(self, piece: TetrisPiece) -> Dict[str, Any]:
        """Analiza características de una pieza."""

        # Calcular complejidad de forma
        shape_complexity = len(piece.shape) * len(piece.shape[0]) / 16.0  # Normalizado

        # Calcular simetría
        symmetry_score = self._calculate_symmetry(piece.shape)

        # Calcular versatilidad (posiciones posibles)
        positions_possible = 0
        for rotation in range(4):
            test_piece = TetrisPiece(piece.shape_type)
            for _ in range(rotation):
                test_piece.rotate()
            positions_possible += self.board.width - len(test_piece.shape[0]) + 1

        versatility = min(1.0, positions_possible / 40.0)  # Normalizado

        return {
            'complexity': shape_complexity,
            'symmetry': symmetry_score,
            'versatility': versatility,
            'overall_score': (shape_complexity + symmetry_score + versatility) / 3.0
        }

    def _calculate_symmetry(self, shape: List[List[int]]) -> float:
        """Calcula simetría de una forma."""

        if not shape or not shape[0]:
            return 0.0

        rows, cols = len(shape), len(shape[0])
        symmetry_score = 0.0

        # Simetría horizontal
        for row in shape:
            for i in range(cols // 2):
                if row[i] == row[cols - 1 - i]:
                    symmetry_score += 1.0

        # Simetría vertical
        for col in range(cols):
            for i in range(rows // 2):
                if shape[i][col] == shape[rows - 1 - i][col]:
                    symmetry_score += 1.0

        total_possible = (rows * (cols // 2)) + (cols * (rows // 2))
        return symmetry_score / total_possible if total_possible > 0 else 0.0

    def _generate_possible_actions(self, piece: TetrisPiece) -> List[Dict[str, Any]]:
        """Genera todas las acciones posibles para una pieza."""

        actions = []

        for rotation in range(4):
            test_piece = TetrisPiece(piece.shape_type)
            for _ in range(rotation):
                test_piece.rotate()

            for x_pos in range(-len(test_piece.shape[0]) + 1, self.board.width):
                test_piece.x = x_pos
                test_piece.y = 0

                # Encontrar posición final
                while self.board.is_valid_position(test_piece, 0, 1):
                    test_piece.y += 1

                if self.board.is_valid_position(test_piece):
                    actions.append({
                        'rotation': rotation,
                        'position': x_pos,
                        'final_y': test_piece.y,
                        'piece_type': piece.shape_type
                    })

        return actions

    async def _evaluate_actions(self, actions: List[Dict[str, Any]], piece: TetrisPiece,
                              board_state: np.ndarray, game_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evalúa todas las acciones posibles."""

        evaluated_actions = []

        for action in actions:
            # Crear pieza en la posición evaluada
            test_piece = TetrisPiece(piece.shape_type)
            for _ in range(action['rotation']):
                test_piece.rotate()

            test_piece.x = action['position']
            test_piece.y = action['final_y']

            # Calcular score de la acción
            action_score = self._calculate_action_score(test_piece, board_state, game_metrics, action)

            evaluated_actions.append({
                **action,
                'score': action_score,
                'evaluation': self._categorize_action_quality(action_score)
            })

        # Ordenar por score
        evaluated_actions.sort(key=lambda x: x['score'], reverse=True)

        return evaluated_actions

    def _calculate_action_score(self, piece: TetrisPiece, board_state: np.ndarray,
                              game_metrics: Dict[str, Any], action: Dict[str, Any]) -> float:
        """Calcula el score de una acción específica."""

        # Factor 1: Líneas limpiadas
        lines_cleared = self._simulate_lines_cleared(piece, board_state)
        lines_score = lines_cleared * 100

        # Factor 2: Impacto estructural
        structural_impact = self._calculate_structural_impact(piece, board_state)
        structural_score = structural_impact * 20

        # Factor 3: Posición estratégica
        position_score = self._evaluate_position_strategic_value(piece, action)

        # Factor 4: Preparación para siguiente pieza
        setup_score = self._evaluate_setup_quality(piece, board_state)

        # Factor 5: Aprendizaje adaptativo
        learning_score = self._apply_adaptive_learning(action, game_metrics)

        total_score = (lines_score + structural_score + position_score +
                      setup_score + learning_score)

        return total_score

    def _evaluate_position_strategic_value(self, piece: TetrisPiece, action: Dict[str, Any]) -> float:
        """Evalúa el valor estratégico de una posición."""

        strategic_score = 0.0

        # Bonus por posiciones centrales (más versátiles)
        center_distance = abs(action['position'] - self.board.width // 2)
        strategic_score += (self.board.width // 2 - center_distance) * 2

        # Bonus por alturas equilibradas
        final_height = action['final_y'] + len(piece.shape)
        if final_height < self.board.height * 0.8:  # No demasiado alto
            strategic_score += 10

        # Penalización por posiciones extremas
        if action['position'] <= 0 or action['position'] >= self.board.width - len(piece.shape[0]):
            strategic_score -= 15

        return strategic_score

    def _evaluate_setup_quality(self, piece: TetrisPiece, board_state: np.ndarray) -> float:
        """Evalúa qué tan bien prepara el tablero para la siguiente pieza."""

        # Simular tablero después de colocar
        temp_board = board_state.copy()

        # Colocar pieza
        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    board_y = piece.y + y
                    board_x = piece.x + x
                    if 0 <= board_y < temp_board.shape[0] and 0 <= board_x < temp_board.shape[1]:
                        temp_board[board_y, board_x] = 2

        # Evaluar "jugabilidad" del tablero resultante
        playable_positions = 0

        # Para una pieza típica (I-piece), contar posiciones válidas
        test_piece = TetrisPiece('I')

        for rotation in range(4):
            rot_piece = TetrisPiece('I')
            for _ in range(rotation):
                rot_piece.rotate()

            for x_pos in range(-len(rot_piece.shape[0]) + 1, temp_board.shape[1]):
                rot_piece.x = x_pos
                rot_piece.y = 0

                # Encontrar posición de caída
                while self._is_valid_position_on_temp_board(rot_piece, temp_board, 0, 1):
                    rot_piece.y += 1

                if self._is_valid_position_on_temp_board(rot_piece, temp_board):
                    playable_positions += 1

        # Normalizar score
        max_possible_positions = 4 * temp_board.shape[1]  # 4 rotaciones × ancho del tablero
        setup_score = (playable_positions / max_possible_positions) * 50

        return setup_score

    def _is_valid_position_on_temp_board(self, piece: TetrisPiece, temp_board: np.ndarray,
                                       dx: int = 0, dy: int = 0) -> bool:
        """Verifica posición válida en tablero temporal."""

        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = piece.x + x + dx
                    new_y = piece.y + y + dy

                    if (new_x < 0 or new_x >= temp_board.shape[1] or
                        new_y >= temp_board.shape[0] or
                        (new_y >= 0 and temp_board[new_y, new_x] > 0)):
                        return False
        return True

    def _apply_adaptive_learning(self, action: Dict[str, Any], game_metrics: Dict[str, Any]) -> float:
        """Aplica aprendizaje adaptativo a la evaluación de acciones."""

        learning_score = 0.0

        # Bonus por acciones que han funcionado antes
        action_key = f"{action['piece_type']}_rot{action['rotation']}_pos{action['position']}"
        if action_key in self.adaptive_learning['success_rates']:
            success_rate = self.adaptive_learning['success_rates'][action_key]
            learning_score += success_rate * 30

        # Bonus por patrones de timing aprendidos
        current_level = game_metrics['level']
        if current_level in self.adaptive_learning['timing_patterns']:
            timing_bonus = self.adaptive_learning['timing_patterns'][current_level]
            learning_score += timing_bonus

        # Exploración vs explotación
        exploration_factor = min(0.3, len(self.game_memory) / 1000.0)  # Más exploración al inicio
        learning_score += np.random.normal(0, exploration_factor * 10)

        return learning_score

    def _select_adaptive_action(self, action_evaluation: List[Dict[str, Any]],
                              reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """Selecciona la mejor acción usando criterios adaptativos."""

        if not action_evaluation:
            return {'type': 'no_valid_moves', 'rotation': 0, 'position': self.board.width // 2}

        # Factor de confianza del razonamiento
        confidence_factor = reasoning.get('confidence', 0.5)

        # Si confianza alta, elegir la mejor acción
        if confidence_factor > 0.8:
            return action_evaluation[0]

        # Si confianza media, considerar creatividad
        elif confidence_factor > 0.6:
            # Mezclar mejor acción con insights creativos
            best_action = action_evaluation[0].copy()
            creative_modifier = self._apply_creative_modifier(best_action, reasoning)
            return {**best_action, **creative_modifier}

        # Si confianza baja, ser más conservador
        else:
            # Elegir acción con buen balance riesgo/recompensa
            conservative_actions = [a for a in action_evaluation[:5] if a['score'] > -50]
            return conservative_actions[0] if conservative_actions else action_evaluation[0]

    def _apply_creative_modifier(self, action: Dict[str, Any], reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica modificadores creativos a una acción."""

        modifier = {}

        creative_insights = reasoning.get('creative_insights', [])
        if creative_insights:
            # Aplicar el insight más creativo
            best_insight = max(creative_insights, key=lambda x: x.get('creativity', 0))

            if best_insight['action'] == 'fill_holes':
                # Intentar posición que llene huecos
                modifier['creative_adjustment'] = 'hole_filling'
                modifier['creativity_boost'] = 0.2

            elif best_insight['action'] == 'creative_placement':
                # Añadir rotación extra para complejidad
                modifier['rotation'] = (action['rotation'] + 1) % 4
                modifier['creativity_boost'] = 0.3

        return modifier

    async def _learn_from_decision(self, action: Dict[str, Any], reasoning: Dict[str, Any],
                                 game_metrics: Dict[str, Any]):
        """Aprende de la decisión tomada."""

        # Registrar en memoria de juego
        game_state = {
            'action': action,
            'reasoning': reasoning,
            'game_metrics': game_metrics,
            'timestamp': datetime.now().isoformat(),
            'creativity_applied': action.get('creativity_boost', 0) > 0
        }

        self.game_memory.append(game_state)

        # Limitar memoria
        if len(self.game_memory) > 100:
            self.game_memory = self.game_memory[-100:]

        # Actualizar aprendizaje adaptativo
        self._update_adaptive_learning(action, reasoning, game_metrics)

        # Interiorizar en sistema cognitivo
        await self.audio_processor._interiorize_audio(
            type('MockAudio', (), {
                'waveform': np.random.randn(1000),
                'sample_rate': 22050,
                'duration': 0.045,
                'source': f'tetris_decision_{len(self.game_memory)}',
                'timestamp': time.time()
            })(),
            type('MockReasoning', (), {
                'event_type': 'tetris_decision',
                'confidence': reasoning.get('confidence', 0.5),
                'cognitive_insights': [i.get('description', '') for i in reasoning.get('creative_insights', [])]
            })()
        )

    def _update_adaptive_learning(self, action: Dict[str, Any], reasoning: Dict[str, Any],
                                game_metrics: Dict[str, Any]):
        """Actualiza el aprendizaje adaptativo."""

        action_key = f"{action['piece_type']}_rot{action['rotation']}_pos{action['position']}"

        # Actualizar tasas de éxito
        if action_key not in self.adaptive_learning['success_rates']:
            self.adaptive_learning['success_rates'][action_key] = 0.5  # Inicial

        # Ajustar basado en resultado (simplificado)
        current_success = 1.0 if game_metrics['lines_cleared'] > 0 else 0.0
        old_rate = self.adaptive_learning['success_rates'][action_key]
        new_rate = (old_rate * 0.9) + (current_success * 0.1)  # Media móvil
        self.adaptive_learning['success_rates'][action_key] = new_rate

        # Actualizar patrones de timing
        level = game_metrics['level']
        if level not in self.adaptive_learning['timing_patterns']:
            self.adaptive_learning['timing_patterns'][level] = 0.0

        # Ajustar timing basado en rendimiento
        timing_adjustment = 0.1 if game_metrics['score'] > 100 else -0.05
        self.adaptive_learning['timing_patterns'][level] += timing_adjustment
        self.adaptive_learning['timing_patterns'][level] = max(-1.0, min(1.0,
            self.adaptive_learning['timing_patterns'][level]))

    def _calculate_decision_metrics(self, action: Dict[str, Any], reasoning: Dict[str, Any],
                                  thinking_time: float) -> Dict[str, Any]:
        """Calcula métricas de la decisión tomada."""

        return {
            'thinking_time': thinking_time,
            'decision_quality': action.get('score', 0) / 100.0,  # Normalizado
            'creativity_applied': action.get('creativity_boost', 0) > 0,
            'reasoning_confidence': reasoning.get('confidence', 0.0),
            'pattern_recognition': len(reasoning.get('board_analysis', {}).get('problematic_patterns', [])),
            'adaptive_learning_used': action.get('piece_type', '') in self.adaptive_learning.get('success_rates', {}),
            'risk_assessment': reasoning.get('risk_assessment', {}).get('net_assessment', 0)
        }

    def _update_real_time_metrics(self, decision_metrics: Dict[str, Any]):
        """Actualiza métricas en tiempo real."""

        for key, value in decision_metrics.items():
            if key in self.real_time_metrics:
                self.real_time_metrics[key].append(value)
                # Mantener solo últimas 50 mediciones
                if len(self.real_time_metrics[key]) > 50:
                    self.real_time_metrics[key] = self.real_time_metrics[key][-50:]

    def _get_cognitive_state(self) -> Dict[str, Any]:
        """Obtiene el estado cognitivo actual."""

        return {
            'memory_size': len(self.game_memory),
            'patterns_learned': len(self.adaptive_learning['success_rates']),
            'creativity_avg': np.mean(self.real_time_metrics['creativity_index']) if self.real_time_metrics['creativity_index'] else 0.0,
            'thinking_time_avg': np.mean(self.real_time_metrics['thinking_time']) if self.real_time_metrics['thinking_time'] else 0.0,
            'decision_quality_avg': np.mean(self.real_time_metrics['decision_quality']) if self.real_time_metrics['decision_quality'] else 0.0,
            'cognitive_load': len(self.game_memory) / 100.0  # Normalizado
        }

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas en tiempo real para visualización."""

        return {
            'cognitive_metrics': self._get_cognitive_state(),
            'performance_metrics': self.board.get_game_metrics(),
            'learning_metrics': {
                'patterns_learned': len(self.adaptive_learning['success_rates']),
                'memory_utilization': len(self.game_memory) / 100.0,
                'adaptability_score': self._calculate_adaptability_score()
            },
            'creativity_metrics': {
                'creativity_index': np.mean(self.real_time_metrics['creativity_index'][-10:]) if self.real_time_metrics['creativity_index'] else 0.0,
                'innovative_decisions': sum(1 for m in self.real_time_metrics['creativity_index'][-20:] if m and m > 0.5) if self.real_time_metrics['creativity_index'] else 0
            },
            'reasoning_metrics': {
                'avg_confidence': np.mean([m for m in self.real_time_metrics['reasoning_confidence'] if m] or [0]),
                'pattern_recognition_rate': np.mean(self.real_time_metrics['pattern_recognition'][-20:]) if self.real_time_metrics['pattern_recognition'] else 0,
                'decision_speed': np.mean(self.real_time_metrics['thinking_time'][-20:]) if self.real_time_metrics['thinking_time'] else 0
            }
        }

    def _calculate_adaptability_score(self) -> float:
        """Calcula score de adaptabilidad basado en aprendizaje."""

        if not self.adaptive_learning['success_rates']:
            return 0.0

        # Promedio de tasas de éxito aprendidas
        avg_success_rate = np.mean(list(self.adaptive_learning['success_rates'].values()))

        # Diversidad de estrategias aprendidas
        strategy_diversity = len(self.adaptive_learning['success_rates']) / 50.0  # Normalizado

        # Adaptabilidad de timing
        timing_adaptability = len(self.adaptive_learning['timing_patterns']) / 10.0  # Normalizado

        return min(1.0, (avg_success_rate + strategy_diversity + timing_adaptability) / 3.0)