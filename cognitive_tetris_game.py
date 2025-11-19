#!/usr/bin/env python3
"""
Cognitive Tetris Game - Interfaz gráfica y loop principal
Sistema de visualización en tiempo real de métricas cognitivas
"""

import pygame
import asyncio
import sys
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import random
import math

# Configurar pygame para modo headless si es necesario
if 'SDL_VIDEODRIVER' not in os.environ:
    os.environ['SDL_VIDEODRIVER'] = 'x11'

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from cognitive_tetris import TetrisBoard, CognitiveTetrisPlayer, TetrisPiece

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from cognitive_tetris import TetrisBoard, CognitiveTetrisPlayer, TetrisPiece

class CognitiveTetrisGame:
    """Juego completo de Tetris con capacidades cognitivas y visualización en tiempo real."""

    def __init__(self):
        pygame.init()
        pygame.mixer.init()  # Inicializar sistema de audio

        # Configuración de pantalla
        self.screen_width = 1400
        self.screen_height = 900
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("U-CogNet Cognitive Tetris - Real-time AGI Metrics")

        # Inicializar joystick/gamepad
        pygame.joystick.init()
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"🎮 Gamepad detectado: {self.joystick.get_name()}")

        # Colores mejorados
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.PURPLE = (128, 0, 128)
        self.CYAN = (0, 255, 255)
        self.ORANGE = (255, 165, 0)
        self.DARK_BLUE = (0, 0, 128)
        self.DARK_GREEN = (0, 128, 0)

        # Fuentes mejoradas
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 36)
        self.font_title = pygame.font.Font(None, 48)
        self.font_bold = pygame.font.Font(None, 32)

        # Sistema de audio
        self.audio_enabled = True
        self.sound_effects = {}
        self.music_enabled = True
        self._load_audio()

        # Configuración del juego
        self.cell_size = 35
        self.board_width = 10
        self.board_height = 20
        self.board_offset_x = 80
        self.board_offset_y = 120

        # Componentes del juego
        self.board = TetrisBoard(self.board_width, self.board_height)
        self.player = CognitiveTetrisPlayer(self.board)

        # Estado del juego
        self.current_piece = None
        self.next_piece = None
        self.game_running = True
        self.paused = False
        self.game_over = False
        self.show_metrics = True
        self.auto_restart = True  # Reinicio automático activado
        self.senses_enabled = True  # Sentidos activados

        # Control de tiempo
        self.clock = pygame.time.Clock()
        self.fall_time = 0
        self.fall_speed = 500  # ms
        self.last_update = time.time()

        # Sistema de aprendizaje continuo
        self.session_start_time = time.time()
        self.total_games = 0
        self.best_score = 0
        self.learning_sessions = []

        # Métricas en tiempo real
        self.metrics_history = []
        self.game_start_time = time.time()

        # Efectos visuales
        self.particles = []
        self.screen_shake = 0
        self.flash_effect = 0

        # Botones con texturas
        self.buttons = {}
        self._create_buttons()

        # Inicializar juego
        self._spawn_new_piece()
        self._spawn_next_piece()
        self._play_background_music()

    def _load_audio(self):
        """Carga efectos de sonido y música."""
        try:
            # Crear sonidos sintéticos si no hay archivos
            self._create_synthetic_audio()
            print("🔊 Sistema de audio inicializado")
        except Exception as e:
            print(f"⚠️  Error cargando audio: {e}")
            self.audio_enabled = False

    def _create_synthetic_audio(self):
        """Crea efectos de sonido sintéticos."""
        # Frecuencias para diferentes sonidos
        frequencies = {
            'line_clear': 800,
            'piece_place': 400,
            'game_over': 200,
            'level_up': 600,
            'button_click': 1000
        }

        for sound_name, freq in frequencies.items():
            # Crear onda sinusoidal
            sample_rate = 44100
            duration = 0.3
            samples = int(sample_rate * duration)

            # Generar onda
            t = np.linspace(0, duration, samples, False)
            wave = np.sin(freq * 2 * np.pi * t)

            # Envelope para suavizar
            envelope = np.exp(-t * 8)
            wave = wave * envelope

            # Convertir a formato pygame
            wave = (wave * 32767).astype(np.int16)

            # Crear sonido
            sound = pygame.mixer.Sound(wave)
            self.sound_effects[sound_name] = sound

    def _play_background_music(self):
        """Reproduce música de fondo."""
        if not self.music_enabled or not self.audio_enabled:
            return

        try:
            # Crear música procedural simple
            sample_rate = 44100
            duration = 30  # 30 segundos
            samples = sample_rate * duration

            # Generar melodía simple
            t = np.linspace(0, duration, samples, False)

            # Notas musicales (frecuencias)
            notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]  # Do mayor

            # Crear secuencia melódica
            melody = np.zeros(samples)
            note_duration = sample_rate // 4  # negra

            for i in range(0, len(melody) - note_duration, note_duration):
                note = random.choice(notes)
                end_idx = min(i + note_duration, len(melody))
                segment_samples = end_idx - i

                t_segment = np.linspace(0, segment_samples / sample_rate, segment_samples, False)
                wave_segment = np.sin(note * 2 * np.pi * t_segment)
                envelope = np.exp(-t_segment * 2)
                melody[i:end_idx] = wave_segment * envelope * 0.3

            # Convertir y reproducir
            melody = (melody * 32767).astype(np.int16)
            music_sound = pygame.mixer.Sound(melody)

            # Reproducir en loop
            music_sound.play(loops=-1)
            music_sound.set_volume(0.2)

        except Exception as e:
            print(f"⚠️  Error creando música: {e}")

    def _create_buttons(self):
        """Crea botones con texturas para la interfaz."""
        button_width = 120
        button_height = 40
        button_y = self.screen_height - 60

        buttons_data = [
            ("Auto-Restart", self.screen_width - 500, button_y, self.auto_restart),
            ("Audio", self.screen_width - 370, button_y, self.audio_enabled),
            ("Sentidos", self.screen_width - 240, button_y, self.senses_enabled),
            ("Métricas", self.screen_width - 110, button_y, self.show_metrics)
        ]

        for name, x, y, state in buttons_data:
            self.buttons[name.lower().replace('-', '_')] = {
                'rect': pygame.Rect(x, y, button_width, button_height),
                'text': name,
                'state': state,
                'hover': False,
                'texture': self._create_button_texture(button_width, button_height, state)
            }

    def _create_button_texture(self, width: int, height: int, active: bool) -> pygame.Surface:
        """Crea textura para botones."""
        texture = pygame.Surface((width, height))
        texture.fill(self.DARK_BLUE if active else self.GRAY)

        # Bordes redondeados
        pygame.draw.rect(texture, self.BLUE if active else self.WHITE, (0, 0, width, height), 2, border_radius=5)

        # Efecto de gradiente
        for i in range(height):
            alpha = 255 - (i * 100 // height)
            color = (0, 100, 200, alpha) if active else (150, 150, 150, alpha)
            pygame.draw.line(texture, color, (0, i), (width, i))

        return texture

    def _play_sound(self, sound_name: str):
        """Reproduce un efecto de sonido."""
        if not self.audio_enabled or sound_name not in self.sound_effects:
            return

        try:
            self.sound_effects[sound_name].play()
        except Exception as e:
            print(f"⚠️  Error reproduciendo sonido {sound_name}: {e}")

    def _handle_gamepad_input(self):
        """Maneja entrada del gamepad."""
        if not self.joystick:
            return

        # Ejes analógicos
        left_x = self.joystick.get_axis(0)
        left_y = self.joystick.get_axis(1)

        # Botones
        button_a = self.joystick.get_button(0)  # A - Rotar
        button_b = self.joystick.get_button(1)  # B - Acelerar caída
        button_x = self.joystick.get_button(2)  # X - Mover izquierda
        button_y = self.joystick.get_button(3)  # Y - Mover derecha
        button_start = self.joystick.get_button(7)  # Start - Pausa
        button_select = self.joystick.get_button(6)  # Select - Movimiento cognitivo

        # Movimiento horizontal con stick izquierdo
        if abs(left_x) > 0.5:
            if left_x < -0.5 and self.board.is_valid_position(self.current_piece, -1, 0):
                self.current_piece.x -= 1
            elif left_x > 0.5 and self.board.is_valid_position(self.current_piece, 1, 0):
                self.current_piece.x += 1

        # Acelerar caída con stick izquierdo hacia abajo
        if left_y > 0.5:
            if self.board.is_valid_position(self.current_piece, 0, 1):
                self.current_piece.y += 1
            else:
                self._place_piece()

        # Botones
        if button_a:  # Rotar
            original_shape = [row[:] for row in self.current_piece.shape]
            self.current_piece.rotate()
            if not self.board.is_valid_position(self.current_piece):
                self.current_piece.shape = original_shape

        if button_b:  # Acelerar caída
            if self.board.is_valid_position(self.current_piece, 0, 1):
                self.current_piece.y += 1
            else:
                self._place_piece()

        if button_x:  # Mover izquierda
            if self.board.is_valid_position(self.current_piece, -1, 0):
                self.current_piece.x -= 1

        if button_y:  # Mover derecha
            if self.board.is_valid_position(self.current_piece, 1, 0):
                self.current_piece.x += 1

        if button_start:  # Pausa
            self.paused = not self.paused

        if button_select:  # Movimiento cognitivo
            asyncio.create_task(self._make_cognitive_move())
        self._play_background_music()

    def _spawn_new_piece(self):
        """Genera una nueva pieza actual."""
        if self.next_piece:
            self.current_piece = self.next_piece
        else:
            self.current_piece = self.board.spawn_piece()

        self.current_piece.x = self.board.width // 2 - len(self.current_piece.shape[0]) // 2
        self.current_piece.y = 0

    def _spawn_next_piece(self):
        """Genera la siguiente pieza."""
        self.next_piece = self.board.spawn_piece()

    async def run_game(self):
        """Loop principal del juego con aprendizaje continuo."""

        print("🚀 Iniciando Cognitive Tetris con U-CogNet")
        print("🎮 Controles: Gamepad + Teclado | Auto-restart: ON | Audio: ON")
        print("🧠 Aprendizaje continuo no supervisado activado")
        print()

        while self.game_running:
            current_time = time.time()

            # Manejar eventos
            await self._handle_events()

            # Manejar gamepad
            self._handle_gamepad_input()

            if not self.paused and not self.game_over:
                # Actualizar juego
                await self._update_game(current_time)

                # Movimiento cognitivo automático cada cierto tiempo
                if self.senses_enabled and random.random() < 0.02:  # 2% de probabilidad por frame
                    asyncio.create_task(self._make_cognitive_move())

            # Dibujar pantalla
            self._draw_screen()

            # Actualizar efectos visuales
            self._update_effects()

            # Control de FPS
            self.clock.tick(60)

            # Pequeña pausa para no saturar CPU
            await asyncio.sleep(0.01)

        self._cleanup()

    async def _handle_events(self):
        """Maneja eventos de entrada."""

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_running = False

            elif event.type == pygame.USEREVENT + 1:
                # Evento de reinicio automático
                self._restart_game()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.paused = not self.paused
                    self._play_sound('button_click')
                    print("⏸️  Juego pausado" if self.paused else "▶️  Juego reanudado")

                elif event.key == pygame.K_m:
                    self.show_metrics = not self.show_metrics
                    self._play_sound('button_click')
                    print("📊 Métricas" + (" mostradas" if self.show_metrics else " ocultas"))

                elif event.key == pygame.K_r and self.game_over:
                    self._restart_game()
                    self._play_sound('level_up')

                elif event.key == pygame.K_a:
                    self.audio_enabled = not self.audio_enabled
                    self._play_sound('button_click')
                    print("🔊 Audio " + ("activado" if self.audio_enabled else "desactivado"))

                elif event.key == pygame.K_s:
                    self.senses_enabled = not self.senses_enabled
                    self._play_sound('button_click')
                    print("👁️  Sentidos " + ("activados" if self.senses_enabled else "desactivados"))

                elif not self.paused and not self.game_over:
                    await self._handle_game_input(event.key)

    def _handle_button_click(self, pos):
        """Maneja clics en botones."""
        for button_name, button_data in self.buttons.items():
            if button_data['rect'].collidepoint(pos):
                self._play_sound('button_click')

                if button_name == 'auto_restart':
                    self.auto_restart = not self.auto_restart
                elif button_name == 'audio':
                    self.audio_enabled = not self.audio_enabled
                elif button_name == 'sentidos':
                    self.senses_enabled = not self.senses_enabled
                elif button_name == 'métricas':
                    self.show_metrics = not self.show_metrics

                # Actualizar textura del botón
                button_data['state'] = not button_data['state']
                button_data['texture'] = self._create_button_texture(
                    button_data['rect'].width,
                    button_data['rect'].height,
                    button_data['state']
                )

                print(f"🔄 {button_name}: {'ON' if button_data['state'] else 'OFF'}")
                break

    async def _handle_game_input(self, key):
        """Maneja entrada del juego."""

        if key == pygame.K_LEFT:
            if self.board.is_valid_position(self.current_piece, -1, 0):
                self.current_piece.x -= 1

        elif key == pygame.K_RIGHT:
            if self.board.is_valid_position(self.current_piece, 1, 0):
                self.current_piece.x += 1

        elif key == pygame.K_DOWN:
            if self.board.is_valid_position(self.current_piece, 0, 1):
                self.current_piece.y += 1
            else:
                self._place_piece()

        elif key == pygame.K_SPACE:
            # Rotación
            original_shape = [row[:] for row in self.current_piece.shape]
            self.current_piece.rotate()

            if not self.board.is_valid_position(self.current_piece):
                # Si no es válida, revertir
                self.current_piece.shape = original_shape

        elif key == pygame.K_RETURN:
            # Hacer movimiento cognitivo inteligente
            await self._make_cognitive_move()

    async def _make_cognitive_move(self):
        """Realiza un movimiento usando capacidades cognitivas."""

        if not self.current_piece:
            return

        print("🧠 Pensando movimiento cognitivo...")

        # Obtener decisión del jugador cognitivo
        decision = await self.player.make_move(self.current_piece, self.next_piece)

        # Aplicar la decisión
        action = decision['action']

        if action['type'] == 'no_valid_moves':
            print("❌ No hay movimientos válidos")
            return

        # Aplicar rotación
        for _ in range(action.get('rotation', 0)):
            self.current_piece.rotate()

        # Aplicar posición
        self.current_piece.x = action.get('position', self.board.width // 2)

        # Colocar pieza
        self._place_piece()

        # Mostrar métricas de la decisión
        metrics = decision['metrics']
        print(f"🧠 Tiempo de pensamiento: {metrics.get('thinking_time', 0):.3f}s")
        print(f"🎯 Calidad de decisión: {metrics.get('decision_quality', 0):.3f}")
        print(f"🎯 Confianza razonamiento: {metrics.get('reasoning_confidence', 0):.3f}")
        print(f"🧠 Creatividad aplicada: {'Sí' if metrics.get('creativity_applied', False) else 'No'}")

        # Registrar métricas
        self._record_metrics(decision)

    def _place_piece(self):
        """Coloca la pieza actual en el tablero."""

        self.board.place_piece(self.current_piece)
        lines_cleared = self.board.clear_lines()

        # Efectos de sonido
        if lines_cleared > 0:
            self._play_sound('line_clear')
            self.flash_effect = 0.5  # Flash de pantalla
            self._add_particles(self.current_piece, self.GREEN if lines_cleared >= 4 else self.BLUE)
            print(f"💎 ¡{lines_cleared} líneas limpiadas!")
        else:
            self._play_sound('piece_place')

        # Generar nueva pieza
        self._spawn_new_piece()
        self._spawn_next_piece()

        # Verificar game over
        if not self.board.is_valid_position(self.current_piece):
            self.game_over = True
            self._play_sound('game_over')
            self.screen_shake = 0.5
            print("💀 ¡Game Over!")

            # Reinicio automático si está activado
            if self.auto_restart:
                pygame.time.set_timer(pygame.USEREVENT + 1, 3000)  # Reiniciar en 3 segundos

    def _restart_game(self):
        """Reinicia el juego completamente."""
        print("🔄 Reiniciando juego...")

        # Guardar métricas de la sesión anterior
        self._save_session_metrics()

        # Resetear tablero y piezas
        self.board = TetrisBoard(self.board_width, self.board_height)
        self.player = CognitiveTetrisPlayer(self.board)

        # Resetear estado
        self.game_over = False
        self.paused = False
        self.total_games += 1

        # Resetear piezas
        self._spawn_new_piece()
        self._spawn_next_piece()

        # Resetear métricas de esta sesión
        self.metrics_history = []
        self.game_start_time = time.time()

        # Actualizar mejor score
        if self.board.score > self.best_score:
            self.best_score = self.board.score
            self._play_sound('level_up')
            print(f"🎉 ¡Nuevo récord: {self.best_score} puntos!")

        print("✅ Juego reiniciado - Continuando aprendizaje...")

    def _add_particles(self, piece, color):
        """Añade partículas para efectos visuales."""
        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    particle_x = self.board_offset_x + (piece.x + x) * self.cell_size + self.cell_size // 2
                    particle_y = self.board_offset_y + (piece.y + y) * self.cell_size + self.cell_size // 2

                    # Crear múltiples partículas por celda
                    for _ in range(3):
                        self.particles.append({
                            'x': particle_x,
                            'y': particle_y,
                            'vx': random.uniform(-2, 2),
                            'vy': random.uniform(-2, 2),
                            'life': 1.0,
                            'color': color
                        })

    def _update_effects(self):
        """Actualiza efectos visuales."""
        # Actualizar partículas
        for particle in self.particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['vy'] += 0.1  # Gravedad
            particle['life'] -= 0.02

            if particle['life'] <= 0:
                self.particles.remove(particle)

        # Actualizar screen shake
        if self.screen_shake > 0:
            self.screen_shake -= 0.02

        # Actualizar flash effect
        if self.flash_effect > 0:
            self.flash_effect -= 0.02

    async def _update_game(self, current_time):
        """Actualiza el estado del juego."""

        # Caída automática
        if current_time - self.last_update > self.fall_speed / 1000.0:
            if self.board.is_valid_position(self.current_piece, 0, 1):
                self.current_piece.y += 1
            else:
                self._place_piece()

            self.last_update = current_time

            # Aumentar velocidad con el nivel
            self.fall_speed = max(50, 500 - (self.board.level - 1) * 50)

    def _draw_screen(self):
        """Dibuja toda la pantalla."""

        # Efecto de flash
        flash_color = (255, 255, 255, int(self.flash_effect * 255)) if self.flash_effect > 0 else None

        # Screen shake
        shake_x = random.randint(-5, 5) * self.screen_shake if self.screen_shake > 0 else 0
        shake_y = random.randint(-5, 5) * self.screen_shake if self.screen_shake > 0 else 0

        # Crear surface temporal para efectos
        temp_surface = pygame.Surface((self.screen_width, self.screen_height))
        temp_surface.fill(self.BLACK)

        # Dibujar título
        title = self.font_title.render("U-CogNet Cognitive Tetris", True, self.CYAN)
        temp_surface.blit(title, (self.screen_width // 2 - title.get_width() // 2 + shake_x, 10 + shake_y))

        # Dibujar tablero
        self._draw_board(temp_surface, shake_x, shake_y)

        # Dibujar pieza actual
        if self.current_piece:
            self._draw_piece(temp_surface, self.current_piece, self.board_offset_x, self.board_offset_y, shake_x, shake_y)

        # Dibujar siguiente pieza
        self._draw_next_piece(temp_surface, shake_x, shake_y)

        # Dibujar métricas
        if self.show_metrics:
            self._draw_metrics(temp_surface, shake_x, shake_y)

        # Dibujar estado del juego
        self._draw_game_status(temp_surface, shake_x, shake_y)

        # Dibujar controles y botones
        self._draw_controls(temp_surface, shake_x, shake_y)
        self._draw_buttons(temp_surface, shake_x, shake_y)

        # Dibujar partículas
        self._draw_particles(temp_surface)

        # Aplicar flash effect
        if flash_color:
            flash_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            flash_surface.fill(flash_color)
            temp_surface.blit(flash_surface, (0, 0))

        # Dibujar en pantalla principal
        self.screen.blit(temp_surface, (0, 0))

        pygame.display.flip()

    def _draw_board(self, surface, shake_x=0, shake_y=0):
        """Dibuja el tablero de juego."""

        # Dibujar borde del tablero
        board_pixel_width = self.board_width * self.cell_size
        board_pixel_height = self.board_height * self.cell_size

        pygame.draw.rect(surface, self.WHITE,
                        (self.board_offset_x - 2 + shake_x, self.board_offset_y - 2 + shake_y,
                         board_pixel_width + 4, board_pixel_height + 4), 2)

        # Dibujar celdas del tablero
        for y in range(self.board.height):
            for x in range(self.board.width):
                if self.board.board[y][x]:
                    color = self._get_piece_color_from_board(x, y)
                    pygame.draw.rect(surface, color,
                                   (self.board_offset_x + x * self.cell_size + shake_x,
                                    self.board_offset_y + y * self.cell_size + shake_y,
                                    self.cell_size - 1, self.cell_size - 1))

        # Dibujar grid
        for x in range(self.board_width + 1):
            pygame.draw.line(surface, self.GRAY,
                           (self.board_offset_x + x * self.cell_size + shake_x, self.board_offset_y + shake_y),
                           (self.board_offset_x + x * self.cell_size + shake_x, self.board_offset_y + board_pixel_height + shake_y))

        for y in range(self.board_height + 1):
            pygame.draw.line(surface, self.GRAY,
                           (self.board_offset_x + shake_x, self.board_offset_y + y * self.cell_size + shake_y),
                           (self.board_offset_x + board_pixel_width + shake_x, self.board_offset_y + y * self.cell_size + shake_y))

    def _draw_piece(self, surface, piece: TetrisPiece, offset_x: int, offset_y: int, shake_x=0, shake_y=0):
        """Dibuja una pieza en pantalla."""

        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(surface, piece.color,
                                   (offset_x + (piece.x + x) * self.cell_size + shake_x,
                                    offset_y + (piece.y + y) * self.cell_size + shake_y,
                                    self.cell_size - 1, self.cell_size - 1))

    def _draw_next_piece(self, surface, shake_x=0, shake_y=0):
        """Dibuja la siguiente pieza."""

        if self.next_piece:
            next_x = self.board_offset_x + (self.board_width + 2) * self.cell_size
            next_y = self.board_offset_y + 50

            # Título
            next_title = self.font_medium.render("Siguiente:", True, self.WHITE)
            surface.blit(next_title, (next_x + shake_x, next_y - 30 + shake_y))

            # Pieza
            temp_piece = TetrisPiece(self.next_piece.shape_type)
            temp_piece.x = 0
            temp_piece.y = 0

            self._draw_piece(surface, temp_piece, next_x, next_y, shake_x, shake_y)

    def _draw_buttons(self, surface, shake_x=0, shake_y=0):
        """Dibuja los botones con texturas."""
        for button_name, button_data in self.buttons.items():
            rect = button_data['rect']
            texture = button_data['texture']
            text = button_data['text']
            hover = button_data['hover']

            # Posición con shake
            draw_x = rect.x + shake_x
            draw_y = rect.y + shake_y

            # Dibujar textura
            surface.blit(texture, (draw_x, draw_y))

            # Dibujar texto
            text_color = self.YELLOW if button_data['state'] else self.WHITE
            if hover:
                text_color = self.CYAN

            text_surface = self.font_small.render(text, True, text_color)
            text_x = draw_x + (rect.width - text_surface.get_width()) // 2
            text_y = draw_y + (rect.height - text_surface.get_height()) // 2
            surface.blit(text_surface, (text_x, text_y))

    def _draw_particles(self, surface):
        """Dibuja partículas para efectos visuales."""
        for particle in self.particles:
            alpha = int(particle['life'] * 255)
            color = particle['color'] + (alpha,)  # Añadir alpha

            # Crear surface con alpha
            particle_surface = pygame.Surface((4, 4), pygame.SRCALPHA)
            particle_surface.fill(color)
            pygame.draw.circle(particle_surface, color, (2, 2), 2)

            surface.blit(particle_surface, (int(particle['x'] - 2), int(particle['y'] - 2)))

    def _draw_metrics(self, surface, shake_x=0, shake_y=0):
        """Dibuja métricas en tiempo real."""

        metrics_x = self.board_offset_x + (self.board_width + 12) * self.cell_size
        metrics_y = self.board_offset_y

        # Título
        title = self.font_large.render("Métricas AGI en Tiempo Real", True, self.GREEN)
        surface.blit(title, (metrics_x + shake_x, metrics_y + shake_y))
        metrics_y += 50

        # Obtener métricas actuales
        real_time_metrics = self.player.get_real_time_metrics()
        game_metrics = self.board.get_game_metrics()

        # Métricas cognitivas
        cog_metrics = real_time_metrics['cognitive_metrics']
        self._draw_metric_section(surface, "🧠 Cognitivas", cog_metrics, metrics_x, metrics_y, shake_x, shake_y)
        metrics_y += 120

        # Métricas de rendimiento
        perf_metrics = real_time_metrics['performance_metrics']
        self._draw_metric_section(surface, "🎯 Rendimiento", perf_metrics, metrics_x, metrics_y, shake_x, shake_y)
        metrics_y += 120

        # Métricas de aprendizaje
        learn_metrics = real_time_metrics['learning_metrics']
        self._draw_metric_section(surface, "📚 Aprendizaje", learn_metrics, metrics_x, metrics_y, shake_x, shake_y)
        metrics_y += 120

        # Métricas de creatividad
        creat_metrics = real_time_metrics['creativity_metrics']
        self._draw_metric_section(surface, "🎨 Creatividad", creat_metrics, metrics_x, metrics_y, shake_x, shake_y)
        metrics_y += 120

        # Métricas de razonamiento
        reason_metrics = real_time_metrics['reasoning_metrics']
        self._draw_metric_section(surface, "🤔 Razonamiento", reason_metrics, metrics_x, metrics_y, shake_x, shake_y)

    def _draw_metric_section(self, surface, title: str, metrics: Dict[str, Any], x: int, y: int, shake_x=0, shake_y=0):
        """Dibuja una sección de métricas."""

        # Título de sección
        section_title = self.font_medium.render(title, True, self.YELLOW)
        surface.blit(section_title, (x + shake_x, y + shake_y))
        y += 25

        # Métricas individuales
        for key, value in metrics.items():
            if isinstance(value, float):
                value_str = f"{value:.3f}"
            elif isinstance(value, int):
                value_str = str(value)
            else:
                value_str = str(value)

            metric_text = self.font_small.render(f"{key}: {value_str}", True, self.WHITE)
            surface.blit(metric_text, (x + shake_x, y + shake_y))
            y += 20

    def _draw_game_status(self, surface, shake_x=0, shake_y=0):
        """Dibuja el estado actual del juego."""

        status_x = self.board_offset_x
        status_y = self.board_offset_y + self.board_height * self.cell_size + 20

        # Score y líneas
        score_text = self.font_medium.render(f"Score: {self.board.score}", True, self.WHITE)
        surface.blit(score_text, (status_x + shake_x, status_y + shake_y))

        lines_text = self.font_medium.render(f"Líneas: {self.board.lines_cleared}", True, self.WHITE)
        surface.blit(lines_text, (status_x + shake_x, status_y + 30 + shake_y))

        level_text = self.font_medium.render(f"Nivel: {self.board.level}", True, self.WHITE)
        surface.blit(level_text, (status_x + shake_x, status_y + 60 + shake_y))

        # Información de sesión
        session_text = self.font_small.render(f"Juegos: {self.total_games} | Mejor: {self.best_score}", True, self.GRAY)
        surface.blit(session_text, (status_x + shake_x, status_y + 90 + shake_y))

        # Estado del juego
        if self.paused:
            pause_text = self.font_large.render("PAUSADO", True, self.YELLOW)
            surface.blit(pause_text, (self.screen_width // 2 - pause_text.get_width() // 2 + shake_x,
                                    self.screen_height // 2 - pause_text.get_height() // 2 + shake_y))

        elif self.game_over:
            game_over_text = self.font_large.render("GAME OVER", True, self.RED)
            surface.blit(game_over_text, (self.screen_width // 2 - game_over_text.get_width() // 2 + shake_x,
                                        self.screen_height // 2 - game_over_text.get_height() // 2 + shake_y))

            if self.auto_restart:
                restart_text = self.font_medium.render("Reiniciando automáticamente...", True, self.WHITE)
                surface.blit(restart_text, (self.screen_width // 2 - restart_text.get_width() // 2 + shake_x,
                                          self.screen_height // 2 + 50 + shake_y))
            else:
                restart_text = self.font_medium.render("Presiona R para reiniciar", True, self.WHITE)
                surface.blit(restart_text, (self.screen_width // 2 - restart_text.get_width() // 2 + shake_x,
                                          self.screen_height // 2 + 50 + shake_y))

    def _draw_controls(self, surface, shake_x=0, shake_y=0):
        """Dibuja los controles del juego."""

        controls_x = self.board_offset_x
        controls_y = self.screen_height - 120

        controls_title = self.font_medium.render("Controles:", True, self.CYAN)
        surface.blit(controls_title, (controls_x + shake_x, controls_y + shake_y))
        controls_y += 25

        controls = [
            "← → ↓ : Mover pieza",
            "ESPACIO : Rotar",
            "ENTER : Movimiento cognitivo",
            "P : Pausa | M : Toggle métricas",
            "A : Audio | S : Sentidos",
            "R : Reiniciar (Game Over)",
            "Gamepad: Sticks + Botones A/B/X/Y"
        ]

        for control in controls:
            control_text = self.font_small.render(control, True, self.WHITE)
            surface.blit(control_text, (controls_x + shake_x, controls_y + shake_y))
            controls_y += 18

    def _get_piece_color_from_board(self, x: int, y: int) -> Tuple[int, int, int]:
        """Obtiene el color de una pieza en el tablero (simplificado)."""

        # En una implementación completa, mantendríamos track de los colores
        # Por ahora, usar colores fijos basados en posición
        colors = [self.CYAN, self.YELLOW, self.PURPLE, self.GREEN, self.RED, self.BLUE, self.ORANGE]
        return colors[(x + y) % len(colors)]

    def _record_metrics(self, decision: Dict[str, Any]):
        """Registra métricas para análisis posterior."""

        timestamp = time.time() - self.game_start_time

        metric_entry = {
            'timestamp': timestamp,
            'game_metrics': self.board.get_game_metrics(),
            'cognitive_metrics': self.player._get_cognitive_state(),
            'decision_metrics': decision.get('metrics', {}),
            'reasoning': decision.get('reasoning', {}),
            'action': decision.get('action', {})
        }

        self.metrics_history.append(metric_entry)

        # Limitar historia a últimas 1000 entradas
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

    def _save_final_metrics(self):
        """Guarda métricas finales al terminar el juego."""

        if not self.metrics_history:
            return

        # Crear directorio si no existe
        output_dir = Path("cognitive_tetris_results")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tetris_session_{timestamp}.json"

        final_metrics = {
            'session_info': {
                'start_time': self.game_start_time,
                'end_time': time.time(),
                'duration': time.time() - self.game_start_time,
                'final_score': self.board.score,
                'final_lines': self.board.lines_cleared,
                'final_level': self.board.level
            },
            'final_cognitive_state': self.player._get_cognitive_state(),
            'metrics_history': self.metrics_history,
            'learning_evolution': self.player.adaptive_learning
        }

        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, indent=2, ensure_ascii=False)

        print(f"💾 Métricas guardadas en: {output_dir / filename}")

        # Generar reporte resumen
        self._generate_summary_report(final_metrics, output_dir / f"tetris_report_{timestamp}.txt")

    def _generate_summary_report(self, metrics: Dict[str, Any], filepath: Path):
        """Genera un reporte resumen de la sesión."""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("🧠 U-CogNet Cognitive Tetris - Reporte de Sesión\n")
            f.write("=" * 50 + "\n\n")

            session = metrics['session_info']
            f.write(f"📊 Estadísticas del Juego:\n")
            f.write(f"  ⏱️  Duración: {session['duration']:.1f} segundos\n")
            f.write(f"  🎯 Score Final: {session['final_score']}\n")
            f.write(f"  💎 Líneas Totales: {session['final_lines']}\n")
            f.write(f"  📈 Nivel Máximo: {session['final_level']}\n\n")

            cognitive = metrics['final_cognitive_state']
            f.write(f"🧠 Estado Cognitivo Final:\n")
            f.write(f"  🧠 Tamaño Memoria: {cognitive['memory_size']}\n")
            f.write(f"  📚 Patrones Aprendidos: {cognitive['patterns_learned']}\n")
            f.write(f"  🎨 Creatividad Promedio: {cognitive['creativity_avg']:.3f}\n")
            f.write(f"  ⚡ Tiempo de Pensamiento: {cognitive['thinking_time_avg']:.3f}s\n")
            f.write(f"  🎯 Calidad de Decisiones: {cognitive['decision_quality_avg']:.3f}\n")
            f.write(f"  🧠 Carga Cognitiva: {cognitive['cognitive_load']:.3f}\n\n")

            # Análisis de evolución
            if metrics['metrics_history']:
                f.write(f"📈 Evolución del Rendimiento:\n")
                early_metrics = metrics['metrics_history'][:len(metrics['metrics_history'])//4]
                late_metrics = metrics['metrics_history'][-len(metrics['metrics_history'])//4:]

                if early_metrics and late_metrics:
                    early_score = np.mean([m['game_metrics']['score'] for m in early_metrics])
                    late_score = np.mean([m['game_metrics']['score'] for m in late_metrics])

                    early_creativity = np.mean([m['cognitive_metrics']['creativity_avg'] for m in early_metrics if m['cognitive_metrics']['creativity_avg'] > 0])
                    late_creativity = np.mean([m['cognitive_metrics']['creativity_avg'] for m in late_metrics if m['cognitive_metrics']['creativity_avg'] > 0])

                    f.write(f"  📊 Mejora Score: {early_score:.1f} → {late_score:.1f} ({((late_score/early_score-1)*100):+.1f}%)\n")
                    if early_creativity and late_creativity:
                        f.write(f"  🎨 Mejora Creatividad: {early_creativity:.3f} → {late_creativity:.3f} ({((late_creativity/early_creativity-1)*100):+.1f}%)\n")

            f.write(f"\n🤖 Evaluación AGI:\n")
            adaptability = cognitive['cognitive_load'] * 0.3 + cognitive['creativity_avg'] * 0.4 + (cognitive['patterns_learned'] / 50.0) * 0.3
            f.write(f"  🔄 Adaptabilidad: {min(1.0, adaptability):.3f}/1.0\n")

            reasoning_quality = cognitive['decision_quality_avg'] * 0.5 + cognitive['thinking_time_avg'] * -0.2 + cognitive['creativity_avg'] * 0.3
            f.write(f"  🤔 Calidad Razonamiento: {max(0.0, min(1.0, reasoning_quality)):.3f}/1.0\n")

            learning_efficiency = len(metrics['learning_evolution']['success_rates']) / 50.0
            f.write(f"  📚 Eficiencia Aprendizaje: {min(1.0, learning_efficiency):.3f}/1.0\n")

            overall_agi = (adaptability + reasoning_quality + learning_efficiency) / 3.0
            f.write(f"  🚀 Score AGI General: {overall_agi:.3f}/1.0\n")

        print(f"📄 Reporte generado: {filepath}")

    def _save_session_metrics(self):
        """Guarda métricas de la sesión actual antes de reiniciar."""

        if not self.metrics_history:
            return

        # Crear directorio si no existe
        output_dir = Path("cognitive_tetris_sessions")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{self.total_games}_{timestamp}.json"

        session_metrics = {
            'session_number': self.total_games,
            'game_duration': time.time() - self.game_start_time,
            'final_score': self.board.score,
            'final_lines': self.board.lines_cleared,
            'final_level': self.board.level,
            'best_score': self.best_score,
            'metrics_history': self.metrics_history[-100:],  # Últimas 100 entradas
            'cognitive_state': self.player._get_cognitive_state()
        }

        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(session_metrics, f, indent=2, ensure_ascii=False)

        print(f"💾 Sesión {self.total_games} guardada: {output_dir / filename}")

    def _draw_screen(self):
        """Dibuja toda la pantalla del juego."""

        # Calcular shake
        shake_x = int(random.uniform(-self.screen_shake * 10, self.screen_shake * 10)) if self.screen_shake > 0 else 0
        shake_y = int(random.uniform(-self.screen_shake * 10, self.screen_shake * 10)) if self.screen_shake > 0 else 0

        # Limpiar pantalla
        self.screen.fill(self.BLACK)

        # Aplicar flash effect
        if self.flash_effect > 0:
            flash_surface = pygame.Surface((self.screen_width, self.screen_height))
            flash_surface.fill(self.WHITE)
            flash_surface.set_alpha(int(self.flash_effect * 128))
            self.screen.blit(flash_surface, (0, 0))

        # Dibujar tablero
        self._draw_board(self.screen, shake_x, shake_y)

        # Dibujar piezas
        self._draw_piece(self.screen, self.current_piece, self.board_offset_x, self.board_offset_y, shake_x, shake_y)
        self._draw_next_piece(self.screen, shake_x, shake_y)

        # Dibujar estado del juego
        self._draw_game_status(self.screen, shake_x, shake_y)

        # Dibujar controles
        self._draw_controls(self.screen, shake_x, shake_y)

        # Dibujar métricas si están activadas
        if self.show_metrics:
            self._draw_metrics(self.screen, shake_x, shake_y)

        # Dibujar botones
        self._draw_buttons(self.screen, shake_x, shake_y)

        # Dibujar partículas
        self._draw_particles(self.screen)

        # Actualizar pantalla
        pygame.display.flip()

    def _cleanup(self):
        """Limpieza al salir."""
        pygame.quit()
        print("👋 ¡Gracias por jugar Cognitive Tetris!")

async def main():
    """Función principal."""
    game = CognitiveTetrisGame()
    await game.run_game()

if __name__ == "__main__":
    asyncio.run(main())