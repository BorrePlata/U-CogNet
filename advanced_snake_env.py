#!/usr/bin/env python3
"""
Entorno Snake Avanzado para Examen de Gating Multimodal
Incluye múltiples modalidades: visual, audio, texto, táctil
Entorno 3D conceptual con obstáculos dinámicos y múltiples objetivos
"""

import pygame
import numpy as np
import random
from enum import Enum
from typing import List, Tuple, Dict, Optional

class Modality(Enum):
    VISUAL = "visual"
    AUDIO = "audio"
    TEXT = "text"
    TACTILE = "tactile"

class ObstacleType(Enum):
    STATIC = "static"
    MOVING = "moving"
    HAZARD = "hazard"

class AdvancedSnakeEnv:
    def __init__(self, width=30, height=30, render=True):
        self.width = width
        self.height = height
        self.render = render

        # Estado del juego
        self.snake = [(width//2, height//2)]
        self.direction = (0, -1)  # Arriba
        self.food = []
        self.obstacles = []
        self.score = 0
        self.steps = 0
        self.max_steps = 2000

        # Elementos avanzados
        self.power_ups = []  # Comida especial
        self.hazards = []    # Peligros
        self.dynamic_obstacles = []

        # Modalidades
        self.text_commands = []
        self.audio_events = []
        self.tactile_sensors = {}

        # Inicialización
        self._place_food()
        self._place_obstacles()
        self._init_pygame()

    def _init_pygame(self):
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width * 20, self.height * 20))
            pygame.display.set_caption("Advanced Snake - Multimodal Gating Test")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

    def _place_food(self):
        """Coloca múltiples tipos de comida"""
        for _ in range(3):  # 3 comidas normales
            while True:
                x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
                if (x, y) not in self.snake and not self._is_obstacle(x, y):
                    self.food.append((x, y, 'normal'))
                    break

        # Comida especial (power-up)
        while True:
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            if (x, y) not in self.snake and not self._is_obstacle(x, y):
                self.power_ups.append((x, y, 'speed_boost'))
                break

    def _place_obstacles(self):
        """Coloca obstáculos estáticos y dinámicos"""
        # Obstáculos estáticos
        for _ in range(10):
            while True:
                x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
                if (x, y) not in self.snake and (x, y) not in [f[:2] for f in self.food]:
                    self.obstacles.append((x, y, ObstacleType.STATIC))
                    break

        # Obstáculos dinámicos
        for _ in range(3):
            while True:
                x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
                if (x, y) not in self.snake and not self._is_obstacle(x, y):
                    self.dynamic_obstacles.append({
                        'pos': [x, y],
                        'direction': random.choice([(0,1), (1,0), (0,-1), (-1,0)]),
                        'speed': 1
                    })
                    break

    def _is_obstacle(self, x, y):
        """Verifica si una posición tiene obstáculo"""
        return any(obs[0] == x and obs[1] == y for obs in self.obstacles)

    def _move_dynamic_obstacles(self):
        """Mueve obstáculos dinámicos"""
        for obs in self.dynamic_obstacles:
            x, y = obs['pos']
            dx, dy = obs['direction']

            new_x, new_y = x + dx, y + dy

            # Rebote en bordes
            if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height:
                obs['direction'] = (-dx, -dy)
            else:
                obs['pos'] = [new_x, new_y]

    def _generate_multimodal_signals(self):
        """Genera señales para todas las modalidades"""
        head_x, head_y = self.snake[0]

        # Señales táctiles (sensores de proximidad)
        self.tactile_sensors = {
            'front_distance': self._get_distance_in_direction(self.direction),
            'left_distance': self._get_distance_in_direction(self._rotate_direction(self.direction, -90)),
            'right_distance': self._get_distance_in_direction(self._rotate_direction(self.direction, 90)),
            'back_distance': self._get_distance_in_direction(self._rotate_direction(self.direction, 180))
        }

        # Eventos de audio
        self.audio_events = []
        if self._is_near_food():
            self.audio_events.append('food_near')
        if self._is_near_obstacle():
            self.audio_events.append('danger_near')
        if self._is_near_powerup():
            self.audio_events.append('powerup_near')

        # Comandos de texto (simulados)
        self.text_commands = []
        if random.random() < 0.1:  # 10% chance
            self.text_commands.append(random.choice([
                'move_forward', 'turn_left', 'turn_right', 'avoid_obstacle', 'seek_food'
            ]))

    def _get_distance_in_direction(self, direction):
        """Calcula distancia al obstáculo en una dirección"""
        x, y = self.snake[0]
        distance = 0
        while True:
            x += direction[0]
            y += direction[1]
            distance += 1
            if x < 0 or x >= self.width or y < 0 or y >= self.height or self._is_obstacle(x, y):
                return distance
            if distance > 10:  # Máximo rango
                return 10

    def _rotate_direction(self, direction, degrees):
        """Rota una dirección 90 grados"""
        dx, dy = direction
        if degrees == 90:
            return (-dy, dx)
        elif degrees == -90:
            return (dy, -dx)
        elif degrees == 180:
            return (-dx, -dy)

    def _is_near_food(self):
        head_x, head_y = self.snake[0]
        return any(abs(head_x - fx) + abs(head_y - fy) <= 3 for fx, fy, _ in self.food)

    def _is_near_obstacle(self):
        head_x, head_y = self.snake[0]
        return any(abs(head_x - ox) + abs(head_y - oy) <= 2 for ox, oy, _ in self.obstacles)

    def _is_near_powerup(self):
        head_x, head_y = self.snake[0]
        return any(abs(head_x - px) + abs(head_y - py) <= 3 for px, py, _ in self.power_ups)

    def reset(self):
        """Reinicia el entorno"""
        self.snake = [(self.width//2, self.height//2)]
        self.direction = (0, -1)
        self.food = []
        self.power_ups = []
        self.obstacles = []
        self.dynamic_obstacles = []
        self.score = 0
        self.steps = 0

        self._place_food()
        self._place_obstacles()
        return self._get_state()

    def _get_state(self):
        """Obtiene el estado actual del entorno"""
        head_x, head_y = self.snake[0]

        # Estado básico
        state = [
            head_x / self.width,  # Normalizado
            head_y / self.height,
            self.direction[0],
            self.direction[1],
            len(self.snake),  # Longitud de la serpiente
        ]

        # Distancia a la comida más cercana
        if self.food:
            closest_food = min(self.food, key=lambda f: abs(head_x - f[0]) + abs(head_y - f[1]))
            state.extend([
                (closest_food[0] - head_x) / self.width,
                (closest_food[1] - head_y) / self.height
            ])
        else:
            state.extend([0, 0])

        # Sensores táctiles
        state.extend([
            self.tactile_sensors.get('front_distance', 5) / 10,
            self.tactile_sensors.get('left_distance', 5) / 10,
            self.tactile_sensors.get('right_distance', 5) / 10,
            self.tactile_sensors.get('back_distance', 5) / 10
        ])

        return np.array(state, dtype=np.float32)

    def step(self, action):
        """Ejecuta una acción"""
        # Mapear acción a dirección
        if action == 0:  # Izquierda
            self.direction = self._rotate_direction(self.direction, -90)
        elif action == 1:  # Derecha
            self.direction = self._rotate_direction(self.direction, 90)
        elif action == 2:  # Adelante (mantener dirección)
            pass

        # Mover serpiente
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # Verificar colisiones
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height or
            new_head in self.snake or
            self._is_obstacle(new_head[0], new_head[1])):
            return self._get_state(), -100, True, {'collision': True}

        # Mover obstáculos dinámicos
        self._move_dynamic_obstacles()

        # Actualizar serpiente
        self.snake.insert(0, new_head)

        # Verificar comida
        reward = 0
        food_eaten = None
        for i, (fx, fy, ftype) in enumerate(self.food):
            if new_head == (fx, fy):
                reward += 10
                self.score += 10
                food_eaten = i
                break

        if food_eaten is not None:
            self.food.pop(food_eaten)
            # Nueva comida
            while True:
                x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
                if (x, y) not in self.snake and not self._is_obstacle(x, y):
                    self.food.append((x, y, 'normal'))
                    break
        else:
            # Si no comió, quitar cola
            self.snake.pop()

        # Verificar power-ups
        for i, (px, py, ptype) in enumerate(self.power_ups):
            if new_head == (px, py):
                if ptype == 'speed_boost':
                    reward += 20
                    self.score += 20
                self.power_ups.pop(i)
                break

        # Penalización por tiempo
        reward -= 0.1

        self.steps += 1

        # Generar señales multimodales
        self._generate_multimodal_signals()

        done = self.steps >= self.max_steps or len(self.snake) >= self.width * self.height

        return self._get_state(), reward, done, {}

    def render_frame(self):
        """Renderiza el frame actual"""
        if not self.render:
            return

        self.screen.fill((0, 0, 0))

        # Dibujar grid
        for x in range(self.width):
            for y in range(self.height):
                pygame.draw.rect(self.screen, (50, 50, 50),
                               (x*20, y*20, 20, 20), 1)

        # Dibujar serpiente
        for i, (x, y) in enumerate(self.snake):
            color = (0, 255, 0) if i == 0 else (0, 200, 0)
            pygame.draw.rect(self.screen, color, (x*20, y*20, 18, 18))

        # Dibujar comida
        for x, y, _ in self.food:
            pygame.draw.circle(self.screen, (255, 0, 0), (x*20+10, y*20+10), 8)

        # Dibujar power-ups
        for x, y, _ in self.power_ups:
            pygame.draw.circle(self.screen, (255, 255, 0), (x*20+10, y*20+10), 8)

        # Dibujar obstáculos
        for x, y, otype in self.obstacles:
            color = (128, 128, 128) if otype == ObstacleType.STATIC else (255, 165, 0)
            pygame.draw.rect(self.screen, color, (x*20, y*20, 18, 18))

        # Dibujar obstáculos dinámicos
        for obs in self.dynamic_obstacles:
            x, y = obs['pos']
            pygame.draw.rect(self.screen, (255, 0, 255), (x*20, y*20, 18, 18))

        # Información
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(10)

    def get_multimodal_data(self):
        """Obtiene datos multimodales para el sistema de atención"""
        return {
            'visual': {
                'snake_length': len(self.snake),
                'food_distance': min([abs(self.snake[0][0] - fx) + abs(self.snake[0][1] - fy)
                                    for fx, fy, _ in self.food]) if self.food else 10,
                'obstacle_proximity': min([abs(self.snake[0][0] - ox) + abs(self.snake[0][1] - oy)
                                         for ox, oy, _ in self.obstacles]) if self.obstacles else 10,
                'powerup_near': self._is_near_powerup()
            },
            'audio': {
                'events': self.audio_events.copy(),
                'intensity': len(self.audio_events) * 0.3
            },
            'text': {
                'commands': self.text_commands.copy(),
                'confidence': 0.8 if self.text_commands else 0.0
            },
            'tactile': self.tactile_sensors.copy()
        }