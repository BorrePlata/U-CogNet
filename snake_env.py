#!/usr/bin/env python3
"""
Entorno de Snake para U-CogNet
Entorno de prueba para aprendizaje incremental y memoria.
"""

import numpy as np
import random
from typing import Tuple, List, Dict
from enum import Enum

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class SnakeEnv:
    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.reset()

    def reset(self) -> Dict:
        """Reinicia el entorno"""
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = Direction.RIGHT
        self.food = self._place_food()
        self.score = 0
        self.done = False
        return self._get_state()

    def _place_food(self) -> Tuple[int, int]:
        """Coloca comida en posición aleatoria"""
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in self.snake:
                return (x, y)

    def _get_state(self) -> Dict:
        """Obtiene el estado actual del entorno"""
        # Representación simple: grid con snake, food, walls
        grid = np.zeros((self.height, self.width))
        
        # Paredes
        grid[0, :] = -1
        grid[-1, :] = -1
        grid[:, 0] = -1
        grid[:, -1] = -1
        
        # Snake
        for x, y in self.snake:
            grid[y, x] = 1
        
        # Cabeza de snake
        head_x, head_y = self.snake[0]
        grid[head_y, head_x] = 2
        
        # Food
        food_x, food_y = self.food
        grid[food_y, food_x] = 3
        
        return {
            'grid': grid,
            'snake': self.snake.copy(),
            'food': self.food,
            'direction': self.direction,
            'score': self.score,
            'done': self.done
        }

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Ejecuta una acción
        Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        """
        # Map action to direction
        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        new_direction = directions[action]
        
        # No reverse direction
        if (new_direction.value[0] * -1, new_direction.value[1] * -1) == self.direction.value:
            new_direction = self.direction
        
        self.direction = new_direction
        
        # Move snake
        head_x, head_y = self.snake[0]
        dx, dy = self.direction.value
        new_head = (head_x + dx, head_y + dy)
        
        # Check collision with walls
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height):
            self.done = True
            return self._get_state(), -10, True, {}
        
        # Check collision with self
        if new_head in self.snake:
            self.done = True
            return self._get_state(), -10, True, {}
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Check food
        reward = 0
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            self.snake.pop()  # Remove tail
        
        return self._get_state(), reward, self.done, {}

    def render(self):
        """Render simple en consola"""
        state = self._get_state()
        grid = state['grid']
        for row in grid:
            print(''.join(['#' if x == -1 else 'O' if x == 1 else '@' if x == 2 else '*' if x == 3 else ' ' for x in row]))
        print(f"Score: {self.score}")