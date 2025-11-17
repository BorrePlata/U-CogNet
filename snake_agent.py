#!/usr/bin/env python3
"""
Agente de Snake con Aprendizaje Incremental
Usa Q-learning con memoria episódica para aprendizaje continuo.
"""

import numpy as np
import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple
import json
import os

class IncrementalSnakeAgent:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 1.0, epsilon_decay: float = 0.995):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01
        
        # Q-table: state -> action values
        self.q_table = defaultdict(lambda: np.zeros(4))
        
        # Memoria episódica: recordar experiencias pasadas
        self.episodic_memory = deque(maxlen=1000)
        
        # Estadísticas de aprendizaje
        self.learning_stats = {
            'episodes': 0,
            'total_reward': 0,
            'best_score': 0,
            'learning_curve': [],
            'memory_size': 0
        }
        
        # Cargar conocimiento previo si existe
        self.load_knowledge()

    def _get_state_key(self, state: Dict) -> str:
        """Convierte el estado en una clave hashable"""
        grid = state['grid']
        # Simplificar: solo considerar posiciones relativas de cabeza, comida y dirección
        head_x, head_y = state['snake'][0]
        food_x, food_y = state['food']
        
        # Distancia relativa
        dx = food_x - head_x
        dy = food_y - head_y
        
        # Dirección actual
        dir_idx = ['UP', 'DOWN', 'LEFT', 'RIGHT'].index(state['direction'].name)
        
        return f"{dx},{dy},{dir_idx}"

    def choose_action(self, state: Dict) -> int:
        """Elige acción usando epsilon-greedy"""
        state_key = self._get_state_key(state)
        
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # Exploración
        else:
            return np.argmax(self.q_table[state_key])  # Explotación

    def learn(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool):
        """Actualiza Q-table usando Q-learning"""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key]) if not done else 0
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        self.q_table[state_key][action] = new_q
        
        # Almacenar en memoria episódica (solo info necesaria)
        self.episodic_memory.append({
            'state': state_key,
            'action': action,
            'reward': reward,
            'next_state': next_state_key,
            'done': done
        })
        
        # Replay learning: aprender de experiencias pasadas
        if len(self.episodic_memory) > 10:
            self._replay_learning()

    def _replay_learning(self):
        """Aprende de experiencias almacenadas (memoria)"""
        # Tomar una muestra aleatoria de experiencias
        sample_size = min(10, len(self.episodic_memory))
        experiences = random.sample(list(self.episodic_memory), sample_size)
        
        for exp in experiences:
            state_key = exp['state']
            action = exp['action']
            reward = exp['reward']
            next_state_key = exp['next_state']
            done = exp['done']
            
            current_q = self.q_table[state_key][action]
            max_next_q = np.max(self.q_table[next_state_key]) if not done else 0
            new_q = current_q + self.alpha * 0.1 * (reward + self.gamma * max_next_q - current_q)  # Menor alpha para replay
            
            self.q_table[state_key][action] = new_q

    def update_stats(self, episode_reward: float, score: int):
        """Actualiza estadísticas de aprendizaje"""
        self.learning_stats['episodes'] += 1
        self.learning_stats['total_reward'] += episode_reward
        self.learning_stats['best_score'] = max(self.learning_stats['best_score'], score)
        self.learning_stats['learning_curve'].append(episode_reward)
        self.learning_stats['memory_size'] = len(self.episodic_memory)
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_learning_stats(self) -> Dict:
        """Obtiene estadísticas actuales"""
        return self.learning_stats.copy()

    def save_knowledge(self):
        """Guarda el conocimiento aprendido"""
        # Convertir episodic_memory a serializable
        serializable_memory = []
        for exp in self.episodic_memory:
            serializable_exp = exp.copy()
            # Convertir cualquier numpy array a lista
            for key, value in serializable_exp.items():
                if isinstance(value, np.ndarray):
                    serializable_exp[key] = value.tolist()
            serializable_memory.append(serializable_exp)
        
        knowledge = {
            'q_table': {k: v.tolist() for k, v in self.q_table.items()},
            'learning_stats': self.learning_stats,
            'episodic_memory': []  # No guardar memoria completa por simplicidad
        }
        
        with open('snake_knowledge.json', 'w') as f:
            json.dump(knowledge, f, indent=2)

    def load_knowledge(self):
        """Carga conocimiento previo"""
        if os.path.exists('snake_knowledge.json'):
            try:
                with open('snake_knowledge.json', 'r') as f:
                    knowledge = json.load(f)
                
                self.q_table.update({k: np.array(v) for k, v in knowledge.get('q_table', {}).items()})
                self.learning_stats.update(knowledge.get('learning_stats', {}))
                # No cargar episodic_memory por simplicidad
                
                print(f"✅ Conocimiento cargado: {len(self.q_table)} estados aprendidos")
            except Exception as e:
                print(f"⚠️ Error cargando conocimiento: {e}")