#!/usr/bin/env python3
"""
Controlador Adaptativo de Gates - Policy Gradient para Gating Multimodal

Implementa un controlador que aprende políticas de gating usando gradiente de política,
optimizando la activación/desactivación de modalidades basado en recompensas intrínsecas.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import random

class PolicyNetwork(nn.Module):
    """
    Red neuronal para política de gating multimodal.

    Entrada: Estado del sistema (pesos de atención, recompensas recientes, etc.)
    Salida: Probabilidades de acción para cada modalidad (OPEN/FILTERING/CLOSED)
    """

    def __init__(self, input_size: int = 20, hidden_size: int = 64, num_actions: int = 3):
        super(PolicyNetwork, self).__init__()
        self.num_actions = num_actions  # OPEN, FILTERING, CLOSED

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, x):
        logits = self.network(x)
        return torch.softmax(logits, dim=-1)

class AdaptiveGatingController:
    """
    Controlador Adaptativo que aprende políticas de gating usando Policy Gradient.

    Características:
    - Una política por modalidad
    - Memoria episódica para contexto temporal
    - Optimización basada en recompensas intrínsecas
    - Exploración vs explotación balanceada
    """

    def __init__(self,
                 modalities: List[str] = ['visual', 'audio', 'text', 'tactile'],
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay: float = 0.995):
        self.modalities = modalities
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Una red de política por modalidad
        self.policies = {}
        self.optimizers = {}
        self.input_sizes = {
            'visual': 15,   # Más inputs para visual (más complejo)
            'audio': 12,
            'text': 10,
            'tactile': 8
        }

        for modality in modalities:
            input_size = self.input_sizes[modality]
            self.policies[modality] = PolicyNetwork(input_size=input_size)
            self.optimizers[modality] = optim.Adam(self.policies[modality].parameters(), lr=learning_rate)

        # Memoria para experiencias
        self.memory = deque(maxlen=1000)

        # Estado actual de gates
        self.current_gates = {mod: 'open' for mod in modalities}
        self.attention_weights = {mod: 0.5 for mod in modalities}

        # Historial para contexto
        self.reward_history = {mod: deque(maxlen=50) for mod in modalities}
        self.state_history = deque(maxlen=20)

    def get_state_representation(self, modality: str, intrinsic_rewards: Dict[str, float]) -> torch.Tensor:
        """
        Crea representación del estado para una modalidad específica.

        Args:
            modality: Nombre de la modalidad
            intrinsic_rewards: Recompensas intrínsecas actuales

        Returns:
            Tensor con representación del estado
        """
        state_features = []

        # Recompensas intrínsecas actuales
        state_features.extend([
            intrinsic_rewards.get('per', 0.0),
            intrinsic_rewards.get('igr', 0.0),
            intrinsic_rewards.get('um', 0.0),
            intrinsic_rewards.get('tc', 0.0),
            intrinsic_rewards.get('total', 0.0)
        ])

        # Historial de recompensas (últimas 5)
        recent_rewards = list(self.reward_history[modality])[-5:]
        while len(recent_rewards) < 5:
            recent_rewards.insert(0, 0.0)
        state_features.extend(recent_rewards)

        # Estado actual de gates (one-hot)
        gate_states = {'open': [1,0,0], 'filtering': [0,1,0], 'closed': [0,0,1]}
        state_features.extend(gate_states.get(self.current_gates[modality], [1,0,0]))

        # Peso de atención actual
        state_features.append(self.attention_weights[modality])

        # Información contextual global (si disponible)
        if self.state_history:
            # Promedio de recompensas de otras modalidades
            other_modalities = [m for m in self.modalities if m != modality]
            other_rewards = []
            for other_mod in other_modalities:
                if self.reward_history[other_mod]:
                    other_rewards.append(np.mean(list(self.reward_history[other_mod])[-3:]))
                else:
                    other_rewards.append(0.0)
            state_features.extend(other_rewards)

        # Padding si es necesario
        target_size = self.input_sizes[modality]
        while len(state_features) < target_size:
            state_features.append(0.0)

        return torch.tensor(state_features[:target_size], dtype=torch.float32)

    def select_action(self, modality: str, intrinsic_rewards: Dict[str, float]) -> str:
        """
        Selecciona una acción de gating usando la política aprendida.

        Args:
            modality: Nombre de la modalidad
            intrinsic_rewards: Recompensas intrínsecas actuales

        Returns:
            Acción seleccionada: 'open', 'filtering', o 'closed'
        """
        state = self.get_state_representation(modality, intrinsic_rewards)

        # ε-greedy para exploración
        if random.random() < self.epsilon:
            action_idx = random.randint(0, 2)  # Exploración aleatoria
        else:
            with torch.no_grad():
                action_probs = self.policies[modality](state.unsqueeze(0))
                action_idx = torch.argmax(action_probs).item()

        actions = ['open', 'filtering', 'closed']
        selected_action = actions[action_idx]

        # Almacenar experiencia para aprendizaje posterior
        self.memory.append({
            'modality': modality,
            'state': state,
            'action': action_idx,
            'reward': intrinsic_rewards.get('total', 0.0),
            'next_state': None  # Se actualizará después
        })

        return selected_action

    def update_gates(self, new_gates: Dict[str, str]):
        """
        Actualiza el estado de los gates.

        Args:
            new_gates: Nuevo estado de gates por modalidad
        """
        self.current_gates.update(new_gates)

        # Actualizar pesos de atención basados en gates
        gate_to_weight = {'open': 1.0, 'filtering': 0.5, 'closed': 0.0}
        for modality, gate in new_gates.items():
            self.attention_weights[modality] = gate_to_weight[gate]

    def update_reward_history(self, modality: str, reward: float):
        """
        Actualiza el historial de recompensas para una modalidad.
        """
        self.reward_history[modality].append(reward)

    def learn_from_experience(self, batch_size: int = 32):
        """
        Aprende de las experiencias almacenadas usando Policy Gradient.

        Args:
            batch_size: Tamaño del batch para aprendizaje
        """
        if len(self.memory) < batch_size:
            return

        # Procesar por modalidad
        for modality in self.modalities:
            modality_experiences = [exp for exp in self.memory if exp['modality'] == modality]

            if len(modality_experiences) < batch_size:
                continue

            # Sample batch
            batch = random.sample(modality_experiences, min(batch_size, len(modality_experiences)))

            # Calcular retornos descontados
            returns = []
            G = 0
            for exp in reversed(batch):
                G = exp['reward'] + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)

            # Normalizar returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Calcular pérdida de policy gradient
            policy_loss = 0
            for i, exp in enumerate(batch):
                state = exp['state'].unsqueeze(0)
                action = exp['action']

                action_probs = self.policies[modality](state)
                log_prob = torch.log(action_probs[0, action])

                policy_loss += -log_prob * returns[i]

            policy_loss /= len(batch)

            # Optimización
            self.optimizers[modality].zero_grad()
            policy_loss.backward()
            self.optimizers[modality].step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Limpiar memoria antigua
        while len(self.memory) > 500:
            self.memory.popleft()

    def get_policy_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Retorna estadísticas de las políticas aprendidas.
        """
        stats = {}
        for modality in self.modalities:
            policy = self.policies[modality]

            # Probabilidades promedio para cada acción
            test_states = torch.randn(10, self.input_sizes[modality])  # Estados de prueba aleatorios
            with torch.no_grad():
                action_probs = policy(test_states)
                avg_probs = action_probs.mean(dim=0).tolist()

            stats[modality] = {
                'avg_open_prob': avg_probs[0],
                'avg_filtering_prob': avg_probs[1],
                'avg_closed_prob': avg_probs[2],
                'current_gate': self.current_gates[modality],
                'attention_weight': self.attention_weights[modality],
                'exploration_rate': self.epsilon
            }

        return stats

    def save_policies(self, filepath: str):
        """
        Guarda las políticas aprendidas.
        """
        torch.save({
            'policies': {mod: policy.state_dict() for mod, policy in self.policies.items()},
            'current_gates': self.current_gates,
            'attention_weights': self.attention_weights,
            'epsilon': self.epsilon
        }, filepath)

    def load_policies(self, filepath: str):
        """
        Carga políticas previamente aprendidas.
        """
        checkpoint = torch.load(filepath)
        for modality, state_dict in checkpoint['policies'].items():
            if modality in self.policies:
                self.policies[modality].load_state_dict(state_dict)

        self.current_gates = checkpoint.get('current_gates', self.current_gates)
        self.attention_weights = checkpoint.get('attention_weights', self.attention_weights)
        self.epsilon = checkpoint.get('epsilon', self.epsilon)

    def reset(self):
        """Reinicia el controlador."""
        self.current_gates = {mod: 'open' for mod in self.modalities}
        self.attention_weights = {mod: 0.5 for mod in self.modalities}
        self.memory.clear()
        for mod in self.modalities:
            self.reward_history[mod].clear()
        self.epsilon = 1.0</content>
<parameter name="filePath">/mnt/c/Users/desar/Documents/Science/UCogNet/adaptive_gating_controller.py