#!/usr/bin/env python3
"""
Controlador Adaptativo de Gates - Versión simplificada
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List
from collections import deque

class SimplePolicyNetwork(nn.Module):
    def __init__(self, input_size: int = 4):
        super(SimplePolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # 3 acciones: open, filtering, closed
        )

    def forward(self, x):
        return torch.softmax(self.network(x), dim=-1)

class AdaptiveGatingController:
    """Controlador simplificado"""

    def __init__(self, modalities: List[str] = ['visual', 'audio', 'text', 'tactile']):
        self.modalities = modalities
        self.policies = {mod: SimplePolicyNetwork() for mod in modalities}
        self.optimizers = {mod: optim.Adam(self.policies[mod].parameters(), lr=0.01) for mod in modalities}
        self.current_gates = {mod: 'open' for mod in modalities}
        self.attention_weights = {mod: 1.0 if self.current_gates[mod] == 'open' else 0.5 if self.current_gates[mod] == 'filtering' else 0.0 for mod in modalities}
        self.reward_history = {mod: deque(maxlen=100) for mod in modalities}

    def select_action(self, modality: str, intrinsic_rewards: Dict[str, float]) -> str:
        """Seleccionar acción"""
        state = torch.tensor([intrinsic_rewards.get('per', 0),
                             intrinsic_rewards.get('igr', 0),
                             intrinsic_rewards.get('um', 0),
                             intrinsic_rewards.get('total', 0)], dtype=torch.float32)

        with torch.no_grad():
            probs = self.policies[modality](state.unsqueeze(0))
            action_idx = torch.argmax(probs).item()

        actions = ['open', 'filtering', 'closed']
        return actions[action_idx]

    def update_gates(self, new_gates: Dict[str, str]):
        """Actualizar gates"""
        self.current_gates.update(new_gates)
        for mod, gate in new_gates.items():
            self.attention_weights[mod] = 1.0 if gate == 'open' else 0.5 if gate == 'filtering' else 0.0

    def update_reward_history(self, modality: str, reward: float):
        """Actualizar historial de recompensas"""
        self.reward_history[modality].append(reward)

    def learn_from_experience(self):
        """Aprender (simplificado)"""
        pass  # Implementación básica por ahora