#!/usr/bin/env python3
"""
Intrinsic Reward Generator (IRG) para recompensas internas autónomas
"""

import numpy as np
from typing import Dict, List
from collections import deque

class IntrinsicRewardGenerator:
    """Generador de Recompensas Intrínsecas"""

    def __init__(self, modalities: List[str] = ['visual', 'audio', 'text', 'tactile']):
        self.modalities = modalities
        self.prediction_history = {mod: deque(maxlen=100) for mod in modalities}
        self.entropy_history = {mod: deque(maxlen=100) for mod in modalities}
        self.utility_history = {mod: deque(maxlen=100) for mod in modalities}

    def update_predictions(self, modality: str, actual_input: float):
        """Actualizar predicciones"""
        self.prediction_history[modality].append(actual_input)

    def update_entropy(self, modality: str, entropy: float):
        """Actualizar entropía"""
        self.entropy_history[modality].append(entropy)

    def update_utility(self, modality: str, utility: float):
        """Actualizar utilidad"""
        self.utility_history[modality].append(utility)

    def calculate_intrinsic_reward(self, modality: str) -> Dict[str, float]:
        """Calcular recompensa intrínseca"""
        per = np.mean(list(self.prediction_history[modality])[-5:]) if self.prediction_history[modality] else 0.0
        igr = -np.mean(list(self.entropy_history[modality])[-5:]) if self.entropy_history[modality] else 0.0
        um = np.mean(list(self.utility_history[modality])[-5:]) if self.utility_history[modality] else 0.0

        total = per + igr + um
        return {'per': per, 'igr': igr, 'um': um, 'total': total}

    def get_all_intrinsic_rewards(self) -> Dict[str, Dict[str, float]]:
        """Obtener recompensas para todas las modalidades"""
        return {mod: self.calculate_intrinsic_reward(mod) for mod in self.modalities}