#!/usr/bin/env python3
"""
Marco Cognitivo Avanzado - Gating Multimodal Autónomo
Intrinsic Reward Generator (IRG) para recompensas internas autónomas

Este módulo implementa el sistema de recompensas intrínsecas que permite
al gating multimodal aprender autónomamente sin intervención humana.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import math

class IntrinsicRewardGenerator:
    """
    Generador de Recompensas Intrínsecas para Gating Multimodal Autónomo

    Calcula recompensas basadas en:
    - Prediction Error Reward (PER): Sorpresa cognitiva
    - Information Gain Reward (IGR): Reducción de entropía
    - Utility of Modality (UM): Mejora de desempeño por atención
    - Temporal Consistency (TC): Persistencia de utilidad
    """

    def __init__(self,
                 modalities: List[str] = ['visual', 'audio', 'text', 'tactile'],
                 history_length: int = 100,
                 learning_rate: float = 0.01):
        self.modalities = modalities
        self.history_length = history_length
        self.learning_rate = learning_rate

        # Memoria para cada modalidad
        self.prediction_history = {mod: deque(maxlen=history_length) for mod in modalities}
        self.entropy_history = {mod: deque(maxlen=history_length) for mod in modalities}
        self.utility_history = {mod: deque(maxlen=history_length) for mod in modalities}
        self.attention_weights_history = {mod: deque(maxlen=history_length) for mod in modalities}

        # Estado interno para predicciones
        self.last_predictions = {mod: 0.5 for mod in modalities}
        self.last_entropies = {mod: 1.0 for mod in modalities}

        # Pesos de aprendizaje para cada componente de recompensa
        self.reward_weights = {
            'per': 1.0,    # Prediction Error Reward
            'igr': 1.0,    # Information Gain Reward
            'um': 1.0,     # Utility of Modality
            'tc': 0.5      # Temporal Consistency
        }

    def update_predictions(self, modality: str, actual_input: float, predicted_input: Optional[float] = None):
        """
        Actualiza las predicciones para una modalidad específica.

        Args:
            modality: Nombre de la modalidad
            actual_input: Valor real del input (0-1)
            predicted_input: Predicción previa (si None, usa la última)
        """
        if predicted_input is None:
            predicted_input = self.last_predictions[modality]

        # Calcular error de predicción
        prediction_error = abs(predicted_input - actual_input)
        self.prediction_history[modality].append(prediction_error)

        # Actualizar predicción para siguiente paso (media móvil simple)
        self.last_predictions[modality] = 0.9 * predicted_input + 0.1 * actual_input

    def update_entropy(self, modality: str, current_state_entropy: float):
        """
        Actualiza la entropía del estado para una modalidad.

        Args:
            modality: Nombre de la modalidad
            current_state_entropy: Entropía actual del estado (0-1, donde 1 es máxima incertidumbre)
        """
        self.entropy_history[modality].append(current_state_entropy)
        self.last_entropies[modality] = current_state_entropy

    def update_utility(self, modality: str, performance_delta: float, attention_weight: float):
        """
        Actualiza la utilidad de una modalidad.

        Args:
            modality: Nombre de la modalidad
            performance_delta: Cambio en el desempeño global
            attention_weight: Peso de atención actual de la modalidad
        """
        if attention_weight > 0:
            utility = performance_delta / attention_weight
        else:
            utility = 0.0

        self.utility_history[modality].append(utility)
        self.attention_weights_history[modality].append(attention_weight)

    def calculate_prediction_error_reward(self, modality: str) -> float:
        """
        Calcula la recompensa por error de predicción (PER).

        Alto cuando el sistema se sorprende (no anticipa bien).
        """
        if not self.prediction_history[modality]:
            return 0.0

        # Error de predicción promedio reciente
        recent_errors = list(self.prediction_history[modality])[-10:]  # últimos 10
        avg_error = np.mean(recent_errors) if recent_errors else 0.0

        # Normalizar y convertir a recompensa (alto error = alta recompensa)
        reward = min(avg_error * 2.0, 1.0)  # máximo 1.0

        return reward

    def calculate_information_gain_reward(self, modality: str) -> float:
        """
        Calcula la recompensa por ganancia de información (IGR).

        Alto cuando reduce la entropía del estado.
        """
        if len(self.entropy_history[modality]) < 2:
            return 0.0

        # Ganancia de información = H(anterior) - H(actual)
        recent_entropies = list(self.entropy_history[modality])[-5:]  # últimos 5
        if len(recent_entropies) >= 2:
            entropy_reduction = recent_entropies[-2] - recent_entropies[-1]
            reward = max(entropy_reduction, 0.0)  # solo positivo
        else:
            reward = 0.0

        return min(reward, 1.0)  # máximo 1.0

    def calculate_utility_reward(self, modality: str) -> float:
        """
        Calcula la recompensa por utilidad de modalidad (UM).

        Alto cuando la modalidad contribuye al desempeño.
        """
        if not self.utility_history[modality]:
            return 0.0

        # Utilidad promedio reciente
        recent_utilities = list(self.utility_history[modality])[-10:]
        avg_utility = np.mean(recent_utilities) if recent_utilities else 0.0

        # Convertir a recompensa (utilidad positiva = recompensa positiva)
        reward = max(avg_utility, 0.0)

        return min(reward, 1.0)  # máximo 1.0

    def calculate_temporal_consistency_reward(self, modality: str) -> float:
        """
        Calcula la recompensa por consistencia temporal (TC).

        Alto cuando la utilidad se mantiene estable en el tiempo.
        """
        if len(self.utility_history[modality]) < 5:
            return 0.0

        # Calcular varianza de utilidad reciente
        recent_utilities = list(self.utility_history[modality])[-20:]
        if len(recent_utilities) >= 5:
            variance = np.var(recent_utilities)
            # Baja varianza = alta consistencia = alta recompensa
            consistency = 1.0 / (1.0 + variance)  # 0-1, donde 1 es máxima consistencia
            reward = consistency * 0.5  # máximo 0.5 para no dominar otras recompensas
        else:
            reward = 0.0

        return reward

    def calculate_intrinsic_reward(self, modality: str) -> Dict[str, float]:
        """
        Calcula todas las recompensas intrínsecas para una modalidad.

        Returns:
            Dict con todas las componentes de recompensa
        """
        rewards = {
            'per': self.calculate_prediction_error_reward(modality),
            'igr': self.calculate_information_gain_reward(modality),
            'um': self.calculate_utility_reward(modality),
            'tc': self.calculate_temporal_consistency_reward(modality)
        }

        # Recompensa total ponderada
        total_reward = sum(rewards[comp] * self.reward_weights[comp] for comp in rewards)

        rewards['total'] = total_reward

        return rewards

    def get_all_intrinsic_rewards(self) -> Dict[str, Dict[str, float]]:
        """
        Calcula recompensas intrínsecas para todas las modalidades.

        Returns:
            Dict[modality, Dict[reward_type, value]]
        """
        all_rewards = {}
        for modality in self.modalities:
            all_rewards[modality] = self.calculate_intrinsic_reward(modality)

        return all_rewards

    def adapt_reward_weights(self, feedback: Dict[str, float]):
        """
        Adapta los pesos de las recompensas basado en feedback del sistema.

        Args:
            feedback: Dict con feedback por componente de recompensa
        """
        for component in self.reward_weights:
            if component in feedback:
                # Actualización simple basada en feedback
                adjustment = feedback[component] * self.learning_rate
                self.reward_weights[component] += adjustment
                # Mantener en rango razonable
                self.reward_weights[component] = max(0.1, min(2.0, self.reward_weights[component]))

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Retorna estadísticas del generador de recompensas.

        Returns:
            Estadísticas por modalidad
        """
        stats = {}
        for modality in self.modalities:
            rewards = self.calculate_intrinsic_reward(modality)
            stats[modality] = {
                'avg_prediction_error': np.mean(list(self.prediction_history[modality])) if self.prediction_history[modality] else 0.0,
                'avg_entropy': np.mean(list(self.entropy_history[modality])) if self.entropy_history[modality] else 0.0,
                'avg_utility': np.mean(list(self.utility_history[modality])) if self.utility_history[modality] else 0.0,
                'current_total_reward': rewards['total'],
                'reward_components': rewards
            }

        return stats

    def reset(self):
        """Reinicia el estado del generador de recompensas."""
        for modality in self.modalities:
            self.prediction_history[modality].clear()
            self.entropy_history[modality].clear()
            self.utility_history[modality].clear()
            self.attention_weights_history[modality].clear()

        self.last_predictions = dict()
        self.last_entropies = dict()
        for mod in self.modalities:
            self.last_predictions[mod] = 0.5
            self.last_entropies[mod] = 1.0</content>
<parameter name="filePath">/mnt/c/Users/desar/Documents/Science/UCogNet/intrinsic_reward_generator.py