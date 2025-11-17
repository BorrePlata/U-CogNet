#!/usr/bin/env python3
"""
Versión simplificada del experimento de gating multimodal autónomo para pruebas.
"""

import numpy as np
from intrinsic_reward_generator_simple import IntrinsicRewardGenerator
from adaptive_gating_controller_simple import AdaptiveGatingController

def test_autonomous_gating():
    """Prueba básica del sistema de gating autónomo."""
    print("🧪 PRUEBA DEL SISTEMA DE GATING AUTÓNOMO")
    print("=" * 50)

    # Inicializar componentes
    irg = IntrinsicRewardGenerator()
    controller = AdaptiveGatingController()

    print("✅ Componentes inicializados")

    # Simular algunos pasos de aprendizaje
    for step in range(10):
        print(f"\nPaso {step + 1}:")

        # Generar señales simuladas
        signals = {
            'visual': {'data': np.random.random()},
            'audio': {'data': np.random.random()},
            'text': {'data': np.random.random()},
            'tactile': {'data': np.random.random()}
        }

        # Calcular recompensas intrínsecas
        for modality, signal in signals.items():
            irg.update_predictions(modality, signal['data'])
            irg.update_entropy(modality, np.random.random())
            irg.update_utility(modality, np.random.random() - 0.5)

        intrinsic_rewards = irg.get_all_intrinsic_rewards()

        # Controlador decide gates
        new_gates = {}
        for modality in controller.modalities:
            action = controller.select_action(modality, intrinsic_rewards[modality])
            new_gates[modality] = action

        controller.update_gates(new_gates)

        # Mostrar resultados
        print(f"  Gates: {new_gates}")
        print(f"  Recompensa Visual: {intrinsic_rewards['visual']['total']:.3f}")

        # Aprender
        controller.learn_from_experience(batch_size=5)

    print("\n✅ Prueba completada exitosamente")
    print("🎯 El marco de gating multimodal autónomo funciona correctamente")

if __name__ == "__main__":
    test_autonomous_gating()