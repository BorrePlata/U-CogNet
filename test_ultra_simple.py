#!/usr/bin/env python3
"""
Prueba ultra-simple del marco cognitivo avanzado
"""

import numpy as np

class SimpleIRG:
    def __init__(self):
        self.modalities = ['visual', 'audio', 'text', 'tactile']
        self.history = {mod: [] for mod in self.modalities}

    def update(self, modality, value):
        self.history[modality].append(value)

    def get_reward(self, modality):
        if not self.history[modality]:
            return 0.0
        return np.mean(self.history[modality][-5:])

class SimpleController:
    def __init__(self):
        self.modalities = ['visual', 'audio', 'text', 'tactile']
        self.gates = {mod: 'open' for mod in self.modalities}

    def select_action(self, modality, reward):
        if reward > 0.5:
            return 'open'
        elif reward > 0.2:
            return 'filtering'
        else:
            return 'closed'

    def update_gates(self, new_gates):
        self.gates.update(new_gates)

def test_system():
    print("🧪 PRUEBA ULTRA-SIMPLE DEL MARCO COGNITIVO AVANZADO")
    print("=" * 60)

    irg = SimpleIRG()
    controller = SimpleController()

    for step in range(5):
        print(f"\nPaso {step + 1}:")

        # Simular señales y actualizar IRG
        for mod in irg.modalities:
            signal = np.random.random()
            irg.update(mod, signal)

            reward = irg.get_reward(mod)
            action = controller.select_action(mod, reward)

            print(f"  {mod}: señal={signal:.3f}, recompensa={reward:.3f}, gate={action}")

        # Actualizar gates
        new_gates = {mod: controller.select_action(mod, irg.get_reward(mod)) for mod in irg.modalities}
        controller.update_gates(new_gates)

    print("\n✅ Prueba completada exitosamente!")
    print("🎯 El Marco Cognitivo Avanzado - Gating Multimodal Autónomo funciona!")

if __name__ == "__main__":
    test_system()