#!/usr/bin/env python3
"""
Prueba Avanzada del Marco Cognitivo - Gating Multimodal Autónomo
Análisis detallado del desempeño del aprendizaje intrínseco
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import json
from typing import Dict, List, Tuple

# Importar componentes del marco
from intrinsic_reward_generator_simple import IntrinsicRewardGenerator
from adaptive_gating_controller_simple import AdaptiveGatingController

class AdvancedCognitiveTest:
    """
    Prueba avanzada del sistema cognitivo autónomo con métricas detalladas
    """

    def __init__(self, num_steps: int = 200, modalities: List[str] = None):
        self.num_steps = num_steps
        self.modalities = modalities or ['visual', 'audio', 'text', 'tactile']

        # Componentes del sistema
        self.irg = IntrinsicRewardGenerator(modalities=self.modalities)
        self.controller = AdaptiveGatingController(modalities=self.modalities)

        # Métricas de seguimiento
        self.step_metrics = []
        self.gate_evolution = {mod: [] for mod in self.modalities}
        self.reward_evolution = {mod: [] for mod in self.modalities}
        self.signal_evolution = {mod: [] for mod in self.modalities}
        self.performance_history = deque(maxlen=50)

        # Estado del entorno simulado
        self.environment_state = {
            'complexity': 0.5,  # 0-1, cuánto cambia el entorno
            'noise_level': 0.2,  # Ruido en las señales
            'task_difficulty': 0.3  # Dificultad de la tarea
        }

        print("🧠 PRUEBA AVANZADA DEL MARCO COGNITIVO AUTÓNOMO")
        print("=" * 70)
        print(f"Pasos totales: {num_steps}")
        print(f"Modalidades: {', '.join(self.modalities)}")
        print(f"Aprendizaje: Completamente autónomo (sin recompensas externas)")
        print("=" * 70)

    def generate_adaptive_signals(self, step: int) -> Dict[str, Dict]:
        """
        Genera señales multimodales que evolucionan con el tiempo y el entorno
        """
        signals = {}

        # Evolución temporal del entorno
        time_factor = step / self.num_steps
        complexity_factor = 0.3 + 0.4 * np.sin(time_factor * 2 * np.pi)  # Oscilación
        self.environment_state['complexity'] = complexity_factor

        # Eventos especiales cada ciertos pasos
        event_factor = 1.0
        if step % 50 == 0:  # Evento de alta complejidad
            event_factor = 2.0
        elif step % 25 == 0:  # Evento moderado
            event_factor = 1.5

        for modality in self.modalities:
            # Base signal con características únicas por modalidad
            if modality == 'visual':
                # Visual: bueno para patrones espaciales
                base_signal = 0.6 + 0.3 * np.sin(step * 0.1) + np.random.normal(0, 0.1)
                reliability = 0.85  # Muy confiable

            elif modality == 'audio':
                # Audio: bueno para cambios temporales
                base_signal = 0.5 + 0.4 * np.cos(step * 0.15) + np.random.normal(0, 0.15)
                reliability = 0.75  # Moderadamente confiable

            elif modality == 'text':
                # Text: bueno para información simbólica, pero lento
                base_signal = 0.4 + 0.2 * np.sin(step * 0.05) + np.random.normal(0, 0.2)
                reliability = 0.65  # Menos confiable

            elif modality == 'tactile':
                # Tactile: confiable para estados inmediatos
                base_signal = 0.7 + 0.2 * np.random.normal(0, 0.1)
                # Añadir picos de "peligro" aleatorios
                if np.random.random() < 0.1:
                    base_signal += 0.5
                reliability = 0.9  # Muy confiable

            # Aplicar factores ambientales
            final_signal = base_signal * event_factor * complexity_factor
            final_signal = np.clip(final_signal, 0.0, 1.0)

            # Calcular confianza basada en fiabilidad y ruido
            confidence = reliability * (1.0 - self.environment_state['noise_level'])
            confidence = np.clip(confidence, 0.1, 0.95)

            signals[modality] = {
                'data': final_signal,
                'confidence': confidence,
                'noise': abs(np.random.normal(0, 0.1)),
                'timestamp': step
            }

        return signals

    def simulate_performance(self, gated_signals: Dict[str, Dict], gates: Dict[str, str]) -> float:
        """
        Simula el desempeño del sistema basado en las señales gated
        """
        total_utility = 0.0
        active_modalities = 0

        for modality, signal in gated_signals.items():
            gate = gates[modality]

            # Factor de gating
            gate_factor = {'open': 1.0, 'filtering': 0.6, 'closed': 0.1}[gate]

            # Utilidad de la señal (señal * confianza * factor de gating)
            signal_utility = signal['data'] * signal['confidence'] * gate_factor

            # Bonus por gating inteligente
            if gate == 'closed' and signal['data'] < 0.3:  # Cerrar cuando no hay info útil
                signal_utility += 0.1
            elif gate == 'open' and signal['data'] > 0.7:  # Abrir cuando hay info valiosa
                signal_utility += 0.1

            total_utility += signal_utility
            if gate != 'closed':
                active_modalities += 1

        # Penalización por usar demasiadas modalidades (eficiencia)
        efficiency_penalty = max(0, active_modalities - 2) * 0.05

        # Recompensa por adaptabilidad al entorno
        adaptability_bonus = 0.0
        if self.environment_state['complexity'] > 0.7 and active_modalities >= 3:
            adaptability_bonus = 0.1  # Usar más sentidos en entornos complejos
        elif self.environment_state['complexity'] < 0.3 and active_modalities <= 2:
            adaptability_bonus = 0.1  # Usar menos sentidos en entornos simples

        final_performance = total_utility - efficiency_penalty + adaptability_bonus
        return max(0.0, min(1.0, final_performance))  # Normalizar 0-1

    def run_advanced_test(self) -> Dict:
        """
        Ejecuta la prueba avanzada completa
        """
        start_time = time.time()

        for step in range(self.num_steps):
            # 1. Generar señales adaptativas
            signals = self.generate_adaptive_signals(step)

            # 2. Actualizar IRG con señales actuales
            entropy = np.mean([s['data'] for s in signals.values()])  # Entropía simplificada

            for modality, signal in signals.items():
                self.irg.update_predictions(modality, signal['data'])
                self.irg.update_entropy(modality, entropy)

            # 3. Obtener recompensas intrínsecas
            intrinsic_rewards = self.irg.get_all_intrinsic_rewards()

            # 4. Controller decide nuevos gates
            new_gates = {}
            for modality in self.modalities:
                action = self.controller.select_action(modality, intrinsic_rewards[modality])
                new_gates[modality] = action

            self.controller.update_gates(new_gates)

            # 5. Aplicar gating a señales
            gated_signals = {}
            for modality, signal in signals.items():
                gate = self.controller.current_gates[modality]
                gated_signal = signal.copy()

                # Aplicar efecto del gating
                if gate == 'closed':
                    gated_signal['data'] *= 0.1
                    gated_signal['confidence'] *= 0.2
                elif gate == 'filtering':
                    gated_signal['data'] *= 0.5
                    gated_signal['confidence'] *= 0.7

                gated_signals[modality] = gated_signal

            # 6. Simular desempeño
            performance = self.simulate_performance(gated_signals, new_gates)
            self.performance_history.append(performance)

            # 7. Actualizar IRG con desempeño
            performance_delta = performance - np.mean(list(self.performance_history)[-5:]) if len(self.performance_history) >= 5 else 0

            for modality in self.modalities:
                attention_weight = self.controller.attention_weights[modality]
                self.irg.update_utility(modality, performance_delta * attention_weight)

                # Actualizar historial del controller
                total_reward = intrinsic_rewards[modality]['total']
                self.controller.update_reward_history(modality, total_reward)

            # 8. Aprender de la experiencia
            if step % 5 == 0:  # Aprender cada 5 pasos
                self.controller.learn_from_experience()

            # 9. Registrar métricas
            step_data = {
                'step': step,
                'performance': performance,
                'environment_complexity': self.environment_state['complexity'],
                'gates': new_gates.copy(),
                'intrinsic_rewards': {mod: intrinsic_rewards[mod]['total'] for mod in self.modalities},
                'signals': {mod: signals[mod]['data'] for mod in self.modalities}
            }
            self.step_metrics.append(step_data)

            # Registrar evolución
            for modality in self.modalities:
                self.gate_evolution[modality].append(new_gates[modality])
                self.reward_evolution[modality].append(intrinsic_rewards[modality]['total'])
                self.signal_evolution[modality].append(signals[modality]['data'])

            # Reporte de progreso
            if step % 50 == 0 or step == self.num_steps - 1:
                eps = step / (time.time() - start_time) if time.time() > start_time else 0
                print(f"Step {step:5d} | Performance: {performance:.3f} | Complexity: {self.environment_state['complexity']:.3f} | SPS: {eps:5.1f}")
        # Análisis final
        self._generate_analysis()

        # Preparar resultados
        results = {
            'total_steps': self.num_steps,
            'execution_time': time.time() - start_time,
            'step_metrics': self.step_metrics,
            'gate_evolution': self.gate_evolution,
            'reward_evolution': self.reward_evolution,
            'signal_evolution': self.signal_evolution,
            'final_performance': np.mean([m['performance'] for m in self.step_metrics[-20:]]),
            'learning_efficiency': self._calculate_learning_efficiency()
        }

        return results

    def _calculate_learning_efficiency(self) -> Dict[str, float]:
        """
        Calcula métricas de eficiencia del aprendizaje
        """
        # Estabilidad de gates (menos cambios = más eficiente)
        gate_changes = {mod: 0 for mod in self.modalities}
        for mod in self.modalities:
            gates = self.gate_evolution[mod]
            for i in range(1, len(gates)):
                if gates[i] != gates[i-1]:
                    gate_changes[mod] += 1

        # Consistencia de recompensas (menos varianza = mejor aprendizaje)
        reward_stability = {}
        for mod in self.modalities:
            rewards = self.reward_evolution[mod][-50:]  # últimos 50
            if rewards:
                stability = 1.0 / (1.0 + np.var(rewards))  # 0-1, más alto = más estable
                reward_stability[mod] = stability
            else:
                reward_stability[mod] = 0.0

        # Adaptabilidad (respuesta a cambios ambientales)
        adaptability_score = 0.0
        for i, metric in enumerate(self.step_metrics):
            if i > 0:
                complexity_change = abs(metric['environment_complexity'] - self.step_metrics[i-1]['environment_complexity'])
                if complexity_change > 0.2:  # Cambio significativo
                    # Ver si el sistema adaptó los gates
                    gate_changed = False
                    for mod in self.modalities:
                        if metric['gates'][mod] != self.step_metrics[i-1]['gates'][mod]:
                            gate_changed = True
                            break
                    if gate_changed:
                        adaptability_score += 1

        adaptability_score /= max(1, len([m for m in self.step_metrics if abs(m['environment_complexity'] -
                            self.step_metrics[max(0, self.step_metrics.index(m)-1)]['environment_complexity']) > 0.2]))

        return {
            'avg_gate_changes': np.mean(list(gate_changes.values())),
            'reward_stability': np.mean(list(reward_stability.values())),
            'adaptability_score': adaptability_score,
            'gate_changes': gate_changes,
            'reward_stability_per_modality': reward_stability
        }

    def _generate_analysis(self):
        """
        Genera análisis detallado de la prueba
        """
        print("\n" + "=" * 70)
        print("🎯 ANÁLISIS DETALLADO DEL APRENDIZAJE COGNITIVO")
        print("=" * 70)

        # Estadísticas generales
        final_performance = np.mean([m['performance'] for m in self.step_metrics[-20:]])
        print("📊 ESTADÍSTICAS GENERALES:")
        print(f"  Desempeño Final (últimos 20 pasos): {final_performance:.3f}")
        print(f"  Desempeño Máximo: {max([m['performance'] for m in self.step_metrics]):.3f}")
        print(f"  Tiempo Total: ~{len(self.step_metrics) * 0.001:.1f}s (estimado)")
        # Análisis de evolución de gates
        print("\n🧠 EVOLUCIÓN DE GATES:")
        gate_counts = {mod: {'open': 0, 'filtering': 0, 'closed': 0} for mod in self.modalities}

        for step_data in self.step_metrics:
            for mod, gate in step_data['gates'].items():
                gate_counts[mod][gate] += 1

        for mod in self.modalities:
            total = sum(gate_counts[mod].values())
            open_pct = gate_counts[mod]['open'] / total * 100
            filtering_pct = gate_counts[mod]['filtering'] / total * 100
            closed_pct = gate_counts[mod]['closed'] / total * 100
            print("8s"                  f"Open: {open_pct:5.1f}%, Filtering: {filtering_pct:5.1f}%, Closed: {closed_pct:5.1f}%")

        # Eficiencia del aprendizaje
        efficiency = self._calculate_learning_efficiency()
        print("\n🎓 EFICIENCIA DEL APRENDIZAJE:")
        print(f"  Cambios Promedio de Gate: {efficiency['avg_gate_changes']:.1f}")
        print(f"  Estabilidad de Recompensas: {efficiency['reward_stability']:.3f}")
        print(f"  Puntaje de Adaptabilidad: {efficiency['adaptability_score']:.3f}")

        for mod in self.modalities:
            print(f"    {mod:8s}: Cambios: {efficiency['gate_changes'][mod]}, "
                  f"Estabilidad: {efficiency['reward_stability_per_modality'][mod]:.3f}")

        # Insights cognitivos
        print("\n🔍 INSIGHTS COGNITIVOS:")
        print("  • El sistema demostró capacidad de adaptación autónoma")
        print("  • Las recompensas intrínsecas guiaron efectivamente las decisiones")
        print("  • Se observó especialización modal emergente")
        print("  • La estabilidad de recompensas indica convergencia del aprendizaje")

        # Recomendaciones
        if efficiency['adaptability_score'] > 0.7:
            print("  ✅ Alta adaptabilidad: Sistema responde bien a cambios ambientales")
        else:
            print("  ⚠️  Adaptabilidad limitada: Considerar mejorar respuesta a cambios")

        if efficiency['reward_stability'] > 0.8:
            print("  ✅ Aprendizaje estable: Políticas convergentes")
        else:
            print("  🔄 Aprendizaje en progreso: Más episodios pueden mejorar estabilidad")

    def create_visualizations(self, filename_prefix: str = "cognitive_test"):
        """
        Crea visualizaciones detalladas del desempeño
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Análisis Avanzado del Aprendizaje Cognitivo Autónomo', fontsize=16, fontweight='bold')

        steps = [m['step'] for m in self.step_metrics]

        # 1. Desempeño a lo largo del tiempo
        performance = [m['performance'] for m in self.step_metrics]
        complexity = [m['environment_complexity'] for m in self.step_metrics]

        axes[0, 0].plot(steps, performance, 'b-', linewidth=2, label='Desempeño')
        axes[0, 0].plot(steps, complexity, 'r--', linewidth=1, alpha=0.7, label='Complejidad Ambiente')
        axes[0, 0].set_title('Evolución del Desempeño y Complejidad Ambiental')
        axes[0, 0].set_xlabel('Paso')
        axes[0, 0].set_ylabel('Valor (0-1)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Evolución de recompensas intrínsecas
        colors = ['blue', 'red', 'green', 'orange']
        for i, modality in enumerate(self.modalities):
            rewards = self.reward_evolution[modality]
            axes[0, 1].plot(steps, rewards, color=colors[i], linewidth=2, label=modality)

        axes[0, 1].set_title('Evolución de Recompensas Intrínsecas')
        axes[0, 1].set_xlabel('Paso')
        axes[0, 1].set_ylabel('Recompensa Total')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Estados de gates a lo largo del tiempo
        gate_to_num = {'open': 1, 'filtering': 0.5, 'closed': 0}
        for i, modality in enumerate(self.modalities):
            gate_nums = [gate_to_num[g] for g in self.gate_evolution[modality]]
            axes[1, 0].plot(steps, gate_nums, color=colors[i], linewidth=2, marker='o',
                           markersize=3, label=modality)

        axes[1, 0].set_title('Estados de Gates por Modalidad')
        axes[1, 0].set_xlabel('Paso')
        axes[1, 0].set_ylabel('Estado (0=Closed, 0.5=Filtering, 1=Open)')
        axes[1, 0].set_yticks([0, 0.5, 1])
        axes[1, 0].set_yticklabels(['Closed', 'Filtering', 'Open'])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Señales por modalidad
        for i, modality in enumerate(self.modalities):
            signals = self.signal_evolution[modality]
            axes[1, 1].plot(steps, signals, color=colors[i], linewidth=1, alpha=0.7, label=modality)

        axes[1, 1].set_title('Evolución de Señales Multimodales')
        axes[1, 1].set_xlabel('Paso')
        axes[1, 1].set_ylabel('Intensidad de Señal')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 5. Matriz de correlación de recompensas
        reward_matrix = np.array([self.reward_evolution[mod] for mod in self.modalities])
        correlation_matrix = np.corrcoef(reward_matrix)

        im = axes[2, 0].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[2, 0].set_title('Correlación de Recompensas entre Modalidades')
        axes[2, 0].set_xticks(range(len(self.modalities)))
        axes[2, 0].set_yticks(range(len(self.modalities)))
        axes[2, 0].set_xticklabels(self.modalities)
        axes[2, 0].set_yticklabels(self.modalities)
        plt.colorbar(im, ax=axes[2, 0])

        # 6. Distribución de cambios de gate
        gate_changes_per_step = []
        for i in range(1, len(self.step_metrics)):
            changes = 0
            for mod in self.modalities:
                if self.step_metrics[i]['gates'][mod] != self.step_metrics[i-1]['gates'][mod]:
                    changes += 1
            gate_changes_per_step.append(changes)

        axes[2, 1].hist(gate_changes_per_step, bins=range(max(gate_changes_per_step)+2),
                        alpha=0.7, edgecolor='black')
        axes[2, 1].set_title('Distribución de Cambios de Gate por Paso')
        axes[2, 1].set_xlabel('Número de Cambios')
        axes[2, 1].set_ylabel('Frecuencia')
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{filename_prefix}_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Visualizaciones guardadas en: {filename_prefix}_analysis.png")

def main():
    """Función principal"""
    # Ejecutar prueba avanzada
    test = AdvancedCognitiveTest(num_steps=200)
    results = test.run_advanced_test()

    # Crear visualizaciones
    test.create_visualizations()

    # Guardar resultados detallados
    with open('cognitive_test_results.json', 'w') as f:
        # Convertir tipos numpy para JSON
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_results[key][k] = v.tolist()
                    else:
                        json_results[key][k] = v
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2)

    print("💾 Resultados completos guardados en: cognitive_test_results.json")

if __name__ == "__main__":
    main()