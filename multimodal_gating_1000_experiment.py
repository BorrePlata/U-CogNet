#!/usr/bin/env python3
"""
Experimento Completo de Gating Multimodal - 1000 Episodios
Con visualización en tiempo real, checkpoints cada 100 episodios y gráficas finales
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from collections import defaultdict

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from snake_env import SnakeEnv
from snake_agent import IncrementalSnakeAgent
from multimodal_attention import (
    GatingAttentionController, Modality as AttModality,
    create_visual_signal, create_audio_signal, create_text_signal, create_tactile_signal
)

class MultimodalGatingExperiment:
    """Experimento completo con visualización en tiempo real"""

    def __init__(self):
        self.env = SnakeEnv(width=25, height=25)
        self.agent = IncrementalSnakeAgent()
        self.attention_controller = GatingAttentionController(
            adaptation_rate=0.3,
            open_threshold=0.5,
            close_threshold=0.2,
            filter_threshold=0.35
        )

        # Resultados
        self.results = {
            'episodes': [],
            'checkpoints': [],
            'attention_evolution': [],
            'performance_stats': []
        }

        # Para visualización en tiempo real
        self.episode_scores = []
        self.episode_rewards = []
        self.gate_states_history = defaultdict(list)
        self.modality_weights_history = defaultdict(list)

    def create_multimodal_signals(self, state, performance):
        """Crea señales multimodales basadas en el estado del entorno Snake"""
        signals = []

        # Extraer información del estado
        head_x, head_y = state['snake'][0]
        food_x, food_y = state['food']
        food_distance = abs(head_x - food_x) + abs(head_y - food_y)

        # Señal visual
        visual_data = {
            'snake_length': len(state['snake']),
            'food_distance': food_distance,
            'danger_level': len(state['snake']) / 625.0,  # Normalizado por área
            'visual_clarity': 0.8
        }
        visual_signal = create_visual_signal(
            data=visual_data,
            confidence=min(0.9, 0.6 + 0.8),
            priority=0.7
        )
        signals.append(visual_signal)

        # Señal audio (simulada basada en distancia a comida y longitud serpiente)
        audio_events = []
        if food_distance <= 3:
            audio_events.append('food_near')
        if len(state['snake']) > 5:  # Riesgo de colisión
            audio_events.append('danger_near')

        audio_data = {
            'events': audio_events,
            'intensity': len(audio_events) * 0.3
        }
        audio_signal = create_audio_signal(
            data=audio_data,
            confidence=min(0.9, 0.4 + len(audio_events) * 0.2),
            priority=0.8
        )
        signals.append(audio_signal)

        # Señal de texto (simulada ocasionalmente)
        text_commands = []
        if np.random.random() < 0.1:  # 10% chance
            text_commands.append(np.random.choice([
                'move_forward', 'turn_left', 'turn_right', 'avoid_obstacle', 'seek_food'
            ]))

        text_data = {
            'commands': text_commands,
            'confidence': 0.8 if text_commands else 0.0
        }
        if text_commands:
            text_signal = create_text_signal(
                data=text_data,
                confidence=text_data.get('confidence', 0.0),
                priority=0.6
            )
            signals.append(text_signal)

        # Señal táctil (sensores de proximidad simulados basados en posición)
        # Simular sensores de distancia a paredes y obstáculos
        grid_width, grid_height = 25, 25
        tactile_data = {
            'front_distance': min(10, grid_height - head_y - 1),  # Distancia al borde inferior
            'left_distance': min(10, head_x),  # Distancia al borde izquierdo
            'right_distance': min(10, grid_width - head_x - 1),  # Distancia al borde derecho
            'back_distance': min(10, head_y)  # Distancia al borde superior
        }
        tactile_signal = create_tactile_signal(
            data=tactile_data,
            confidence=min(0.8, 0.5 + min(tactile_data.values()) / 20),
            priority=0.9
        )
        signals.append(tactile_signal)

        return signals

    def run_experiment(self, total_episodes=1000):
        """Ejecuta el experimento completo con visualización en tiempo real"""
        print("🚀 EXPERIMENTO COMPLETO DE GATING MULTIMODAL - 1000 EPISODIOS")
        print("=" * 80)
        print("Checkpoints cada 100 episodios | Visualización en tiempo real")
        print("=" * 80)

        start_time = time.time()

        for episode in range(1, total_episodes + 1):
            episode_start_time = time.time()

            # Reset entorno
            state = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0
            episode_score = 0

            # Datos del episodio para análisis
            episode_attention_states = []
            episode_signals = []

            while not done and steps < 2000:
                # Crear señales multimodales basadas en el estado
                modality_signals = self.create_multimodal_signals(state, episode_score)

                # Calcular rendimiento actual
                score_bonus = episode_score * 0.1
                current_performance = (episode_score / max(1, steps + 1)) * (1 + score_bonus)

                # Procesar con atención
                fused_signal, attention_state = self.attention_controller.process_multimodal_input(
                    modality_signals, current_performance
                )

                # Elegir acción
                action = self.agent.choose_action(state)

                # Ejecutar acción
                next_state, reward, done, info = self.env.step(action)

                # Calcular reward mejorado
                enhancement_factor = 1.0
                if fused_signal:
                    enhancement_factor = 1.0 + (fused_signal.confidence * 0.1) + (fused_signal.priority * 0.05)

                    # Enhancement específico por modalidad
                    if (fused_signal.modality == AttModality.VISUAL and
                        attention_state.active_modalities[AttModality.VISUAL].name == 'OPEN'):
                        enhancement_factor *= 1.15
                    elif (fused_signal.modality == AttModality.AUDIO and
                          attention_state.active_modalities[AttModality.AUDIO].name == 'OPEN'):
                        enhancement_factor *= 1.10
                    elif (fused_signal.modality == AttModality.TEXT and
                          attention_state.active_modalities[AttModality.TEXT].name == 'OPEN'):
                        enhancement_factor *= 1.05
                    elif (fused_signal.modality == AttModality.TACTILE and
                          attention_state.active_modalities[AttModality.TACTILE].name == 'OPEN'):
                        enhancement_factor *= 1.20

                reward = reward * enhancement_factor

                # Aprender
                self.agent.learn(state, action, reward, next_state, done)

                # Registrar datos para análisis
                episode_attention_states.append({
                    'step': steps,
                    'gates': {mod.value: attention_state.active_modalities.get(mod, 'CLOSED').name
                             for mod in [AttModality.VISUAL, AttModality.AUDIO, AttModality.TEXT, AttModality.TACTILE]},
                    'weights': attention_state.modality_weights.copy()
                })
                episode_signals.append(len(modality_signals))

                state = next_state
                episode_reward += reward
                episode_score = self.env.score
                steps += 1

            # Calcular tiempo del episodio
            episode_time = time.time() - episode_start_time

            # Registrar episodio
            episode_data = {
                'episode': episode,
                'score': episode_score,
                'reward': episode_reward,
                'steps': steps,
                'time': episode_time,
                'attention_states': episode_attention_states,
                'signals_processed': episode_signals,
                'final_attention': self.attention_controller.get_attention_status()
            }
            self.results['episodes'].append(episode_data)

            # Actualizar historiales para gráficas
            self.episode_scores.append(episode_score)
            self.episode_rewards.append(episode_reward)

            # Registrar evolución de gates y pesos
            final_attention = episode_data['final_attention']
            for modality, state in final_attention['active_gates'].items():
                self.gate_states_history[modality].append(state)
            for modality, weight in final_attention['modality_weights'].items():
                self.modality_weights_history[modality].append(weight)

            # Mostrar progreso en tiempo real
            if episode % 10 == 0:
                recent_scores = self.episode_scores[-10:]
                avg_score = np.mean(recent_scores)
                max_score = np.max(recent_scores)
                total_time = time.time() - start_time
                eps_per_sec = episode / total_time

                print(f"Episode {episode:4d} | Score: {avg_score:5.1f} | Max: {max_score:3.0f} | EPS: {eps_per_sec:4.2f}")

            # Checkpoint cada 100 episodios
            if episode % 100 == 0:
                self._save_checkpoint(episode)
                self._show_checkpoint_analysis(episode)

        # Análisis final
        self._generate_final_analysis()
        self._create_plots()

        return self.results

    def _save_checkpoint(self, episode):
        """Guarda checkpoint cada 100 episodios"""
        checkpoint = {
            'episode': int(episode),
            'timestamp': str(np.datetime64('now')),
            'avg_score_last_100': float(np.mean(self.episode_scores[-100:])),
            'max_score_last_100': int(np.max(self.episode_scores[-100:])),
            'total_avg_score': float(np.mean(self.episode_scores)),
            'total_max_score': int(np.max(self.episode_scores)),
            'attention_status': self.attention_controller.get_attention_status(),
            'q_states': int(len(self.agent.q_table)),
            'gate_evolution': dict(self.gate_states_history),
            'weight_evolution': {k: [float(x) for x in v] for k, v in self.modality_weights_history.items()}
        }

        self.results['checkpoints'].append(checkpoint)

        # Guardar en archivo
        filename = f'multimodal_gating_checkpoint_{episode}.json'
        with open(filename, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"💾 Checkpoint {episode} guardado: {filename}")

    def _show_checkpoint_analysis(self, episode):
        """Muestra análisis del checkpoint"""
        print(f"\n📊 ANÁLISIS CHECKPOINT {episode}")
        print("-" * 50)

        # Estadísticas de rendimiento
        recent_scores = self.episode_scores[-100:]
        print("🎯 Rendimiento (últimos 100 eps):")
        print(f"  Promedio: {np.mean(recent_scores):.2f}")
        print(f"  Máximo: {np.max(recent_scores)}")
        print(f"  Desviación: {np.std(recent_scores):.2f}")
        # Evolución de atención
        attention = self.attention_controller.get_attention_status()
        print("\n🧠 Estado de Atención:")
        for modality, state in attention['active_gates'].items():
            weight = attention['modality_weights'][modality]
            print(f"  {modality}: {state} (peso: {weight:.3f})")

        # Estadísticas de señales
        total_signals = sum(len(ep['signals_processed']) for ep in self.results['episodes'][-100:])
        avg_signals = total_signals / 100
        print(f"  Señales promedio por episodio: {avg_signals:.1f}")
    def _generate_final_analysis(self):
        """Genera análisis final completo"""
        print("\n🎉 ANÁLISIS FINAL - 1000 EPISODIOS")
        print("=" * 80)

        all_scores = self.episode_scores
        all_rewards = self.episode_rewards

        # Estadísticas generales
        print("📈 ESTADÍSTICAS GENERALES:")
        print(f"  Score Promedio Total: {np.mean(all_scores):.2f}")
        print(f"  Desviación Estándar: {np.std(all_scores):.2f}")
        print(f"  Mejor Score: {np.max(all_scores)}")
        print(f"  Score Promedio Final (últimos 200): {np.mean(all_scores[-200:]):.2f}")
        print(f"  Total Q-States: {len(self.agent.q_table)}")

        # Análisis por fases
        phases = [
            ("Inicio", 0, 200),
            ("Desarrollo", 200, 500),
            ("Madurez", 500, 800),
            ("Estabilidad", 800, 1000)
        ]

        print("\n📊 ANÁLISIS POR FASES:")
        for phase_name, start, end in phases:
            phase_scores = all_scores[start:end]
            if phase_scores:
                print(f"  {phase_name} ({start}-{end}):")
                print(f"    Promedio: {np.mean(phase_scores):.2f}")
                print(f"    Máximo: {np.max(phase_scores)}")

        # Análisis de atención final
        final_attention = self.results['episodes'][-1]['final_attention']
        print("\n🧠 ANÁLISIS DE ATENCIÓN FINAL:")
        print("  Estados de Gates:")
        for modality, state in final_attention['active_gates'].items():
            weight = final_attention['modality_weights'][modality]
            evolution = self.modality_weights_history[modality]
            change = evolution[-1] - evolution[0] if len(evolution) > 1 else 0
            print(f"    {modality}: {state} | Peso: {weight:.3f} | Cambio: {change:+.3f}")

        # Éxitos
        high_scores = len([s for s in all_scores if s >= 5])
        success_rate = high_scores / len(all_scores) * 100
        print("\n✅ MÉTRICAS DE ÉXITO:")
        print(f"  Score Promedio: {np.mean(all_scores):.2f}")
        print(f"  Episodios con Score ≥5: {high_scores}")
        print(f"  Tasa de Éxito: {success_rate:.1f}%")

    def _create_plots(self):
        """Crea gráficas completas del experimento"""
        print("\n📊 GENERANDO GRÁFICAS...")
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Experimento Completo de Gating Multimodal - 1000 Episodios', fontsize=16)

        episodes = range(1, len(self.episode_scores) + 1)

        # 1. Score por episodio
        axes[0, 0].plot(episodes, self.episode_scores, 'b-', alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Score por Episodio')
        axes[0, 0].set_xlabel('Episodio')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Score promedio móvil (ventana 50)
        if len(self.episode_scores) >= 50:
            moving_avg = [np.mean(self.episode_scores[max(0, i-25):i+25])
                         for i in range(len(self.episode_scores))]
            axes[0, 1].plot(episodes, moving_avg, 'r-', linewidth=2)
            axes[0, 1].set_title('Score Promedio Móvil (50 eps)')
            axes[0, 1].set_xlabel('Episodio')
            axes[0, 1].set_ylabel('Score Promedio')
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Evolución de pesos de modalidad
        colors = ['blue', 'red', 'green', 'orange']
        modalities = ['visual', 'audio', 'text', 'tactile']

        for i, modality in enumerate(modalities):
            if modality in self.modality_weights_history:
                weights = self.modality_weights_history[modality]
                if weights:
                    axes[0, 2].plot(range(1, len(weights)+1), weights,
                                   color=colors[i], label=modality, linewidth=2)

        axes[0, 2].set_title('Evolución de Pesos de Modalidad')
        axes[0, 2].set_xlabel('Episodio')
        axes[0, 2].set_ylabel('Peso')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Distribución de scores
        axes[1, 0].hist(self.episode_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Distribución de Scores')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Reward por episodio
        axes[1, 1].plot(episodes, self.episode_rewards, 'g-', alpha=0.7, linewidth=1)
        axes[1, 1].set_title('Reward por Episodio')
        axes[1, 1].set_xlabel('Episodio')
        axes[1, 1].set_ylabel('Reward Total')
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Estados de gates a lo largo del tiempo (últimos 200 episodios)
        recent_episodes = range(max(1, len(episodes)-199), len(episodes)+1)
        gate_data = {mod: [] for mod in modalities}

        for ep_idx in range(max(0, len(self.results['episodes'])-200), len(self.results['episodes'])):
            attention = self.results['episodes'][ep_idx]['final_attention']
            for mod in modalities:
                state = attention['active_gates'].get(mod, 'closed')
                # Convertir a número para plot
                state_num = {'open': 1, 'filtering': 0.5, 'closed': 0}.get(state.lower(), 0)
                gate_data[mod].append(state_num)

        for i, modality in enumerate(modalities):
            if gate_data[modality]:
                axes[1, 2].plot(recent_episodes, gate_data[modality],
                               color=colors[i], label=modality, linewidth=2, marker='o', markersize=3)

        axes[1, 2].set_title('Estados de Gates (últimos 200 eps)')
        axes[1, 2].set_xlabel('Episodio')
        axes[1, 2].set_ylabel('Estado (0=Closed, 0.5=Filtering, 1=Open)')
        axes[1, 2].set_yticks([0, 0.5, 1])
        axes[1, 2].set_yticklabels(['Closed', 'Filtering', 'Open'])
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('multimodal_gating_experiment_1000_plots.png', dpi=300, bbox_inches='tight')
        plt.close()  # Cerrar la figura para liberar memoria
        print("✅ Gráficas guardadas en: multimodal_gating_experiment_1000_plots.png")

        # Guardar datos para análisis adicional
        # Convertir tipos numpy a tipos Python
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        results_copy = convert_numpy_types(self.results)

        with open('multimodal_gating_experiment_1000_results.json', 'w') as f:
            json.dump(results_copy, f, indent=2)

        print("💾 Resultados completos guardados en: multimodal_gating_experiment_1000_results.json")

def main():
    """Función principal"""
    experiment = MultimodalGatingExperiment()
    results = experiment.run_experiment(total_episodes=1000)

    print("\n🎊 EXPERIMENTO COMPLETADO EXITOSAMENTE!")
    print("Archivos generados:")
    print("- multimodal_gating_experiment_1000_results.json")
    print("- multimodal_gating_experiment_1000_plots.png")
    print("- Checkpoints: multimodal_gating_checkpoint_[100,200,...,1000].json")

if __name__ == "__main__":
    main()