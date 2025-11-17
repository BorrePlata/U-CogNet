#!/usr/bin/env python3
"""
Examen Acelerado de Gating Multimodal - Versión de Prueba
Ejecuta una versión reducida del examen completo para validación rápida
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from snake_env import SnakeEnv
from snake_agent import IncrementalSnakeAgent
from multimodal_attention import (
    GatingAttentionController, Modality as AttModality,
    create_visual_signal, create_audio_signal, create_text_signal, create_tactile_signal
)

class FastMultimodalGatingExaminer:
    """Versión acelerada del examinador para pruebas rápidas"""

    def __init__(self):
        self.env = SnakeEnv(width=20, height=20)
        self.agent = IncrementalSnakeAgent()
        self.attention_controller = GatingAttentionController(
            adaptation_rate=0.3,
            open_threshold=0.5,
            close_threshold=0.2,
            filter_threshold=0.35
        )

        self.results = {
            'gating_experiment': [],
            'baseline_experiment': []
        }

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
            'danger_level': len(state['snake']) / 400.0,
            'visual_clarity': 0.8
        }
        visual_signal = create_visual_signal(
            data=visual_data,
            confidence=min(0.9, 0.5 + 0.8),
            priority=0.7
        )
        signals.append(visual_signal)

        # Señal audio (simulada)
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

        # Señal táctil (sensores de proximidad simulados)
        tactile_data = {
            'front_distance': 5,  # Simulado
            'left_distance': 5,
            'right_distance': 5,
            'back_distance': 5
        }
        tactile_signal = create_tactile_signal(
            data=tactile_data,
            confidence=0.5,
            priority=0.9
        )
        signals.append(tactile_signal)

        return signals

    def run_fast_gating_experiment(self, episodes=100):
        """Experimento gating reducido"""
        print("🧠 Experimento Gating Acelerado (100 episodios)")

        for episode in range(1, episodes + 1):
            state = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0
            episode_score = 0

            while not done and steps < 1000:
                # Crear señales multimodales basadas en el estado
                modality_signals = self.create_multimodal_signals(state, episode_score)

                score_bonus = episode_score * 0.1
                current_performance = (episode_score / max(1, steps + 1)) * (1 + score_bonus)

                fused_signal, attention_state = self.attention_controller.process_multimodal_input(
                    modality_signals, current_performance
                )

                action = self.agent.choose_action(state)
                next_state, reward, done, info = self.env.step(action)

                enhancement_factor = 1.0
                if fused_signal:
                    enhancement_factor = 1.0 + (fused_signal.confidence * 0.1) + (fused_signal.priority * 0.05)

                    # Check gate status from attention_state
                    if (fused_signal.modality == AttModality.VISUAL and
                        attention_state.active_modalities[AttModality.VISUAL] == 'open'):
                        enhancement_factor *= 1.15
                    elif (fused_signal.modality == AttModality.AUDIO and
                          attention_state.active_modalities[AttModality.AUDIO] == 'open'):
                        enhancement_factor *= 1.10
                    elif (fused_signal.modality == AttModality.TEXT and
                          attention_state.active_modalities[AttModality.TEXT] == 'open'):
                        enhancement_factor *= 1.05
                    elif (fused_signal.modality == AttModality.TACTILE and
                          attention_state.active_modalities[AttModality.TACTILE] == 'open'):
                        enhancement_factor *= 1.20

                reward = reward * enhancement_factor
                self.agent.learn(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                episode_score = self.env.score
                steps += 1

            episode_data = {
                'episode': episode,
                'score': episode_score,
                'reward': episode_reward,
                'steps': steps,
                'attention_status': self.attention_controller.get_attention_status()
            }
            self.results['gating_experiment'].append(episode_data)

            if episode % 20 == 0:
                avg_score = np.mean([ep['score'] for ep in self.results['gating_experiment'][-20:]])
                print(f"Episode {episode}: Avg Score {avg_score:.2f}")

        return self.results['gating_experiment']

    def run_fast_baseline_experiment(self, episodes=50):
        """Experimento baseline reducido"""
        print("📊 Experimento Baseline Acelerado (50 episodios)")

        baseline_env = SnakeEnv(width=20, height=20)
        baseline_agent = IncrementalSnakeAgent()

        for episode in range(1, episodes + 1):
            state = baseline_env.reset()
            done = False
            episode_reward = 0
            steps = 0
            episode_score = 0

            while not done and steps < 1000:
                action = baseline_agent.choose_action(state)
                next_state, reward, done, info = baseline_env.step(action)

                baseline_agent.learn(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                episode_score = baseline_env.score
                steps += 1

            episode_data = {
                'episode': episode,
                'score': episode_score,
                'reward': episode_reward,
                'steps': steps
            }
            self.results['baseline_experiment'].append(episode_data)

            if episode % 10 == 0:
                avg_score = np.mean([ep['score'] for ep in self.results['baseline_experiment'][-10:]])
                print(f"Baseline Episode {episode}: Avg Score {avg_score:.2f}")

        return self.results['baseline_experiment']

    def analyze_fast_results(self):
        """Análisis rápido de resultados"""
        print("\n📊 ANÁLISIS DE RESULTADOS ACELERADO")
        print("=" * 50)

        gating_scores = [ep['score'] for ep in self.results['gating_experiment']]
        baseline_scores = [ep['score'] for ep in self.results['baseline_experiment']]

        gating_avg = np.mean(gating_scores)
        baseline_avg = np.mean(baseline_scores)

        if baseline_avg > 0:
            improvement = ((gating_avg - baseline_avg) / baseline_avg) * 100
        else:
            improvement = 0

        print(f"Gating Score Promedio: {gating_avg:.2f}")
        print(f"Baseline Score Promedio: {baseline_avg:.2f}")
        print(f"Mejora: {improvement:.2f}%")

        # Evolución de gates
        print("\n🧠 Estados de Gates Finales:")
        final_attention = self.results['gating_experiment'][-1]['attention_status']
        for modality, state in final_attention['active_gates'].items():
            weight = final_attention['modality_weights'][modality]
            print(f"  {modality}: {state} (peso: {weight:.3f})")

        # Métricas de éxito
        print("\n✅ MÉTRICAS DE ÉXITO:")
        print(f"  Score >50% sobre baseline: {'✅' if improvement > 50 else '❌'} ({improvement:.1f}%)")
        print(f"  Score promedio gating > 10: {'✅' if gating_avg > 10 else '❌'} ({gating_avg:.1f})")

        return {
            'gating_avg': gating_avg,
            'baseline_avg': baseline_avg,
            'improvement': improvement
        }

def main():
    print("🚀 EXAMEN ACELERADO DE GATING MULTIMODAL")
    print("Versión de prueba rápida")
    print("=" * 50)

    examiner = FastMultimodalGatingExaminer()

    # Experimentos reducidos
    examiner.run_fast_gating_experiment(episodes=100)
    examiner.run_fast_baseline_experiment(episodes=50)

    # Análisis
    metrics = examiner.analyze_fast_results()

    print("\n🎯 RESULTADO:")
    print(f"{'APROBADO' if metrics['improvement'] > 20 else 'REPROBADO'} - Mejora: {metrics['improvement']:.1f}%")

if __name__ == "__main__":
    main()