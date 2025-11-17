#!/usr/bin/env python3
"""
Examen Completo de Gating Multimodal - Versión AGI Hacker
Implementa todos los criterios de evaluación para sistemas de atención adaptativa
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import time

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from advanced_snake_env import AdvancedSnakeEnv, Modality
from snake_agent import IncrementalSnakeAgent
from multimodal_attention import (
    GatingAttentionController, Modality as AttModality,
    create_visual_signal, create_audio_signal, create_text_signal, create_tactile_signal
)

class MultimodalGatingExaminer:
    """Clase principal para ejecutar el examen completo de gating multimodal"""

    def __init__(self):
        self.env = AdvancedSnakeEnv(width=30, height=30, render=False)
        self.agent = IncrementalSnakeAgent()
        self.attention_controller = GatingAttentionController(
            adaptation_rate=0.3,
            open_threshold=0.5,
            close_threshold=0.2,
            filter_threshold=0.35
        )

        # Resultados
        self.results = {
            'gating_experiment': [],
            'baseline_experiment': [],
            'scalability_test': [],
            'generalization_test': [],
            'diagnostics': []
        }

    def create_multimodal_signals(self, env_data, performance):
        """Crea señales para todas las modalidades"""
        signals = []

        # Señal visual
        visual_data = env_data['visual']
        visual_signal = create_visual_signal(
            data=visual_data,
            confidence=min(0.9, 0.6 + visual_data.get('powerup_near', False) * 0.3),
            priority=0.7
        )
        signals.append(visual_signal)

        # Señal audio
        audio_data = env_data['audio']
        audio_signal = create_audio_signal(
            data=audio_data,
            confidence=min(0.9, 0.4 + len(audio_data.get('events', [])) * 0.2),
            priority=0.8
        )
        signals.append(audio_signal)

        # Señal de texto
        text_data = env_data['text']
        if text_data.get('commands'):
            text_signal = create_text_signal(
                data=text_data,
                confidence=text_data.get('confidence', 0.0),
                priority=0.6
            )
            signals.append(text_signal)

        # Señal táctil
        tactile_data = env_data['tactile']
        tactile_signal = create_tactile_signal(
            data=tactile_data,
            confidence=min(0.8, 0.5 + min(tactile_data.values()) / 20),
            priority=0.9  # Alta prioridad para sensores de proximidad
        )
        signals.append(tactile_signal)

        return signals

    def run_gating_experiment(self, episodes=1000):
        """Ejecuta el experimento principal con gating"""
        print("🧠 Ejecutando Experimento Principal con Gating (1000 episodios)")

        for episode in range(1, episodes + 1):
            state = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0
            episode_score = 0

            episode_diagnostics = {
                'episode': episode,
                'gate_states': [],
                'modality_weights': [],
                'signals_processed': [],
                'performance_trend': []
            }

            while not done and steps < 2000:
                # Obtener datos multimodales
                multimodal_data = self.env.get_multimodal_data()

                # Crear señales
                modality_signals = self.create_multimodal_signals(multimodal_data, episode_score)

                # Procesar con atención
                score_bonus = episode_score * 0.1
                current_performance = (episode_score / max(1, steps + 1)) * (1 + score_bonus)

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

                    # Modality-specific enhancements
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
                        enhancement_factor *= 1.20  # Alta prioridad para táctil

                reward = reward * enhancement_factor

                # Aprender
                self.agent.learn(state, action, reward, next_state, done)

                # Registrar diagnósticos
                episode_diagnostics['gate_states'].append({
                    'step': steps,
                    'gates': {mod.value: attention_state.active_modalities.get(mod, 'CLOSED').name
                             for mod in [AttModality.VISUAL, AttModality.AUDIO, AttModality.TEXT, AttModality.TACTILE]}
                })
                episode_diagnostics['modality_weights'].append({
                    'step': steps,
                    'weights': attention_state.modality_weights
                })
                episode_diagnostics['signals_processed'].append(len(modality_signals))
                episode_diagnostics['performance_trend'].append(current_performance)

                state = next_state
                episode_reward += reward
                episode_score = self.env.score
                steps += 1

            # Registrar episodio
            episode_data = {
                'episode': episode,
                'score': episode_score,
                'reward': episode_reward,
                'steps': steps,
                'diagnostics': episode_diagnostics
            }
            self.results['gating_experiment'].append(episode_data)

            if episode % 100 == 0:
                avg_score = np.mean([ep['score'] for ep in self.results['gating_experiment'][-100:]])
                print(f"Episode {episode}: Avg Score {avg_score:.2f}, Q-States {len(self.agent.q_table)}")

        return self.results['gating_experiment']

    def run_baseline_experiment(self, episodes=500):
        """Ejecuta experimento baseline sin gating (todas las modalidades siempre abiertas)"""
        print("📊 Ejecutando Experimento Baseline sin Gating")

        baseline_env = AdvancedSnakeEnv(width=30, height=30, render=False)
        baseline_agent = IncrementalSnakeAgent()

        for episode in range(1, episodes + 1):
            state = baseline_env.reset()
            done = False
            episode_reward = 0
            steps = 0
            episode_score = 0

            while not done and steps < 2000:
                action = baseline_agent.choose_action(state)
                next_state, reward, done, info = baseline_env.step(action)

                # Reward sin enhancement (todas las modalidades activas)
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

            if episode % 50 == 0:
                avg_score = np.mean([ep['score'] for ep in self.results['baseline_experiment'][-50:]])
                print(f"Baseline Episode {episode}: Avg Score {avg_score:.2f}")

        return self.results['baseline_experiment']

    def run_scalability_test(self):
        """Prueba de escalabilidad agregando más modalidades"""
        print("🔧 Ejecutando Prueba de Escalabilidad")

        # Agregar una quinta modalidad simulada (olfativa)
        class OlfactoryModality:
            def __init__(self):
                self.name = "olfactory"

        olfactory = OlfactoryModality()

        # Extender el controlador de atención
        self.attention_controller.modality_weights[olfactory] = 0.5
        self.attention_controller.active_modalities[olfactory] = self.attention_controller.GateState.CLOSED

        # Ejecutar 100 episodios con la nueva modalidad
        for episode in range(1, 101):
            state = self.env.reset()
            done = False
            episode_score = 0
            steps = 0

            while not done and steps < 2000:
                multimodal_data = self.env.get_multimodal_data()

                # Agregar señal olfativa simulada
                olfactory_signal = type('Signal', (), {
                    'modality': olfactory,
                    'data': {'scent_strength': random.random()},
                    'confidence': random.uniform(0.3, 0.7),
                    'priority': 0.4
                })()

                modality_signals = self.create_multimodal_signals(multimodal_data, episode_score)
                modality_signals.append(olfactory_signal)

                current_performance = episode_score / max(1, steps + 1)
                fused_signal, attention_state = self.attention_controller.process_multimodal_input(
                    modality_signals, current_performance
                )

                action = self.agent.choose_action(state)
                next_state, reward, done, info = self.env.step(action)

                if fused_signal:
                    reward *= 1.0 + (fused_signal.confidence * 0.1)

                self.agent.learn(state, action, reward, next_state, done)

                state = next_state
                episode_score = self.env.score
                steps += 1

            episode_data = {
                'episode': episode,
                'score': episode_score,
                'modalities': 5  # Incluyendo olfativa
            }
            self.results['scalability_test'].append(episode_data)

            if episode % 20 == 0:
                avg_score = np.mean([ep['score'] for ep in self.results['scalability_test'][-20:]])
                print(f"Scalability Episode {episode}: Avg Score {avg_score:.2f}")

        return self.results['scalability_test']

    def run_generalization_test(self):
        """Prueba de generalización cambiando la dinámica del entorno"""
        print("🎯 Ejecutando Prueba de Generalización")

        # Cambiar el entorno: más obstáculos, menos comida
        self.env.width = 25  # Más pequeño
        self.env.height = 25

        # Ejecutar 200 episodios con el entorno modificado
        for episode in range(1, 201):
            state = self.env.reset()
            done = False
            episode_score = 0
            steps = 0

            while not done and steps < 2000:
                multimodal_data = self.env.get_multimodal_data()
                modality_signals = self.create_multimodal_signals(multimodal_data, episode_score)

                current_performance = episode_score / max(1, steps + 1)
                fused_signal, attention_state = self.attention_controller.process_multimodal_input(
                    modality_signals, current_performance
                )

                action = self.agent.choose_action(state)
                next_state, reward, done, info = self.env.step(action)

                if fused_signal:
                    reward *= 1.0 + (fused_signal.confidence * 0.1)

                self.agent.learn(state, action, reward, next_state, done)

                state = next_state
                episode_score = self.env.score
                steps += 1

            episode_data = {
                'episode': episode,
                'score': episode_score,
                'environment': 'modified'
            }
            self.results['generalization_test'].append(episode_data)

            if episode % 50 == 0:
                avg_score = np.mean([ep['score'] for ep in self.results['generalization_test'][-50:]])
                print(f"Generalization Episode {episode}: Avg Score {avg_score:.2f}")

        return self.results['generalization_test']

    def analyze_results(self):
        """Analiza todos los resultados y genera métricas"""
        print("\n📊 ANÁLISIS COMPLETO DE RESULTADOS")
        print("=" * 60)

        # 1. Comparación con baseline
        gating_scores = [ep['score'] for ep in self.results['gating_experiment']]
        baseline_scores = [ep['score'] for ep in self.results['baseline_experiment']]

        gating_avg = np.mean(gating_scores[-500:])  # Últimos 500
        baseline_avg = np.mean(baseline_scores)

        improvement = ((gating_avg - baseline_avg) / baseline_avg) * 100
        print(f"  Score Promedio Gating: {gating_avg:.2f}")
        print(f"  Score Promedio Baseline: {baseline_avg:.2f}")
        print(f"  Mejora sobre Baseline: {improvement:.2f}%")
        # 2. Estabilidad
        gating_std = np.std(gating_scores[-500:])
        baseline_std = np.std(baseline_scores)
        stability_improvement = ((baseline_std - gating_std) / baseline_std) * 100
        print(f"  Mejora de Estabilidad: {stability_improvement:.2f}%")
        # 3. Evolución de gates
        print("\n🧠 EVOLUCIÓN DE GATES:")
        for episode_data in self.results['gating_experiment'][-5:]:  # Últimos 5 episodios
            diag = episode_data['diagnostics']
            if diag['gate_states']:
                final_gates = diag['gate_states'][-1]['gates']
                print(f"  Episodio {episode_data['episode']}: {final_gates}")

        # 4. Escalabilidad
        scale_scores = [ep['score'] for ep in self.results['scalability_test']]
        scale_avg = np.mean(scale_scores)
        print("\n🔧 ESCALABILIDAD:")
        print(f"  Score con 5 modalidades: {scale_avg:.2f}")
        # 5. Generalización
        gen_scores = [ep['score'] for ep in self.results['generalization_test']]
        gen_avg = np.mean(gen_scores)
        print("\n🎯 GENERALIZACIÓN:")
        print(f"  Score en entorno modificado: {gen_avg:.2f}")
        # 6. Métricas de éxito
        print("\n✅ MÉTRICAS DE ÉXITO:")
        print(f"  Score >50% sobre baseline: {'✅' if improvement > 50 else '❌'} ({improvement:.1f}%)")
        print(f"  Reducción de varianza: {'✅' if stability_improvement > 15 else '❌'} ({stability_improvement:.1f}%)")
        print(f"  Score promedio gating: {'✅' if gating_avg > 50 else '❌'} ({gating_avg:.1f})")

        return {
            'gating_avg': gating_avg,
            'baseline_avg': baseline_avg,
            'improvement': improvement,
            'stability_improvement': stability_improvement,
            'scalability_avg': scale_avg,
            'generalization_avg': gen_avg
        }

    def save_results(self):
        """Guarda todos los resultados"""
        with open('multimodal_gating_exam_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print("💾 Resultados guardados en: multimodal_gating_exam_results.json")

def main():
    """Ejecuta el examen completo"""
    print("🚀 INICIANDO EXAMEN COMPLETO DE GATING MULTIMODAL")
    print("Versión AGI Hacker - Evaluación exhaustiva del sistema de atención")
    print("=" * 70)

    examiner = MultimodalGatingExaminer()

    # 1. Experimento principal con gating (1000 episodios)
    examiner.run_gating_experiment(episodes=1000)

    # 2. Experimento baseline sin gating
    examiner.run_baseline_experiment(episodes=500)

    # 3. Prueba de escalabilidad
    examiner.run_scalability_test()

    # 4. Prueba de generalización
    examiner.run_generalization_test()

    # 5. Análisis completo
    metrics = examiner.analyze_results()

    # 6. Guardar resultados
    examiner.save_results()

    print("\n🎉 EXAMEN COMPLETADO")
    print("El sistema de gating multimodal ha sido evaluado exhaustivamente.")
    print(f"Resultado final: {'APROBADO' if metrics['improvement'] > 50 else 'REPROBADO'}")

if __name__ == "__main__":
    main()