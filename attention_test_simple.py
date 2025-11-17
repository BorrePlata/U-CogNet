#!/usr/bin/env python3
"""
Simplified Attention Adaptive Experiment
Focused test of gating attention system with progress saving.
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
from snake_audio import SnakeAudioSystem
from multimodal_attention import (
    GatingAttentionController, Modality,
    create_visual_signal, create_audio_signal
)

def run_attention_adaptive_test(episodes=100, save_progress=True):
    """Run attention adaptive experiment with progress saving."""

    print("🧠 Sistema de Atención Adaptativa - Test Acelerado")
    print(f"Episodios: {episodes}")
    print("=" * 60)

    # Initialize systems
    env = SnakeEnv(width=20, height=20)
    agent = IncrementalSnakeAgent()
    audio_system = SnakeAudioSystem(enabled=True)
    attention_controller = GatingAttentionController()

    results = {
        'experiment_name': 'Attention_Adaptive_Test',
        'episodes': [],
        'attention_history': [],
        'progress_checkpoints': []
    }

    try:
        for episode in range(1, episodes + 1):
            state = env.reset()
            done = False
            episode_reward = 0
            steps = 0

            episode_attention_states = []

            while not done and steps < 1000:
                action = agent.choose_action(state)

                next_state, base_reward, done, _ = env.step(action)

                # Create game state for multimodal processing
                game_state = {
                    'snake': env.snake.copy(),
                    'food': env.food,
                    'score': env.score,
                    'current_score': env.score,
                    'episode_steps': steps + 1,
                    'food_distance': abs(env.snake[0][0] - env.food[0]) + abs(env.snake[0][1] - env.food[1]),
                    'danger_level': len(env.snake) / 400.0,
                    'visual_clarity': 0.8,
                    'audio_context': 'eating' if base_reward > 0 else 'dying' if base_reward < 0 else 'moving'
                }

                # Create modality signals
                modality_signals = []

                # Visual signal
                visual_data = {
                    'snake_length': len(game_state['snake']),
                    'food_distance': game_state['food_distance'],
                    'danger_level': game_state['danger_level']
                }
                visual_signal = create_visual_signal(
                    data=visual_data,
                    confidence=min(0.9, 0.5 + game_state['visual_clarity']),
                    priority=0.7
                )
                modality_signals.append(visual_signal)

                # Audio signal
                audio_data = {
                    'reward_type': 'positive' if base_reward > 0 else 'negative',
                    'intensity': abs(base_reward),
                    'context': game_state['audio_context']
                }
                audio_signal = create_audio_signal(
                    data=audio_data,
                    confidence=0.9 if abs(base_reward) > 0 else 0.4,  # Increased from 0.8/0.3
                    priority=0.8  # Increased from 0.6
                )
                modality_signals.append(audio_signal)

                # Process through attention system
                # Improved performance metric: favors high scores over long episodes
                score_bonus = game_state['current_score'] * 0.1  # Bonus for achieving scores
                current_performance = (game_state['current_score'] / max(1, game_state['episode_steps'])) * (1 + score_bonus)
                fused_signal, attention_state = attention_controller.process_multimodal_input(
                    modality_signals, current_performance
                )

                # Calculate enhanced reward
                enhancement_factor = 1.0
                if fused_signal:
                    enhancement_factor = 1.0 + (fused_signal.confidence * 0.1) + (fused_signal.priority * 0.05)

                    # Modality-specific enhancements
                    if (fused_signal.modality == Modality.VISUAL and
                        attention_state.active_modalities[Modality.VISUAL].name == 'OPEN'):
                        enhancement_factor *= 1.15
                    elif (fused_signal.modality == Modality.AUDIO and
                          attention_state.active_modalities[Modality.AUDIO].name == 'OPEN'):
                        enhancement_factor *= 1.10

                    # Audio feedback only if gate is open
                    if (fused_signal.modality == Modality.AUDIO and
                        attention_state.active_modalities[Modality.AUDIO].name == 'OPEN'):
                        if base_reward > 0:
                            audio_system.play_eat_sound()
                        elif base_reward < 0:
                            audio_system.play_death_sound()

                reward = base_reward * enhancement_factor

                # Learn
                agent.learn(state, action, reward, next_state, done)

                # Record attention state for this step
                episode_attention_states.append(attention_controller.get_attention_status())

                state = next_state
                episode_reward += reward
                steps += 1

            score = env.score
            agent.update_stats(episode_reward, score)

            episode_data = {
                'episode': episode,
                'score': score,
                'reward': episode_reward,
                'steps': steps,
                'q_states': len(agent.q_table),
                'attention_summary': episode_attention_states[-1] if episode_attention_states else None
            }
            results['episodes'].append(episode_data)
            results['attention_history'].append(episode_attention_states[-1] if episode_attention_states else None)

            # Progress reporting and saving
            if episode % 20 == 0:
                avg_score = np.mean([ep['score'] for ep in results['episodes'][-20:]])
                print(f"Episode {episode}: Avg Score {avg_score:.2f}, Q-States {len(agent.q_table)}")

                # Save checkpoint
                if save_progress:
                    checkpoint = {
                        'episode': episode,
                        'avg_score_last_20': avg_score,
                        'total_q_states': len(agent.q_table),
                        'attention_status': attention_controller.get_attention_status(),
                        'timestamp': str(np.datetime64('now'))
                    }
                    results['progress_checkpoints'].append(checkpoint)

                    # Save partial results
                    with open(f'attention_test_checkpoint_{episode}.json', 'w') as f:
                        json.dump(checkpoint, f, indent=2)

    finally:
        audio_system.cleanup()

    # Calculate final statistics
    all_scores = [ep['score'] for ep in results['episodes']]
    all_rewards = [ep['reward'] for ep in results['episodes']]

    results['final_stats'] = {
        'mean_score': float(np.mean(all_scores)),
        'std_score': float(np.std(all_scores)),
        'max_score': int(np.max(all_scores)),
        'mean_reward': float(np.mean(all_rewards)),
        'final_q_states': len(agent.q_table),
        'total_episodes': len(results['episodes'])
    }

    print("\n✅ Experimento Completado")
    print(".2f")
    print(f"Mejor Score: {results['final_stats']['max_score']}")
    print(f"Q-States: {results['final_stats']['final_q_states']}")

    # Save final results
    with open('attention_adaptive_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    # Run test with 100 episodes
    results = run_attention_adaptive_test(episodes=100, save_progress=True)
    print(f"\n📊 Resultados guardados en: attention_adaptive_test_results.json")
    print(f"📈 Checkpoints guardados: {len(results.get('progress_checkpoints', []))}")