#!/usr/bin/env python3
"""
Snake Audio Learning Experiment - Simplified Version
Postdoctoral-level examination of multimodal reinforcement learning.
"""

import sys
import os
import time
import json
from pathlib import Path
import numpy as np

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from snake_env import SnakeEnv
from snake_agent import IncrementalSnakeAgent
from snake_audio import SnakeAudioSystem, CognitiveAudioFeedback, audio_enhanced_reward

def run_experiment(audio_enabled: bool, experiment_name: str, num_episodes: int = 500) -> dict:
    """Run experiment with or without audio feedback."""

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Audio: {'ENABLED' if audio_enabled else 'DISABLED'}")
    print(f"Episodes: {num_episodes}")
    print(f"{'='*60}")

    # Initialize systems
    env = SnakeEnv(width=20, height=20)
    agent = IncrementalSnakeAgent()

    audio_system = SnakeAudioSystem(enabled=audio_enabled)
    audio_feedback = CognitiveAudioFeedback(audio_system) if audio_enabled else None

    results = {
        'experiment_name': experiment_name,
        'audio_enabled': audio_enabled,
        'episodes': [],
        'final_stats': {}
    }

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < 1000:
            action = agent.choose_action(state)

            next_state, base_reward, done, _ = env.step(action)

            if audio_enabled:
                audio_context = audio_feedback.get_audio_context()
                enhanced_reward = audio_enhanced_reward(base_reward, audio_context)

                if base_reward > 0:
                    audio_feedback.provide_feedback('positive_reward')
                elif done and base_reward < 0:
                    audio_feedback.provide_feedback('negative_reward')

                reward = enhanced_reward
            else:
                reward = base_reward

            agent.learn(state, action, reward, next_state, done)

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
            'q_states': len(agent.q_table)
        }
        results['episodes'].append(episode_data)

        if episode % 100 == 0:
            avg_score = np.mean([ep['score'] for ep in results['episodes'][-100:]])
            print(f"Episode {episode}: Avg Score {avg_score:.2f}, Q-States {len(agent.q_table)}")
    # Calculate final statistics
    all_scores = [ep['score'] for ep in results['episodes']]
    all_rewards = [ep['reward'] for ep in results['episodes']]

    results['final_stats'] = {
        'mean_score': float(np.mean(all_scores)),
        'std_score': float(np.std(all_scores)),
        'max_score': int(np.max(all_scores)),
        'mean_reward': float(np.mean(all_rewards)),
        'final_q_states': len(agent.q_table)
    }

    print(f"\nCompleted {experiment_name}")
    print(".2f")
    print(f"Best Score: {results['final_stats']['max_score']}")
    print(f"Q-States: {results['final_stats']['final_q_states']}")

    if audio_enabled:
        audio_system.cleanup()

    return results

def main():
    print("Snake Audio Learning Experiment")
    print("Postdoctoral-Level Multimodal Reinforcement Learning Study")
    print("=" * 80)

    # Run experiments
    print("\nRunning Control Experiment (No Audio)...")
    control_results = run_experiment(False, "Control_NoAudio", 500)

    print("\nRunning Audio Experiment...")
    audio_results = run_experiment(True, "Audio_Enhanced", 500)

    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS RESULTS")
    print(f"{'='*60}")

    control_score = control_results['final_stats']['mean_score']
    audio_score = audio_results['final_stats']['mean_score']
    improvement = ((audio_score - control_score) / control_score) * 100 if control_score > 0 else 0

    print(".2f")
    print(".2f")
    print(".1f")

    if abs(improvement) > 5:
        if improvement > 0:
            print("CONCLUSION: Audio feedback enhances learning performance")
        else:
            print("CONCLUSION: Audio feedback degrades learning performance")
    else:
        print("CONCLUSION: Audio feedback has minimal impact")

    # Save results
    with open('snake_audio_control.json', 'w') as f:
        json.dump(control_results, f, indent=2)

    with open('snake_audio_enhanced.json', 'w') as f:
        json.dump(audio_results, f, indent=2)

    print("\nResults saved to JSON files")

if __name__ == "__main__":
    main()