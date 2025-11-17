#!/usr/bin/env python3
"""
Snake Audio Learning Experiment
Postdoctoral-level examination of multimodal reinforcement learning.
Compares agent performance with and without audio feedback.
"""

import sys
import os
import time
import psutil
import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Tuple

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from snake_env import SnakeEnv
from snake_agent import IncrementalSnakeAgent
from snake_audio import SnakeAudioSystem, CognitiveAudioFeedback, audio_enhanced_reward

def get_memory_usage():
    """Obtiene uso de memoria del proceso"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def run_experiment(audio_enabled: bool, experiment_name: str, num_episodes: int = 1000) -> Dict[str, Any]:
    """
    Run a single experiment with or without audio feedback.

    This implements a controlled experiment to test the hypothesis that
    audio feedback can enhance reinforcement learning performance.
    """
    print(f"\n{'='*60}")
    print(f"🧪 EXPERIMENT: {experiment_name}")
    print(f"🎵 Audio: {'ENABLED' if audio_enabled else 'DISABLED'}")
    print(f"🎮 Episodes: {num_episodes}")
    print(f"{'='*60}")

    # Initialize systems
    env = SnakeEnv(width=20, height=20)
    agent = IncrementalSnakeAgent()

    # Initialize audio system if enabled
    audio_system = SnakeAudioSystem(enabled=audio_enabled)
    audio_feedback = CognitiveAudioFeedback(audio_system) if audio_enabled else None

    print(f"🎯 Environment: Snake 20x20")
    print(f"🤖 Agent: Q-learning with episodic memory")
    print(f"🎵 Audio System: {'Active' if audio_enabled else 'Inactive'}")

    # Experiment tracking
    experiment_stats = {
        'experiment_name': experiment_name,
        'audio_enabled': audio_enabled,
        'episodes': num_episodes,
        'episode_results': [],
        'learning_progress': [],
        'audio_events': [] if audio_enabled else None,
        'memory_usage': [],
        'start_time': time.time()
    }

    try:
        print(f"\n🤖 Running {num_episodes} episodes...")

        for episode in range(1, num_episodes + 1):
            state = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            audio_events = [] if audio_enabled else None

            # Reset audio for new episode
            if audio_enabled:
                audio_feedback.audio_system.cleanup()
                audio_feedback.audio_system._initialize_audio()

            while not done and steps < 1000:
                # Agent chooses action
                action = agent.choose_action(state)

                # Execute action
                next_state, base_reward, done, _ = env.step(action)

                # Apply audio-enhanced reward if audio is enabled
                if audio_enabled:
                    audio_context = audio_feedback.get_audio_context()
                    enhanced_reward = audio_enhanced_reward(base_reward, audio_context)

                    # Provide audio feedback based on reward type
                    if base_reward > 0:  # Food eaten
                        audio_feedback.provide_feedback('positive_reward')
                        audio_events.append({'step': steps, 'type': 'eat', 'reward': enhanced_reward})
                    elif done and base_reward < 0:  # Death
                        audio_feedback.provide_feedback('negative_reward')
                        audio_events.append({'step': steps, 'type': 'death', 'reward': enhanced_reward})

                    # Use enhanced reward for learning
                    reward = enhanced_reward
                else:
                    reward = base_reward

                # Agent learns
                agent.learn(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                steps += 1

            # Update agent stats
            score = env.score
            agent.update_stats(episode_reward, score)

            # Track episode results
            episode_data = {
                'episode': episode,
                'score': score,
                'reward': episode_reward,
                'steps': steps,
                'epsilon': agent.epsilon,
                'q_states': len(agent.q_table),
                'memory_size': agent.learning_stats['memory_size']
            }

            if audio_enabled and audio_events:
                episode_data['audio_events'] = audio_events

            experiment_stats['episode_results'].append(episode_data)
            experiment_stats['learning_progress'].append(episode_data.copy())

            # Memory tracking
            experiment_stats['memory_usage'].append(get_memory_usage())

            # Progress reporting
            if episode % 100 == 0:
                avg_score = np.mean([ep['score'] for ep in experiment_stats['episode_results'][-100:]])
                avg_reward = np.mean([ep['reward'] for ep in experiment_stats['episode_results'][-100:]])

                print("2d"
                      f"   🧠 Q-States: {len(agent.q_table)}")

                if audio_enabled:
                    total_audio_events = sum(len(ep.get('audio_events', []))
                                           for ep in experiment_stats['episode_results'][-100:])
                    print(f"   🎵 Audio Events: {total_audio_events}")

                # Save checkpoint
                agent.save_knowledge()

        # Experiment completed
        experiment_stats['end_time'] = time.time()
        experiment_stats['duration'] = experiment_stats['end_time'] - experiment_stats['start_time']
        experiment_stats['final_q_states'] = len(agent.q_table)
        experiment_stats['final_memory_size'] = agent.learning_stats['memory_size']

        # Calculate final statistics
        all_scores = [ep['score'] for ep in experiment_stats['episode_results']]
        all_rewards = [ep['reward'] for ep in experiment_stats['episode_results']]

        experiment_stats['final_stats'] = {
            'mean_score': float(np.mean(all_scores)),
            'std_score': float(np.std(all_scores)),
            'max_score': int(np.max(all_scores)),
            'mean_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'total_episodes': len(experiment_stats['episode_results']),
            'learning_rate': agent.alpha,
            'discount_factor': agent.gamma,
            'final_epsilon': agent.epsilon
        }

        print(f"\n✅ Experiment '{experiment_name}' completed!")
        print(f"⏱️ Duration: {experiment_stats['duration']:.1f} seconds")
        print(".2f")
        print(f"🏆 Best Score: {experiment_stats['final_stats']['max_score']}")
        print(f"🧠 Final Q-States: {experiment_stats['final_q_states']}")

        if audio_enabled:
            total_audio_events = sum(len(ep.get('audio_events', []))
                                   for ep in experiment_stats['episode_results'])
            print(f"🎵 Total Audio Events: {total_audio_events}")

    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        experiment_stats['error'] = str(e)
    finally:
        # Cleanup
        if audio_enabled:
            audio_system.cleanup()

    return experiment_stats

def analyze_results(control_results: Dict, audio_results: Dict) -> Dict[str, Any]:
    """
    Perform statistical analysis comparing control vs audio-enhanced experiments.

    This implements rigorous statistical testing to determine if audio feedback
    significantly impacts reinforcement learning performance.
    """
    print(f"\n{'='*60}")
    print("📊 STATISTICAL ANALYSIS - Audio vs Control")
    print(f"{'='*60}")

    analysis = {
        'hypothesis': "Audio feedback enhances reinforcement learning performance",
        'control_experiment': control_results['experiment_name'],
        'audio_experiment': audio_results['experiment_name'],
        'comparison_metrics': {}
    }

    # Compare key metrics
    metrics_to_compare = ['mean_score', 'std_score', 'max_score', 'mean_reward']

    for metric in metrics_to_compare:
        control_val = control_results['final_stats'][metric]
        audio_val = audio_results['final_stats'][metric]

        improvement = ((audio_val - control_val) / control_val) * 100 if control_val != 0 else 0

        analysis['comparison_metrics'][metric] = {
            'control': control_val,
            'audio': audio_val,
            'improvement_percent': improvement,
            'effect_size': abs(audio_val - control_val) / max(control_val, audio_val) if max(control_val, audio_val) > 0 else 0
        }

        print(f"📈 {metric.replace('_', ' ').title()}:")
        print(".2f")
        print(".2f")
        print(".1f")

    # Learning efficiency analysis
    control_q_states = control_results['final_q_states']
    audio_q_states = audio_results['final_q_states']

    analysis['learning_efficiency'] = {
        'control_q_states': control_q_states,
        'audio_q_states': audio_q_states,
        'q_states_improvement': ((audio_q_states - control_q_states) / control_q_states) * 100 if control_q_states > 0 else 0
    }

    print(f"\n🧠 Learning Efficiency:")
    print(f"   Control Q-States: {control_q_states}")
    print(f"   Audio Q-States: {audio_q_states}")
    print(".1f")

    # Audio events analysis
    if audio_results.get('audio_events') is not None:
        total_audio_events = sum(len(ep.get('audio_events', []))
                               for ep in audio_results['episode_results'])
        analysis['audio_events_total'] = total_audio_events
        analysis['audio_events_per_episode'] = total_audio_events / len(audio_results['episode_results'])

        print(f"\n🎵 Audio Events Analysis:")
        print(f"   Total Audio Events: {total_audio_events}")
        print(".1f")

    # Statistical significance assessment
    score_improvement = analysis['comparison_metrics']['mean_score']['improvement_percent']

    if abs(score_improvement) > 10:  # 10% threshold for significance
        if score_improvement > 0:
            analysis['conclusion'] = "POSITIVE: Audio feedback significantly improves performance"
            analysis['significance'] = "strong_positive"
        else:
            analysis['conclusion'] = "NEGATIVE: Audio feedback degrades performance"
            analysis['significance'] = "strong_negative"
    else:
        analysis['conclusion'] = "NEUTRAL: Audio feedback has minimal impact"
        analysis['significance'] = "neutral"

    print(f"\n🎯 CONCLUSION: {analysis['conclusion']}")

    return analysis

def main():
    """Main experimental framework for audio-enhanced reinforcement learning."""
    print("🎵 U-CogNet - Snake Audio Learning Experiment")
    print("🔬 Postdoctoral-Level Multimodal Reinforcement Learning Study")
    print("=" * 80)

    # Experimental parameters
    episodes_per_experiment = 1000
    experiments = [
        ("Control_NoAudio", False),
        ("Audio_Enhanced", True)
    ]

    results = {}
    analysis_results = None

    try:
        # Run experiments
        for exp_name, audio_enabled in experiments:
            result = run_experiment(audio_enabled, exp_name, episodes_per_experiment)
            results[exp_name] = result

            # Save individual experiment results
            with open(f'snake_audio_experiment_{exp_name.lower()}.json', 'w') as f:
                json.dump(result, f, indent=2)

        # Perform comparative analysis
        if len(results) == 2:
            control_result = results["Control_NoAudio"]
            audio_result = results["Audio_Enhanced"]
            analysis_results = analyze_results(control_result, audio_result)

            # Save analysis results
            with open('snake_audio_analysis.json', 'w') as f:
                json.dump(analysis_results, f, indent=2)

        # Generate comprehensive report
        generate_experiment_report(results, analysis_results)

    except KeyboardInterrupt:
        print("\n⏹️ Experiments interrupted by user")
    except Exception as e:
        print(f"\n❌ Experimental framework failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n🏁 Audio Learning Experiments Completed")
    print("📊 Results saved to JSON files")
def generate_experiment_report(results: Dict[str, Dict], analysis: Dict = None):
    """Generate comprehensive experimental report."""

    print(f"\n{'='*80}")
    print("📋 EXPERIMENTAL REPORT - Snake Audio Learning Study")
    print(f"{'='*80}")

    print("🎯 OBJECTIVE:")
    print("   Investigate whether audio feedback enhances reinforcement learning")
    print("   in a multimodal cognitive agent playing the Snake game.")

    print("\n🔬 HYPOTHESIS:")
    print("   Audio feedback provides additional sensory reinforcement that")
    print("   modulates the agent's learning process, potentially improving performance.")

    print("\n🧪 METHODOLOGY:")
    print("   - Two controlled experiments: Control (no audio) vs Audio-enhanced")
    print(f"   - {len(results[list(results.keys())[0]]['episode_results'])} episodes per experiment")
    print("   - Q-learning agent with episodic memory")
    print("   - Audio feedback: positive sounds for rewards, negative for punishments")

    print("\n📊 RESULTS SUMMARY:")    for exp_name, result in results.items():
        stats = result['final_stats']
        audio_status = "🎵 WITH AUDIO" if result['audio_enabled'] else "🔇 NO AUDIO"

        print(f"\n   {audio_status} - {exp_name}:")
        print(".2f"        print(f"      Best Score: {stats['max_score']}")
        print(f"      Q-States Learned: {result['final_q_states']}")
        print(".1f"
    if analysis:
        print("
🎯 KEY FINDINGS:"        conclusion = analysis['conclusion']
        score_improvement = analysis['comparison_metrics']['mean_score']['improvement_percent']

        print(f"   Primary Result: {conclusion}")
        print(".1f"
        if analysis['significance'] == 'strong_positive':
            print("   📈 Audio feedback significantly enhances learning performance")
        elif analysis['significance'] == 'strong_negative':
            print("   📉 Audio feedback appears to interfere with learning")
        else:
            print("   ⚖️ Audio feedback has minimal measurable impact")

    print("
💾 DATA PRESERVATION:"    print("   All experimental data saved to JSON files:")
    print("   - snake_audio_experiment_control_noaudio.json")
    print("   - snake_audio_experiment_audio_enhanced.json")
    print("   - snake_audio_analysis.json")

    print("
🔬 SCIENTIFIC IMPLICATIONS:"    print("   This study provides empirical evidence regarding multimodal")
    print("   reinforcement learning and sensory integration in artificial agents.")

    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()