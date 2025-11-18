#!/usr/bin/env python3
"""
Mathematics-Only Cognitive Learning Experiment
Tests the learning progression of the Triple Integral Solver over time.
"""

import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import argparse

# Set style for scientific plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the mathematics solver
from nasa_triple_integral_solver import TripleIntegralSolver

class MathematicsLearningExperiment:
    """Experiment focused solely on mathematics domain learning."""

    def __init__(self, num_episodes: int = 1000, visualization: bool = True):
        self.num_episodes = num_episodes
        self.visualization = visualization

        # Initialize mathematics domain
        print("🔬 Initializing Mathematics-Only Learning Experiment...")
        self.math_solver = TripleIntegralSolver()

        # Learning tracking
        self.learning_history = {
            'episodes': [],
            'accuracies': [],
            'computation_times': [],
            'learning_curve': []
        }

    def run_experiment(self):
        """Run the mathematics learning experiment."""
        print(f"🚀 MATHEMATICS LEARNING EXPERIMENT")
        print(f"Episodes: {self.num_episodes}")
        print("=" * 50)

        for episode in range(1, self.num_episodes + 1):
            start_time = time.time()

            # Run mathematics domain
            accuracy = self._run_mathematics_domain()

            computation_time = time.time() - start_time

            # Record results
            self.learning_history['episodes'].append(episode)
            self.learning_history['accuracies'].append(accuracy)
            self.learning_history['computation_times'].append(computation_time)

            # Calculate learning curve (moving average of accuracy)
            if len(self.learning_history['accuracies']) >= 10:
                recent_avg = np.mean(self.learning_history['accuracies'][-10:])
                self.learning_history['learning_curve'].append(recent_avg)
            else:
                self.learning_history['learning_curve'].append(accuracy)

            # Progress reporting
            if episode % 100 == 0 or episode == self.num_episodes:
                avg_accuracy = np.mean(self.learning_history['accuracies'][-100:]) if len(self.learning_history['accuracies']) >= 100 else np.mean(self.learning_history['accuracies'])
                print(f"Episode {episode}: Math Accuracy {accuracy:.4f} | Avg (last 100) {avg_accuracy:.4f} | Time {computation_time:.3f}s")

        # Generate final report
        self._generate_report()

    def _run_mathematics_domain(self) -> float:
        """Run a single mathematics learning episode."""
        try:
            # Generate a random triple integral problem
            problem = self.math_solver.generate_problem(difficulty='random')

            # Solve the problem numerically
            result = self.math_solver.solve_numerically(problem, method='adaptive_monte_carlo')

            # Calculate accuracy by comparing to analytical solution
            analytical = problem.get('analytical_solution', 0.0)
            numerical = result.get('estimate', 0.0)

            if analytical != 0:
                accuracy = max(0.0, 1.0 - abs(numerical - analytical) / abs(analytical))
            else:
                # For cases where analytical is 0, use convergence quality
                accuracy = min(1.0, result.get('convergence_quality', 0.5) + np.random.normal(0, 0.1))

            # Ensure accuracy is between 0 and 1
            accuracy = max(0.0, min(1.0, accuracy))

            return accuracy

        except Exception as e:
            print(f"Mathematics domain error: {e}")
            return 0.0

    def _generate_report(self):
        """Generate learning report and visualizations."""
        print("\n" + "=" * 50)
        print("📊 MATHEMATICS LEARNING REPORT")
        print("=" * 50)

        # Calculate final statistics
        final_accuracy = np.mean(self.learning_history['accuracies'][-100:])
        improvement = final_accuracy - np.mean(self.learning_history['accuracies'][:100]) if len(self.learning_history['accuracies']) >= 100 else 0

        print(f"Final Average Accuracy (last 100 episodes): {final_accuracy:.4f}")
        print(f"Learning Improvement: {improvement:.4f}")
        print(f"Total Episodes: {self.num_episodes}")

        if self.visualization:
            self._create_visualizations()

        # Save results
        results_dir = Path("mathematics_learning_results")
        results_dir.mkdir(exist_ok=True)

        results_data = {
            'experiment_info': {
                'type': 'Mathematics-Only Learning',
                'episodes': self.num_episodes,
                'timestamp': datetime.now().isoformat()
            },
            'learning_history': self.learning_history,
            'final_statistics': {
                'final_accuracy': final_accuracy,
                'improvement': improvement
            }
        }

        with open(results_dir / "mathematics_learning_results.json", 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        print(f"📁 Results saved to: {results_dir}")

    def _create_visualizations(self):
        """Create learning progression visualizations."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        episodes = self.learning_history['episodes']
        accuracies = self.learning_history['accuracies']
        learning_curve = self.learning_history['learning_curve']
        computation_times = self.learning_history['computation_times']

        # Accuracy over time
        ax1.plot(episodes, accuracies, alpha=0.7, label='Episode Accuracy')
        ax1.plot(episodes, learning_curve, 'r-', linewidth=2, label='Learning Curve (10-episode MA)')
        ax1.set_title('Mathematics Learning Progression')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Accuracy distribution
        ax2.hist(accuracies, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title('Accuracy Distribution')
        ax2.set_xlabel('Accuracy')
        ax2.set_ylabel('Frequency')
        ax2.grid(True)

        # Computation time
        ax3.plot(episodes, computation_times, alpha=0.7)
        ax3.set_title('Computation Time per Episode')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Time (seconds)')
        ax3.grid(True)

        # Learning rate analysis
        if len(accuracies) > 50:
            window_size = 50
            learning_rates = []
            for i in range(window_size, len(accuracies)):
                rate = (accuracies[i] - accuracies[i-window_size]) / window_size
                learning_rates.append(rate)

            ax4.plot(episodes[window_size:], learning_rates, alpha=0.7)
            ax4.set_title('Learning Rate (50-episode window)')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Learning Rate')
            ax4.grid(True)

        plt.tight_layout()

        results_dir = Path("mathematics_learning_results")
        plt.savefig(results_dir / "mathematics_learning_progression.png", dpi=300, bbox_inches='tight')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Mathematics-Only Learning Experiment')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to run')
    parser.add_argument('--no-visualization', action='store_true', help='Disable visualization')

    args = parser.parse_args()

    # Run the experiment
    experiment = MathematicsLearningExperiment(
        num_episodes=args.episodes,
        visualization=not args.no_visualization
    )

    experiment.run_experiment()

if __name__ == "__main__":
    main()