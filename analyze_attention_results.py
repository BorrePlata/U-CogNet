#!/usr/bin/env python3
"""
Análisis de Resultados del Experimento Optimizado de Atención Adaptativa
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_attention_results():
    """Analyze the optimized attention experiment results."""

    # Load results
    with open('attention_adaptive_optimized_500_results.json', 'r') as f:
        results = json.load(f)

    print("🔍 Análisis del Experimento Optimizado de Atención Adaptativa")
    print("=" * 70)

    # Extract data
    episodes = results['episodes']
    scores = [ep['score'] for ep in episodes]
    rewards = [ep['reward'] for ep in episodes]

    # Performance analysis
    print("📊 Estadísticas de Rendimiento:")
    print(f"  Score Promedio: {np.mean(scores):.2f}")
    print(f"  Desviación Estándar: {np.std(scores):.2f}")
    print(f"  Score Máximo: {np.max(scores)}")
    print(f"  Score Mínimo: {np.min(scores)}")
    print(f"  Recompensa Promedio: {np.mean(rewards):.2f}")

    # Learning progression (every 50 episodes)
    windows = []
    for i in range(0, len(scores), 50):
        window_scores = scores[i:i+50]
        if len(window_scores) > 0:
            windows.append({
                'episode_range': f"{i+1}-{min(i+50, len(scores))}",
                'avg_score': np.mean(window_scores),
                'max_score': np.max(window_scores)
            })

    print("\n📈 Progresión de Aprendizaje:")
    for window in windows:
        print(f"  Episodios {window['episode_range']}: Avg={window['avg_score']:.2f}, Max={window['max_score']}")

    # Attention evolution analysis
    print("\n🧠 Evolución del Sistema de Atención:")

    # Sample attention states at different points
    checkpoints = [0, 99, 199, 299, 399, 499]  # Episodes 1, 100, 200, 300, 400, 500

    for idx in checkpoints:
        if idx < len(episodes) and episodes[idx]['attention_summary']:
            att = episodes[idx]['attention_summary']
            print(f"\n  Episodio {idx+1}:")
            print(f"    Gates Activos: Visual={att['active_gates']['visual']}, Audio={att['active_gates']['audio']}")
            print(f"    Pesos: Visual={att['modality_weights']['visual']:.3f}, Audio={att['modality_weights']['audio']:.3f}")
            print(f"    Tendencia de Rendimiento: {att['performance_trend']:.3f}")

    # Final analysis
    final_scores = scores[-100:]  # Last 100 episodes
    print("\n🎯 Análisis Final:")
    print(f"  Rendimiento Estable (últimos 100 eps): {np.mean(final_scores):.2f} ± {np.std(final_scores):.2f}")
    print(f"  Mejora Total: {np.mean(final_scores) - np.mean(scores[:100]):.2f} puntos")

    # Success criteria
    target_score = 3.5
    achieved = np.mean(scores) >= target_score
    print(f"  Meta Alcanzada (≥{target_score}): {'✅ SÍ' if achieved else '❌ NO'}")

    # Attention effectiveness
    high_score_episodes = [ep for ep in episodes if ep['score'] >= 5]
    if high_score_episodes:
        print(f"  Episodios con Score ≥5: {len(high_score_episodes)}")
        avg_attention_high = np.mean([ep['attention_summary']['performance_trend']
                                     for ep in high_score_episodes if ep['attention_summary']])
        print(f"  Tendencia de Atención en Scores Altos: {avg_attention_high:.3f}")

    return results

if __name__ == "__main__":
    results = analyze_attention_results()