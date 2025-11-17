#!/usr/bin/env python3
"""
Script para generar gráficas del experimento multimodal gating desde checkpoints guardados.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_checkpoint_data():
    """Cargar datos de todos los checkpoints disponibles."""
    checkpoints = {}
    for i in range(100, 1001, 100):
        filename = f"multimodal_gating_checkpoint_{i}.json"
        if Path(filename).exists():
            try:
                with open(filename, 'r') as f:
                    checkpoints[i] = json.load(f)
                print(f"✅ Checkpoint {i} cargado")
            except Exception as e:
                print(f"❌ Error cargando checkpoint {i}: {e}")
    return checkpoints

def create_plots_from_checkpoints():
    """Crear gráficas completas desde los datos de checkpoints."""

    checkpoints = load_checkpoint_data()
    if not checkpoints:
        print("❌ No se encontraron checkpoints guardados")
        return

    # Extraer datos de todos los checkpoints
    episodes = []
    scores = []
    max_scores = []
    eps_values = []
    attention_weights = {mod: [] for mod in ['visual', 'audio', 'text', 'tactile']}
    gate_states = {mod: [] for mod in ['visual', 'audio', 'text', 'tactile']}

    for ep in sorted(checkpoints.keys()):
        data = checkpoints[ep]
        episodes.append(ep)
        scores.append(data.get('avg_score_last_100', 0))
        max_scores.append(data.get('max_score_last_100', 0))
        eps_values.append(data.get('eps', 0))  # Este campo puede no existir en checkpoints

        # Extraer pesos de atención
        attention = data.get('attention_status', {}).get('modality_weights', {})
        for mod in attention_weights.keys():
            weight = attention.get(mod, 0.5)
            attention_weights[mod].append(weight)

            # Para estados de gates
            gates = data.get('attention_status', {}).get('active_gates', {})
            state = gates.get(mod, 'closed')
            state_num = {'open': 1, 'filtering': 0.5, 'closed': 0}.get(state.lower(), 0)
            gate_states[mod].append(state_num)

    # Crear figura con subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Análisis Completo del Experimento Multimodal Gating - 1000 Episodios', fontsize=16, fontweight='bold')

    # Colores para modalidades
    colors = ['blue', 'red', 'green', 'orange']

    # 1. Score promedio por checkpoint
    axes[0, 0].plot(episodes, scores, 'b-o', linewidth=2, markersize=6, label='Score Promedio')
    axes[0, 0].set_title('Evolución del Score Promedio')
    axes[0, 0].set_xlabel('Episodio')
    axes[0, 0].set_ylabel('Score Promedio')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. Score máximo por checkpoint
    axes[0, 1].plot(episodes, max_scores, 'r-s', linewidth=2, markersize=6, label='Score Máximo')
    axes[0, 1].set_title('Evolución del Score Máximo')
    axes[0, 1].set_xlabel('Episodio')
    axes[0, 1].set_ylabel('Score Máximo')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. EPS (episodios por segundo)
    axes[0, 2].plot(episodes, eps_values, 'g-^', linewidth=2, markersize=6, label='EPS')
    axes[0, 2].set_title('Rendimiento de Ejecución (EPS)')
    axes[0, 2].set_xlabel('Episodio')
    axes[0, 2].set_ylabel('Episodios por Segundo')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()

    # 4. Evolución de pesos de atención
    for i, modality in enumerate(attention_weights.keys()):
        axes[1, 0].plot(episodes, attention_weights[modality],
                       color=colors[i], label=modality, linewidth=2, marker='o', markersize=4)
    axes[1, 0].set_title('Evolución de Pesos de Atención')
    axes[1, 0].set_xlabel('Episodio')
    axes[1, 0].set_ylabel('Peso de Atención')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Estados de gates a lo largo del tiempo
    for i, modality in enumerate(gate_states.keys()):
        axes[1, 1].plot(episodes, gate_states[modality],
                       color=colors[i], label=modality, linewidth=2, marker='s', markersize=4)
    axes[1, 1].set_title('Estados de Gates por Modalidad')
    axes[1, 1].set_xlabel('Episodio')
    axes[1, 1].set_ylabel('Estado (0=Closed, 0.5=Filtering, 1=Open)')
    axes[1, 1].set_yticks([0, 0.5, 1])
    axes[1, 1].set_yticklabels(['Closed', 'Filtering', 'Open'])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Comparación final de pesos vs estados
    modalities = list(attention_weights.keys())
    final_weights = [attention_weights[mod][-1] for mod in modalities]
    final_states = [gate_states[mod][-1] for mod in modalities]

    x = np.arange(len(modalities))
    width = 0.35

    axes[1, 2].bar(x - width/2, final_weights, width, label='Peso Final', alpha=0.7, color='skyblue')
    axes[1, 2].bar(x + width/2, final_states, width, label='Estado Final', alpha=0.7, color='lightcoral')
    axes[1, 2].set_title('Pesos vs Estados Finales')
    axes[1, 2].set_xlabel('Modalidad')
    axes[1, 2].set_ylabel('Valor')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(modalities)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('multimodal_gating_experiment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Análisis completo guardado en: multimodal_gating_experiment_analysis.png")

    # Mostrar estadísticas finales
    print("\n📊 ESTADÍSTICAS FINALES DEL EXPERIMENTO:")
    print(f"  Checkpoints analizados: {len(checkpoints)}")
    print(f"  Score promedio final (últimos 100): {scores[-1]:.3f}")
    print(f"  Score máximo alcanzado: {max(max_scores):.1f}")
    if eps_values and any(x > 0 for x in eps_values):
        print(f"  EPS promedio: {np.mean([x for x in eps_values if x > 0]):.1f}")
        print(f"  EPS máximo: {max(eps_values):.1f}")
    else:
        print("  EPS: No disponible en checkpoints")

    # Análisis de evolución
    print("\n🧠 EVOLUCIÓN DE LA ATENCIÓN:")
    for mod in modalities:
        initial = attention_weights[mod][0]
        final = attention_weights[mod][-1]
        change = final - initial
        print(f"    {mod:6s}: {initial:.3f} → {final:.3f} (cambio: {change:+.3f})")

if __name__ == "__main__":
    create_plots_from_checkpoints()