#!/usr/bin/env python3
"""
Demo de Snake con U-CogNet
Demostración de aprendizaje incremental y memoria en juego de Snake.
"""

import sys
import os
import time
import psutil
import json
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from snake_env import SnakeEnv
from snake_agent import IncrementalSnakeAgent

def get_memory_usage():
    """Obtiene uso de memoria del proceso"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def main():
    print("🐍 U-CogNet - Demo de Snake con Aprendizaje Incremental")
    print("=" * 60)
    
    # Inicializar entorno y agente
    env = SnakeEnv(width=20, height=20)
    agent = IncrementalSnakeAgent()
    
    print("🎮 Entorno: Snake 20x20")
    print("🤖 Agente: Q-learning con memoria episódica")
    print("📊 Midiendo: Memoria y aprendizaje incremental")
    print("-" * 60)
    
    # Estadísticas globales
    global_stats = {
        'total_episodes': 0,
        'total_score': 0,
        'best_score': 0,
        'memory_usage': [],
        'learning_progress': []
    }
    
    try:
        for episode in range(1, 1001):  # 1000 episodios
            state = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done and steps < 1000:  # Límite de pasos
                # Agente elige acción
                action = agent.choose_action(state)
                
                # Ejecutar acción
                next_state, reward, done, _ = env.step(action)
                
                # Agente aprende
                agent.learn(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                steps += 1
            
            # Actualizar estadísticas
            score = env.score
            agent.update_stats(episode_reward, score)
            
            global_stats['total_episodes'] += 1
            global_stats['total_score'] += score
            global_stats['best_score'] = max(global_stats['best_score'], score)
            
            # Medir memoria
            mem_usage = get_memory_usage()
            global_stats['memory_usage'].append(mem_usage)
            
            # Progreso de aprendizaje
            agent_stats = agent.get_learning_stats()
            global_stats['learning_progress'].append({
                'episode': episode,
                'score': score,
                'reward': episode_reward,
                'epsilon': agent.epsilon,
                'q_states': len(agent.q_table),
                'memory_size': agent_stats['memory_size']
            })
            
            # Mostrar progreso cada 50 episodios
            if episode % 50 == 0:
                avg_score = global_stats['total_score'] / global_stats['total_episodes']
                avg_memory = sum(global_stats['memory_usage'][-50:]) / 50
                q_states = len(agent.q_table)
                
                print("2d")
                print(".2f")
                print(f"   🧠 Estados Q aprendidos: {q_states}")
                print(f"   📚 Memoria episódica: {agent_stats['memory_size']}")
                print(f"   🎯 Mejor puntuación: {global_stats['best_score']}")
                print("-" * 50)
                
                # Guardar conocimiento cada 100 episodios
                if episode % 100 == 0:
                    agent.save_knowledge()
                    print("💾 Conocimiento guardado")
    
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrumpida")
    
    # Resultados finales
    print("\n" + "=" * 60)
    print("📊 RESULTADOS FINALES - Snake Learning Demo")
    print("=" * 60)
    
    total_episodes = global_stats['total_episodes']
    avg_score = global_stats['total_score'] / total_episodes
    avg_memory = sum(global_stats['memory_usage']) / len(global_stats['memory_usage'])
    final_q_states = len(agent.q_table)
    
    print(f"🎮 Episodios jugados: {total_episodes}")
    print(".2f")
    print(f"🏆 Mejor puntuación: {global_stats['best_score']}")
    print(".1f")
    print(f"🧠 Estados Q aprendidos: {final_q_states}")
    print(f"📚 Tamaño memoria episódica: {agent.learning_stats['memory_size']}")
    
    # Evaluación del aprendizaje
    recent_scores = [p['score'] for p in global_stats['learning_progress'][-100:]]
    recent_avg = sum(recent_scores) / len(recent_scores) if recent_scores else 0
    
    early_scores = [p['score'] for p in global_stats['learning_progress'][:100]]
    early_avg = sum(early_scores) / len(early_scores) if early_scores else 0
    
    improvement = recent_avg - early_avg
    
    print(f"\n📈 APRENDIZAJE INCREMENTAL")
    print(".2f")
    print(".2f")
    print(".2f")
    
    if improvement > 2:
        print("   ✅ ÉXITO: Aprendizaje significativo detectado!")
        print("   🎉 El agente mejoró su desempeño con el tiempo")
    elif improvement > 0:
        print("   ⚠️ MEJORA MODERADA: Aprendizaje en progreso")
    else:
        print("   ❌ LIMITACIÓN: No se observó mejora significativa")
        print("   💡 Puede requerir más episodios o ajuste de parámetros")
    
    # Guardar resultados finales
    results = {
        'global_stats': global_stats,
        'agent_stats': agent.get_learning_stats(),
        'final_memory_usage': get_memory_usage(),
        'evaluation': {
            'early_avg_score': early_avg,
            'recent_avg_score': recent_avg,
            'improvement': improvement
        }
    }
    
    with open('snake_demo_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Resultados guardados en snake_demo_results.json")
    print("🏁 Demo completada - Sistema U-CogNet operativo en Snake")

if __name__ == "__main__":
    main()