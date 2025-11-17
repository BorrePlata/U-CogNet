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
        print("🤖 Entrenando agente... (esto puede tomar un tiempo)")
        
        for episode in range(1, 5001):  # 5000 episodios para mejor aprendizaje
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
            
            # Mostrar progreso cada 200 episodios
            if episode % 200 == 0:
                avg_score = global_stats['total_score'] / global_stats['total_episodes']
                print("2d")
                print(f"   🎯 Mejor puntuación: {global_stats['best_score']}")
                print(f"   🧠 Estados Q: {len(agent.q_table)}")
                
                # Guardar conocimiento
                agent.save_knowledge()
                print("💾 Conocimiento guardado")
        
        print("\n✅ Entrenamiento completado!")
        
        # Jugar una partida completa y grabarla
        print("🎬 Grabando partida completa del agente entrenado...")
        play_and_record_game(env, agent)
        
        # Crear también una demo corta para ver rápidamente
        print("🎬 Creando demo corta...")
        create_short_demo(env, agent)
    
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
    print(f"\n🏁 Demo completada - Sistema U-CogNet operativo en Snake")

def play_and_record_game(env, agent):
    """Juega una partida completa y la graba en video"""
    import cv2
    import numpy as np
    
    # Configurar grabación
    width, height = 400, 400
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID codec, más confiable
    out = cv2.VideoWriter('snake_gameplay.avi', fourcc, fps, (width, height))
    
    # Resetear entorno
    state = env.reset()
    done = False
    steps = 0
    total_score = 0
    
    print("🎮 Iniciando grabación de partida...")
    
    # Crear ventana para renderizar
    cv2.namedWindow('Snake AI Gameplay', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Snake AI Gameplay', width, height)
    
    try:
        while not done and steps < 2000:  # Límite más alto para partida completa
            # Agente juega (sin aprender)
            action = agent.choose_action(state)
            
            # Ejecutar acción
            next_state, reward, done, _ = env.step(action)
            
            # Actualizar puntuación
            if reward > 0:
                total_score = env.score
            
            state = next_state
            steps += 1
            
            # Crear frame visual
            frame = create_visual_frame(env, width, height)
            
            # Mostrar en ventana
            cv2.imshow('Snake AI Gameplay', frame)
            
            # Escribir al video
            out.write(frame)
            
            # Info cada 50 pasos
            if steps % 50 == 0:
                print(f"🎬 Paso {steps} | Puntuación: {total_score} | Epsilon: {agent.epsilon:.3f}")
            
            # Salir si se presiona ESC
            if cv2.waitKey(100) & 0xFF == 27:
                break
                
    except Exception as e:
        print(f"⚠️ Error durante grabación: {e}")
    
    finally:
        out.release()
        cv2.destroyAllWindows()
    
    print(f"✅ Grabación completada: {steps} pasos, puntuación final: {total_score}")
    print("📹 Video guardado como 'snake_gameplay.avi' (compatible con navegadores)")

def create_short_demo(env, agent):
    """Crea una demo corta de 30 segundos para ver el aprendizaje rápidamente"""
    import cv2
    import numpy as np
    
    # Configurar grabación corta
    width, height = 400, 400
    fps = 15  # Más rápido para demo
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('snake_demo_short.avi', fourcc, fps, (width, height))
    
    # Resetear entorno
    state = env.reset()
    done = False
    steps = 0
    total_score = 0
    
    print("🎮 Creando demo corta...")
    
    try:
        while not done and steps < 450:  # 30 segundos a 15fps
            # Agente juega
            action = agent.choose_action(state)
            
            # Ejecutar acción
            next_state, reward, done, _ = env.step(action)
            
            if reward > 0:
                total_score = env.score
            
            state = next_state
            steps += 1
            
            # Crear frame visual
            frame = create_visual_frame(env, width, height)
            
            # Agregar info de aprendizaje
            cv2.putText(frame, f"AI Learning Demo", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"Score: {total_score}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, f"Steps: {steps}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, f"Epsilon: {agent.epsilon:.3f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Escribir al video
            out.write(frame)
            
            # Info cada 50 pasos
            if steps % 50 == 0:
                print(f"🎬 Demo paso {steps} | Puntuación: {total_score}")
                
    except Exception as e:
        print(f"⚠️ Error en demo corta: {e}")
    
    finally:
        out.release()
    
    print(f"✅ Demo corta completada: {steps} pasos, puntuación final: {total_score}")
    print("📹 Demo guardada como 'snake_demo_short.avi' (compatible con navegadores)")

def create_visual_frame(env, width, height):
    """Crea un frame visual del estado del juego"""
    import cv2
    import numpy as np
    
    # Crear imagen en blanco
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255  # Blanco
    
    # Obtener estado
    state = env._get_state()
    grid = state['grid']
    snake = state['snake']
    food = state['food']
    
    # Tamaño de celda
    cell_size = min(width, height) // max(env.width, env.height)
    offset_x = (width - env.width * cell_size) // 2
    offset_y = (height - env.height * cell_size) // 2
    
    # Dibujar grid
    for y in range(env.height):
        for x in range(env.width):
            cell_value = grid[y, x]
            
            # Color según tipo
            if cell_value == -1:  # Pared
                color = (0, 0, 0)  # Negro
            elif cell_value == 1:  # Cuerpo serpiente
                color = (0, 128, 0)  # Verde oscuro
            elif cell_value == 2:  # Cabeza serpiente
                color = (0, 255, 0)  # Verde brillante
            elif cell_value == 3:  # Comida
                color = (0, 0, 255)  # Rojo
            else:  # Vacío
                color = (255, 255, 255)  # Blanco
            
            # Dibujar rectángulo
            cv2.rectangle(frame, 
                         (offset_x + x * cell_size, offset_y + y * cell_size),
                         (offset_x + (x + 1) * cell_size, offset_y + (y + 1) * cell_size),
                         color, -1)
            
            # Bordes
            cv2.rectangle(frame, 
                         (offset_x + x * cell_size, offset_y + y * cell_size),
                         (offset_x + (x + 1) * cell_size, offset_y + (y + 1) * cell_size),
                         (200, 200, 200), 1)
    
    # Agregar texto de puntuación
    cv2.putText(frame, f"Score: {env.score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f"Steps: {len(snake)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return frame

if __name__ == "__main__":
    main()