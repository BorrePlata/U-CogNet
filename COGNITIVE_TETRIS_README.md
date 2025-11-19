# U-CogNet Cognitive Tetris

Un entorno de Tetris avanzado que integra el sistema cognitivo completo de U-CogNet, permitiendo la evaluación de capacidades AGI en tiempo real a través del juego.

## 🎮 Características

### Sistema Cognitivo Completo
- **Razonamiento**: Análisis profundo del estado del tablero y predicción de consecuencias
- **Aprendizaje Adaptativo**: El sistema aprende de cada decisión y mejora con el tiempo
- **Creatividad**: Generación de estrategias innovadoras y soluciones no convencionales
- **Interiorización**: El sistema "piensa" sobre sus decisiones usando el procesador cognitivo

### Métricas AGI en Tiempo Real
- **Adaptabilidad**: Capacidad de aprender y ajustarse a diferentes situaciones
- **Razonamiento**: Calidad y velocidad del proceso de toma de decisiones
- **Aprendizaje**: Eficiencia en el aprendizaje de patrones y estrategias
- **Creatividad**: Generación de soluciones innovadoras
- **Consciencia Situacional**: Comprensión del estado del juego y riesgos

### Interfaz Visual
- **Tablero de Juego**: Visualización clásica del Tetris
- **Métricas en Vivo**: Panel completo con todas las métricas cognitivas
- **Estado Cognitivo**: Visualización del "pensamiento" del sistema
- **Historial de Decisiones**: Seguimiento de la evolución del aprendizaje

## 🚀 Instalación y Ejecución

### Prerrequisitos
```bash
# Instalar dependencias del sistema (Ubuntu/Debian)
sudo apt-get install python3-pygame python3-matplotlib

# O usando conda
conda install pygame matplotlib
```

### Ejecución
```bash
# Desde el directorio raíz del proyecto
python run_cognitive_tetris.py
```

## 🎯 Controles

| Tecla | Acción |
|-------|--------|
| `←` `→` | Mover pieza horizontalmente |
| `↓` | Bajar pieza más rápido |
| `ESPACIO` | Rotar pieza |
| `ENTER` | Movimiento cognitivo inteligente |
| `P` | Pausar/reanudar juego |
| `M` | Mostrar/ocultar métricas |
| `R` | Reiniciar (al terminar) |

## 🧠 Arquitectura Cognitiva

### Componentes Principales

1. **CognitiveTetrisPlayer**: Jugador principal con capacidades cognitivas
   - Integración completa con U-CogNet
   - Toma de decisiones basada en razonamiento
   - Aprendizaje adaptativo continuo

2. **Sistema de Razonamiento**:
   - Análisis de patrones del tablero
   - Evaluación de riesgos y oportunidades
   - Predicción de consecuencias futuras
   - Generación de insights creativos

3. **Aprendizaje Adaptativo**:
   - Memoria de decisiones exitosas
   - Ajuste de estrategias basado en resultados
   - Optimización de timing y posicionamiento
   - Adaptación a diferentes niveles de dificultad

### Métricas Evaluadas

#### Cognitivas
- **Tamaño de Memoria**: Capacidad de retener información
- **Patrones Aprendidos**: Número de estrategias memorizadas
- **Tiempo de Pensamiento**: Velocidad de procesamiento
- **Carga Cognitiva**: Utilización de recursos mentales

#### De Rendimiento
- **Score**: Puntuación total del juego
- **Líneas Limpias**: Eficiencia en la limpieza
- **Huecos**: Calidad estructural del tablero
- **Altura Máxima**: Gestión del espacio vertical

#### De Creatividad
- **Índice de Creatividad**: Uso de estrategias innovadoras
- **Decisiones Innovadoras**: Frecuencia de soluciones no estándar
- **Adaptabilidad**: Capacidad de cambio de estrategia

## 📊 Resultados y Análisis

### Archivos Generados
Al finalizar cada sesión, se generan automáticamente:
- **`cognitive_tetris_results/tetris_session_YYYYMMDD_HHMMSS.json`**: Datos completos de la sesión
- **`cognitive_tetris_results/tetris_report_YYYYMMDD_HHMMSS.txt`**: Reporte resumen con análisis AGI

### Métricas de Evaluación AGI
- **Adaptabilidad (0-1)**: Capacidad de aprendizaje y ajuste
- **Calidad de Razonamiento (0-1)**: Eficiencia en toma de decisiones
- **Eficiencia de Aprendizaje (0-1)**: Velocidad de mejora
- **Score AGI General (0-1)**: Evaluación global de capacidades AGI

## 🔬 Investigación y Desarrollo

### Objetivos de Investigación
1. **Evaluación AGI Práctica**: Medir capacidades cognitivas en entornos dinámicos
2. **Aprendizaje en Tiempo Real**: Observar evolución del comportamiento
3. **Creatividad Artificial**: Generación de estrategias innovadoras
4. **Consciencia Situacional**: Comprensión contextual del entorno

### Aplicaciones
- **Benchmarking AGI**: Estándar para comparar sistemas cognitivos
- **Entrenamiento**: Desarrollo de capacidades cognitivas
- **Debugging**: Análisis detallado del proceso de toma de decisiones
- **Investigación**: Estudio de inteligencia artificial en juegos

## 🛠️ Desarrollo y Extensiones

### Arquitectura Modular
El sistema está diseñado para ser fácilmente extensible:
- Nuevos tipos de piezas
- Diferentes modos de juego
- Algoritmos cognitivos alternativos
- Métricas adicionales de evaluación

### Integración con U-CogNet
- **AudioCognitiveProcessor**: Para análisis de "pensamiento" interno
- **CognitiveCore**: Núcleo de procesamiento cognitivo
- **SemanticFeedback**: Retroalimentación semántica
- **Sistema de Memoria**: Almacenamiento y recuperación de experiencias

## 📈 Ejemplos de Uso

### Sesión Típica
```python
from cognitive_tetris_game import CognitiveTetrisGame
import asyncio

async def run_session():
    game = CognitiveTetrisGame()
    await game.run_game()

asyncio.run(run_session())
```

### Análisis Post-Juego
```python
import json
from pathlib import Path

# Cargar resultados
with open('cognitive_tetris_results/tetris_session_20241218_143022.json', 'r') as f:
    data = json.load(f)

# Analizar evolución
scores = [m['game_metrics']['score'] for m in data['metrics_history']]
creativity = [m['cognitive_metrics']['creativity_avg'] for m in data['metrics_history']]
```

## 🤝 Contribución

### Áreas de Desarrollo
- **Algoritmos Cognitivos**: Mejoras en razonamiento y aprendizaje
- **Interfaz de Usuario**: Visualizaciones más avanzadas
- **Métricas Adicionales**: Nuevas formas de evaluación AGI
- **Modos de Juego**: Variantes del Tetris para diferentes pruebas

### Guías de Contribución
1. Mantener compatibilidad con U-CogNet
2. Documentar nuevas métricas y algoritmos
3. Incluir tests automatizados
4. Actualizar documentación

## 📄 Licencia

Este proyecto es parte de U-CogNet y sigue la misma licencia.

## 🎯 Estado del Proyecto

- ✅ **Implementado**: Sistema cognitivo básico
- ✅ **Implementado**: Interfaz gráfica y controles
- ✅ **Implementado**: Métricas en tiempo real
- ✅ **Implementado**: Sistema de aprendizaje adaptativo
- 🔄 **En Desarrollo**: Algoritmos de creatividad avanzada
- 🔄 **En Desarrollo**: Análisis estadístico profundo
- 📋 **Planificado**: Modos de juego multijugador
- 📋 **Planificado**: Integración con otros juegos

---

*Desarrollado como parte del proyecto U-CogNet - Explorando los límites de la inteligencia artificial cognitiva*