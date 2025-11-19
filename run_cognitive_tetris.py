#!/usr/bin/env python3
"""
Script de ejecución para Cognitive Tetris
Inicia el juego con métricas AGI en tiempo real
"""

import asyncio
import sys
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from cognitive_tetris_game import main

if __name__ == "__main__":
    print("🚀 Iniciando U-CogNet Cognitive Tetris...")
    print("🎮 Un juego de Tetris con capacidades cognitivas completas")
    print("📊 Métricas AGI evaluadas en tiempo real")
    print("🧠 Sistema cognitivo: razonamiento, aprendizaje adaptativo, creatividad")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)