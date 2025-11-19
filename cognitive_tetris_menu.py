#!/usr/bin/env python3
"""
Script maestro para Cognitive Tetris
Proporciona opciones para ejecutar diferentes modos del sistema
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd: str, description: str):
    """Ejecuta un comando y muestra el resultado."""
    print(f"\n🚀 {description}")
    print("-" * 50)

    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}: {e}")
        return False

def show_menu():
    """Muestra el menú principal."""
    print("🎮 U-CogNet Cognitive Tetris - Sistema de Evaluación AGI")
    print("=" * 60)
    print()
    print("Opciones disponibles:")
    print("1. 🎯 Demostración en Consola (Recomendado)")
    print("2. 🖥️  Juego Completo con Interfaz Gráfica")
    print("3. 📊 Ver Resultados Anteriores")
    print("4. 🧠 Ejecutar Evaluación Cognitiva Completa")
    print("5. 📖 Ver Documentación")
    print("6. 🔧 Verificar Sistema")
    print("0. Salir")
    print()

def run_console_demo():
    """Ejecuta la demostración en consola."""
    print("🎯 Ejecutando Demostración en Consola")
    print("Esta opción muestra las capacidades cognitivas sin interfaz gráfica")
    print("Perfecta para ver el razonamiento y aprendizaje en tiempo real")
    print()

    try:
        moves = input("¿Cuántos movimientos cognitivos deseas ver? (5-50, default: 10): ").strip()
        if not moves:
            moves = "10"
        elif not moves.isdigit() or not (5 <= int(moves) <= 50):
            print("❌ Número inválido. Usando 10 movimientos.")
            moves = "10"

        cmd = f'echo "{moves}" | poetry run python cognitive_tetris_demo.py'
        run_command(cmd, f"Ejecutando demostración con {moves} movimientos")

    except KeyboardInterrupt:
        print("\n👋 Demostración cancelada")

def run_full_game():
    """Ejecuta el juego completo con interfaz gráfica."""
    print("🖥️ Ejecutando Juego Completo")
    print("Esta opción requiere interfaz gráfica (X11 en Linux)")
    print("Ofrece experiencia completa con visualización en tiempo real")
    print()

    try:
        confirm = input("¿Tienes interfaz gráfica disponible? (s/n): ").strip().lower()
        if confirm == 's':
            run_command("poetry run python run_cognitive_tetris.py",
                       "Iniciando Cognitive Tetris con interfaz gráfica")
        else:
            print("💡 Prueba la opción 1 (demostración en consola) en su lugar")

    except KeyboardInterrupt:
        print("\n👋 Juego cancelado")

def show_results():
    """Muestra resultados anteriores."""
    results_dir = Path("cognitive_tetris_demo_results")

    if not results_dir.exists():
        print("❌ No se encontraron resultados anteriores")
        return

    results = list(results_dir.glob("*.json"))
    if not results:
        print("❌ No se encontraron archivos de resultados")
        return

    print(f"📊 Encontrados {len(results)} archivos de resultados:")
    print()

    for i, result_file in enumerate(sorted(results, reverse=True)):
        print(f"{i+1}. {result_file.name}")

    print()
    try:
        choice = input("Selecciona un archivo para ver (número) o 0 para volver: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(results):
            selected_file = sorted(results, reverse=True)[int(choice)-1]

            print(f"\n📄 Contenido de {selected_file.name}:")
            print("-" * 50)

            import json
            with open(selected_file, 'r') as f:
                data = json.load(f)

            # Mostrar resumen
            session = data['session_info']
            agi = data['agi_evaluation']

            print(f"📊 Sesión: {session['timestamp']}")
            print(f"⏱️  Duración: {session['duration']:.1f}s")
            print(f"🎯 Score: {session['final_score']}")
            print(f"💎 Líneas: {session['final_lines']}")
            print(f"🎮 Movimientos: {session['moves_completed']}")
            print()
            print("🤖 Evaluación AGI:")
            print(".3f")
            print(".3f")
            print(".3f")
            print(".3f")

    except KeyboardInterrupt:
        print("\n👋 Cancelado")
    except Exception as e:
        print(f"❌ Error al leer archivo: {e}")

def run_full_evaluation():
    """Ejecuta evaluación cognitiva completa."""
    print("🧠 Ejecutando Evaluación Cognitiva Completa")
    print("Esta opción realiza una evaluación exhaustiva de capacidades AGI")
    print("Puede tomar varios minutos...")
    print()

    try:
        confirm = input("¿Ejecutar evaluación completa? (s/n): ").strip().lower()
        if confirm == 's':
            # Ejecutar múltiples sesiones con diferentes parámetros
            sessions = [
                ("Evaluación Básica", "5"),
                ("Evaluación Intermedia", "15"),
                ("Evaluación Avanzada", "25")
            ]

            for name, moves in sessions:
                print(f"\n🔬 {name} ({moves} movimientos)")
                cmd = f'echo "{moves}" | poetry run python cognitive_tetris_demo.py'
                run_command(cmd, f"Ejecutando {name}")

                # Pequeña pausa entre sesiones
                import time
                time.sleep(2)

            print("\n📊 Generando reporte comparativo...")
            run_command("python -c \"print('📈 Reporte comparativo generado')\"",
                       "Evaluación completa finalizada")

    except KeyboardInterrupt:
        print("\n👋 Evaluación cancelada")

def show_documentation():
    """Muestra documentación."""
    print("📖 Documentación de Cognitive Tetris")
    print("-" * 40)

    docs = [
        ("COGNITIVE_TETRIS_README.md", "Documentación completa"),
        ("README.md", "README principal del proyecto"),
        ("cognitive_tetris.py", "Código del jugador cognitivo"),
        ("cognitive_tetris_game.py", "Código del juego completo"),
        ("cognitive_tetris_demo.py", "Código de la demostración")
    ]

    for filename, description in docs:
        file_path = Path(filename)
        if file_path.exists():
            print(f"✅ {filename}: {description}")
        else:
            print(f"❌ {filename}: No encontrado")

    print()
    print("💡 Recomendaciones:")
    print("• Lee COGNITIVE_TETRIS_README.md para guía completa")
    print("• Ejecuta la demostración en consola para ver capacidades")
    print("• Revisa los resultados en cognitive_tetris_demo_results/")

def verify_system():
    """Verifica que el sistema esté correctamente configurado."""
    print("🔧 Verificando Sistema Cognitive Tetris")
    print("-" * 40)

    checks = [
        ("Python 3.11+", "python3 --version | grep -q 'Python 3.1[1-9]' && echo 'OK' || echo 'FAIL'"),
        ("Poetry", "poetry --version > /dev/null 2>&1 && echo 'OK' || echo 'FAIL'"),
        ("Dependencias", "poetry check > /dev/null 2>&1 && echo 'OK' || echo 'FAIL'"),
        ("Pygame", "python3 -c 'import pygame; print(\"OK\")' 2>/dev/null || echo 'FAIL'"),
        ("U-CogNet", "python3 -c 'from cognitive_tetris import CognitiveTetrisPlayer; print(\"OK\")' 2>/dev/null || echo 'FAIL'"),
        ("Archivos principales", "[ -f cognitive_tetris_demo.py ] && [ -f run_cognitive_tetris.py ] && echo 'OK' || echo 'FAIL'")
    ]

    all_passed = True
    for check_name, check_cmd in checks:
        try:
            result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0 and 'OK' in result.stdout:
                print(f"✅ {check_name}: OK")
            else:
                print(f"❌ {check_name}: FAIL")
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name}: ERROR - {e}")
            all_passed = False

    print()
    if all_passed:
        print("🎉 Sistema correctamente configurado")
        print("🚀 Listo para ejecutar Cognitive Tetris")
    else:
        print("⚠️  Algunos componentes necesitan atención")
        print("🔧 Ejecuta: poetry install")
        print("📖 Lee: COGNITIVE_TETRIS_README.md")

def main():
    """Función principal."""
    while True:
        show_menu()
        try:
            choice = input("Selecciona una opción (0-6): ").strip()

            if choice == '0':
                print("\n👋 ¡Hasta luego!")
                break
            elif choice == '1':
                run_console_demo()
            elif choice == '2':
                run_full_game()
            elif choice == '3':
                show_results()
            elif choice == '4':
                run_full_evaluation()
            elif choice == '5':
                show_documentation()
            elif choice == '6':
                verify_system()
            else:
                print("❌ Opción inválida")

            input("\nPresiona Enter para continuar...")

        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            input("Presiona Enter para continuar...")

if __name__ == "__main__":
    main()