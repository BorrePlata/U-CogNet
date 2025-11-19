#!/usr/bin/env python3
"""
Script de instalación para Cognitive Tetris
Instala las dependencias necesarias para ejecutar el juego
"""

import subprocess
import sys
from pathlib import Path

def run_command(command: str, description: str):
    """Ejecuta un comando y maneja errores."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}: {e}")
        print(f"Output: {e.output}")
        return False

def install_dependencies():
    """Instala las dependencias necesarias."""

    print("🚀 Instalando dependencias para Cognitive Tetris...")
    print("=" * 50)

    # Verificar Python
    if sys.version_info < (3, 11):
        print(f"❌ Se requiere Python 3.11+. Versión actual: {sys.version}")
        return False

    print(f"✅ Python {sys.version.split()[0]} detectado")

    # Instalar/actualizar Poetry si no está disponible
    try:
        subprocess.run(["poetry", "--version"], check=True, capture_output=True)
        print("✅ Poetry detectado")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("📦 Instalando Poetry...")
        if not run_command("curl -sSL https://install.python-poetry.org | python3 -", "Instalación de Poetry"):
            return False

    # Instalar dependencias del proyecto
    if not run_command("poetry install", "Instalación de dependencias del proyecto"):
        return False

    # Instalar dependencias adicionales del sistema (si es necesario)
    system_deps = [
        "python3-pygame",
        "python3-matplotlib",
        "python3-opencv",
        "libsdl2-dev",  # Para pygame en algunos sistemas
        "libsdl2-image-dev",
        "libsdl2-mixer-dev",
        "libsdl2-ttf-dev"
    ]

    # Detectar sistema operativo
    import platform
    system = platform.system().lower()

    if system == "linux":
        print("🐧 Detectado sistema Linux")
        try:
            # Verificar si apt está disponible
            subprocess.run(["which", "apt"], check=True, capture_output=True)
            if not run_command("sudo apt-get update", "Actualización de lista de paquetes"):
                print("⚠️  No se pudo actualizar lista de paquetes (continuando...)")

            # Instalar solo si no están ya instalados
            for dep in system_deps:
                try:
                    subprocess.run(["dpkg", "-s", dep], check=True, capture_output=True)
                    print(f"✅ {dep} ya está instalado")
                except subprocess.CalledProcessError:
                    if not run_command(f"sudo apt-get install -y {dep}", f"Instalación de {dep}"):
                        print(f"⚠️  No se pudo instalar {dep} (continuando...)")

        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  No se detectó apt. Instale manualmente las dependencias del sistema si es necesario.")

    elif system == "darwin":  # macOS
        print("🍎 Detectado sistema macOS")
        try:
            subprocess.run(["which", "brew"], check=True, capture_output=True)
            for dep in ["sdl2", "sdl2_image", "sdl2_mixer", "sdl2_ttf"]:
                if not run_command(f"brew install {dep}", f"Instalación de {dep}"):
                    print(f"⚠️  No se pudo instalar {dep} (continuando...)")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  No se detectó Homebrew. Instale manualmente las dependencias si es necesario.")

    elif system == "windows":
        print("🪟 Detectado sistema Windows")
        print("ℹ️  En Windows, pygame debería instalarse automáticamente con Poetry.")
        print("ℹ️  Si hay problemas, instale Visual Studio Build Tools.")

    # Verificar instalación
    print("\n🔍 Verificando instalación...")

    # Verificar imports críticos
    imports_to_check = [
        ("pygame", "Interfaz gráfica"),
        ("numpy", "Computación numérica"),
        ("matplotlib", "Visualización"),
        ("cv2", "Computer vision"),
        ("librosa", "Procesamiento de audio")
    ]

    all_good = True
    for module, description in imports_to_check:
        try:
            if module == "cv2":
                import cv2
            else:
                __import__(module)
            print(f"✅ {description}: {module}")
        except ImportError:
            print(f"❌ {description}: {module} no disponible")
            all_good = False

    # Verificar módulos del proyecto
    try:
        from cognitive_tetris import CognitiveTetrisPlayer
        print("✅ Cognitive Tetris: Módulo principal")
    except ImportError as e:
        print(f"❌ Cognitive Tetris: Error al importar - {e}")
        all_good = False

    if all_good:
        print("\n🎉 ¡Instalación completada exitosamente!")
        print("\n🚀 Para ejecutar Cognitive Tetris:")
        print("   python run_cognitive_tetris.py")
        print("\n📖 Lee COGNITIVE_TETRIS_README.md para más información")
        return True
    else:
        print("\n❌ Algunos componentes no se instalaron correctamente.")
        print("🔧 Revisa los errores arriba y vuelve a intentar.")
        return False

if __name__ == "__main__":
    success = install_dependencies()
    sys.exit(0 if success else 1)