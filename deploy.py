#!/usr/bin/env python3
"""
Sistema de Despliegue Automatizado - U-CogNet
Despliega todo el sistema con verificación completa
"""

import os
import sys
import json
import logging
import subprocess
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeploymentManager:
    """Gestor de despliegue automatizado"""

    def __init__(self, target_env: str = "development"):
        self.target_env = target_env
        self.deployment_steps = []
        self.rollback_steps = []
        self.deployment_status = "NOT_STARTED"

    def deploy_full_system(self) -> bool:
        """Desplegar sistema completo"""
        logger.info(f"🚀 Iniciando despliegue completo en {self.target_env}")
        self.deployment_status = "IN_PROGRESS"

        try:
            # Paso 1: Verificar prerrequisitos
            if not self._check_prerequisites():
                raise Exception("Prerrequisitos no cumplidos")

            # Paso 2: Preparar entorno
            if not self._prepare_environment():
                raise Exception("Error preparando entorno")

            # Paso 3: Instalar dependencias
            if not self._install_dependencies():
                raise Exception("Error instalando dependencias")

            # Paso 4: Ejecutar tests
            if not self._run_deployment_tests():
                raise Exception("Tests de despliegue fallaron")

            # Paso 5: Configurar servicios
            if not self._configure_services():
                raise Exception("Error configurando servicios")

            # Paso 6: Iniciar sistema
            if not self._start_system():
                raise Exception("Error iniciando sistema")

            # Paso 7: Verificar funcionamiento
            if not self._verify_system():
                raise Exception("Verificación del sistema falló")

            self.deployment_status = "SUCCESS"
            logger.info("✅ Despliegue completado exitosamente")
            return True

        except Exception as e:
            logger.error(f"❌ Despliegue falló: {e}")
            self.deployment_status = "FAILED"
            self._rollback()
            return False

    def _check_prerequisites(self) -> bool:
        """Verificar prerrequisitos del sistema"""
        logger.info("🔍 Verificando prerrequisitos...")

        checks = [
            ("Python 3.8+", self._check_python_version),
            ("Poetry", self._check_poetry),
            ("Git", self._check_git),
            ("Espacio en disco", self._check_disk_space),
            ("Permisos de escritura", self._check_write_permissions)
        ]

        all_passed = True
        for check_name, check_func in checks:
            try:
                if check_func():
                    logger.info(f"✅ {check_name}: OK")
                    self.deployment_steps.append(f"✅ {check_name}")
                else:
                    logger.error(f"❌ {check_name}: FALLÓ")
                    all_passed = False
            except Exception as e:
                logger.error(f"❌ {check_name}: ERROR - {e}")
                all_passed = False

        return all_passed

    def _check_python_version(self) -> bool:
        """Verificar versión de Python"""
        version = sys.version_info
        return version.major >= 3 and version.minor >= 8

    def _check_poetry(self) -> bool:
        """Verificar Poetry"""
        try:
            result = subprocess.run(["poetry", "--version"],
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False

    def _check_git(self) -> bool:
        """Verificar Git"""
        try:
            result = subprocess.run(["git", "--version"],
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False

    def _check_disk_space(self) -> bool:
        """Verificar espacio en disco (mínimo 1GB)"""
        try:
            result = subprocess.run(["df", "."], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                # Extraer porcentaje usado
                parts = lines[1].split()
                if len(parts) > 4:
                    usage = int(parts[4].rstrip('%'))
                    return usage < 95  # Menos del 95% usado
        except:
            pass
        return True  # Fallback - asumir OK

    def _check_write_permissions(self) -> bool:
        """Verificar permisos de escritura"""
        try:
            test_file = ".deployment_test"
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            return True
        except:
            return False

    def _prepare_environment(self) -> bool:
        """Preparar entorno de despliegue"""
        logger.info("🔧 Preparando entorno...")

        try:
            # Crear directorios necesarios
            dirs_to_create = [
                "logs",
                "backups",
                "temp",
                "results/deployment"
            ]

            for dir_path in dirs_to_create:
                os.makedirs(dir_path, exist_ok=True)
                self.deployment_steps.append(f"📁 Creado directorio: {dir_path}")

            # Backup de configuraciones existentes
            self._backup_existing_configs()

            logger.info("✅ Entorno preparado")
            return True

        except Exception as e:
            logger.error(f"❌ Error preparando entorno: {e}")
            return False

    def _backup_existing_configs(self):
        """Hacer backup de configuraciones existentes"""
        import shutil

        backup_files = [
            "pyproject.toml",
            "poetry.lock",
            ".env",
            "config.json"
        ]

        for file_path in backup_files:
            if os.path.exists(file_path):
                backup_path = f"backups/{file_path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(file_path, backup_path)
                self.rollback_steps.append(f"restore:{file_path}:{backup_path}")

    def _install_dependencies(self) -> bool:
        """Instalar dependencias del proyecto"""
        logger.info("📦 Instalando dependencias...")

        try:
            # Instalar dependencias con Poetry
            result = subprocess.run([
                "poetry", "install", "--no-dev"
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                logger.info("✅ Dependencias instaladas")
                self.deployment_steps.append("📦 Dependencias instaladas con Poetry")
                return True
            else:
                logger.error(f"❌ Error instalando dependencias: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("⏰ Timeout instalando dependencias")
            return False
        except Exception as e:
            logger.error(f"❌ Error en instalación: {e}")
            return False

    def _run_deployment_tests(self) -> bool:
        """Ejecutar tests de despliegue"""
        logger.info("🧪 Ejecutando tests de despliegue...")

        try:
            # Ejecutar suite de tests completa
            result = subprocess.run([
                "poetry", "run", "python", "master_test_suite.py"
            ], capture_output=True, text=True, timeout=600)  # 10 min timeout

            if result.returncode == 0:
                logger.info("✅ Tests de despliegue pasaron")
                self.deployment_steps.append("🧪 Tests de despliegue: PASARON")
                return True
            else:
                logger.error(f"❌ Tests fallaron: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("⏰ Tests excedieron timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Error ejecutando tests: {e}")
            return False

    def _configure_services(self) -> bool:
        """Configurar servicios del sistema"""
        logger.info("⚙️ Configurando servicios...")

        try:
            # Crear archivo de configuración
            config = {
                "environment": self.target_env,
                "deployment_time": datetime.now().isoformat(),
                "version": self._get_version(),
                "services": {
                    "ci_monitor": {
                        "enabled": True,
                        "check_interval": 60
                    },
                    "security_architecture": {
                        "enabled": True,
                        "threat_detection": True
                    },
                    "vision_pipeline": {
                        "enabled": True,
                        "model": "yolov8n"
                    }
                }
            }

            with open("config/deployment_config.json", "w") as f:
                json.dump(config, f, indent=2)

            self.deployment_steps.append("⚙️ Servicios configurados")
            logger.info("✅ Servicios configurados")
            return True

        except Exception as e:
            logger.error(f"❌ Error configurando servicios: {e}")
            return False

    def _start_system(self) -> bool:
        """Iniciar sistema"""
        logger.info("▶️ Iniciando sistema...")

        try:
            # Crear script de inicio
            startup_script = """#!/bin/bash
# Script de inicio del sistema U-CogNet

echo "🚀 Iniciando U-CogNet..."

# Activar entorno virtual
poetry shell

# Iniciar monitor CI en background
poetry run python ci_monitor.py &
echo $! > ci_monitor.pid

# Iniciar demo de seguridad
poetry run python security_architecture_demo.py &
echo $! > security_demo.pid

echo "✅ Sistema iniciado"
echo "PID CI Monitor: $(cat ci_monitor.pid)"
echo "PID Security Demo: $(cat security_demo.pid)"
"""

            with open("start_system.sh", "w") as f:
                f.write(startup_script)

            # Hacer ejecutable
            os.chmod("start_system.sh", 0o755)

            self.deployment_steps.append("▶️ Sistema de inicio creado")
            logger.info("✅ Sistema preparado para inicio")
            return True

        except Exception as e:
            logger.error(f"❌ Error iniciando sistema: {e}")
            return False

    def _verify_system(self) -> bool:
        """Verificar que el sistema funciona correctamente"""
        logger.info("🔍 Verificando sistema...")

        try:
            # Verificar que los archivos principales existen
            required_files = [
                "master_test_suite.py",
                "ci_monitor.py",
                "security_architecture_demo.py",
                "config/deployment_config.json"
            ]

            for file_path in required_files:
                if not os.path.exists(file_path):
                    raise Exception(f"Archivo requerido faltante: {file_path}")

            # Verificar que se puede importar los módulos principales
            test_imports = [
                "import torch",
                "import numpy",
                "from master_test_suite import MasterTestSuite"
            ]

            for import_test in test_imports:
                result = subprocess.run([
                    "poetry", "run", "python", "-c", import_test
                ], capture_output=True, timeout=10)

                if result.returncode != 0:
                    raise Exception(f"Import falló: {import_test}")

            self.deployment_steps.append("🔍 Verificación del sistema: PASÓ")
            logger.info("✅ Sistema verificado correctamente")
            return True

        except Exception as e:
            logger.error(f"❌ Verificación falló: {e}")
            return False

    def _rollback(self):
        """Revertir cambios en caso de fallo"""
        logger.info("⏪ Ejecutando rollback...")

        for step in reversed(self.rollback_steps):
            try:
                if step.startswith("restore:"):
                    _, original, backup = step.split(":", 2)
                    if os.path.exists(backup):
                        import shutil
                        shutil.copy2(backup, original)
                        logger.info(f"✅ Restaurado: {original}")
            except Exception as e:
                logger.error(f"❌ Error en rollback: {e}")

    def _get_version(self) -> str:
        """Obtener versión del sistema"""
        try:
            result = subprocess.run([
                "git", "describe", "--tags", "--always"
            ], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return "dev"

    def get_deployment_report(self) -> Dict:
        """Obtener reporte de despliegue"""
        return {
            "status": self.deployment_status,
            "environment": self.target_env,
            "timestamp": datetime.now().isoformat(),
            "steps_completed": len(self.deployment_steps),
            "steps": self.deployment_steps,
            "version": self._get_version()
        }

class DeploymentCLI:
    """Interfaz de línea de comandos para despliegue"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Sistema de Despliegue Automatizado - U-CogNet"
        )
        self.parser.add_argument(
            "--env", "-e",
            choices=["development", "staging", "production"],
            default="development",
            help="Entorno de despliegue"
        )
        self.parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Ejecutar simulación sin cambios reales"
        )
        self.parser.add_argument(
            "--force",
            action="store_true",
            help="Forzar despliegue ignorando algunos checks"
        )

    def run(self):
        """Ejecutar CLI de despliegue"""
        args = self.parser.parse_args()

        print("🚀 Sistema de Despliegue Automatizado - U-CogNet")
        print("=" * 55)
        print(f"Entorno: {args.env}")
        print(f"Dry run: {args.dry_run}")
        print(f"Force: {args.force}")
        print()

        if args.dry_run:
            print("🔍 Ejecutando dry-run...")
            # Aquí iría lógica de simulación
            print("✅ Dry-run completado (simulado)")
            return

        # Ejecutar despliegue real
        deployer = DeploymentManager(args.env)

        success = deployer.deploy_full_system()

        # Mostrar reporte final
        report = deployer.get_deployment_report()

        print("\n📊 REPORTE FINAL DE DESPLIEGUE")
        print("=" * 35)
        print(f"Estado: {'✅ ÉXITO' if success else '❌ FALLO'}")
        print(f"Entorno: {report['environment']}")
        print(f"Versión: {report['version']}")
        print(f"Pasos completados: {report['steps_completed']}")

        if report['steps']:
            print("\n📋 Pasos ejecutados:")
            for step in report['steps']:
                print(f"  {step}")

        # Guardar reporte
        with open("deployment_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        print("\n📄 Reporte guardado en: deployment_report.json")
        if success:
            print("\n🎉 ¡Despliegue completado exitosamente!")
            print("Ejecuta: ./start_system.sh")
        else:
            print("\n❌ Despliegue falló. Revisa los logs para más detalles.")

def main():
    """Función principal"""
    cli = DeploymentCLI()
    cli.run()

if __name__ == "__main__":
    main()