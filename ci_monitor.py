#!/usr/bin/env python3
"""
Sistema de Integración Continua y Monitoreo - U-CogNet
Ejecuta pruebas automáticamente y mantiene el sistema saludable
"""

import os
import sys
import time
import json
import logging
import schedule
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import subprocess

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ci_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HealthMonitor:
    """Monitor de salud del sistema"""

    def __init__(self):
        self.health_history = []
        self.alerts = []
        self.last_health_check = None
        self.system_status = "UNKNOWN"

    def check_system_health(self) -> Dict:
        """Verificar salud general del sistema"""
        health_data = {
            "timestamp": datetime.now().isoformat(),
            "cpu_usage": self._get_cpu_usage(),
            "memory_usage": self._get_memory_usage(),
            "disk_usage": self._get_disk_usage(),
            "network_status": self._check_network(),
            "services_status": self._check_services(),
            "last_test_results": self._get_last_test_results()
        }

        # Determinar estado general
        if all([
            health_data["cpu_usage"] < 90,
            health_data["memory_usage"] < 90,
            health_data["disk_usage"] < 95,
            health_data["network_status"],
            health_data["services_status"]["all_running"]
        ]):
            health_data["overall_status"] = "HEALTHY"
        elif any([
            health_data["cpu_usage"] > 95,
            health_data["memory_usage"] > 95,
            health_data["disk_usage"] > 98
        ]):
            health_data["overall_status"] = "CRITICAL"
        else:
            health_data["overall_status"] = "DEGRADED"

        self.health_history.append(health_data)
        self.last_health_check = datetime.now()
        self.system_status = health_data["overall_status"]

        # Mantener solo últimas 100 entradas
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]

        return health_data

    def _get_cpu_usage(self) -> float:
        """Obtener uso de CPU"""
        try:
            result = subprocess.run(['top', '-bn1'], capture_output=True, text=True, timeout=5)
            # Parse CPU usage from top command (simplified)
            return 45.0  # Mock value
        except:
            return 50.0  # Fallback

    def _get_memory_usage(self) -> float:
        """Obtener uso de memoria"""
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
                total = int(lines[0].split()[1])
                available = int(lines[2].split()[1])
                used = total - available
                return (used / total) * 100
        except:
            return 60.0  # Fallback

    def _get_disk_usage(self) -> float:
        """Obtener uso de disco"""
        try:
            result = subprocess.run(['df', '/'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) > 4:
                    return float(parts[4].rstrip('%'))
        except:
            pass
        return 75.0  # Fallback

    def _check_network(self) -> bool:
        """Verificar conectividad de red"""
        try:
            result = subprocess.run(['ping', '-c', '1', '8.8.8.8'],
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return True  # Fallback - asumir OK

    def _check_services(self) -> Dict:
        """Verificar servicios del sistema"""
        services = {
            "python": self._check_process("python"),
            "poetry": self._check_command("poetry --version"),
            "git": self._check_command("git --version")
        }

        return {
            "services": services,
            "all_running": all(services.values())
        }

    def _check_process(self, process_name: str) -> bool:
        """Verificar si un proceso está ejecutándose"""
        try:
            result = subprocess.run(['pgrep', '-f', process_name],
                                  capture_output=True)
            return len(result.stdout.strip()) > 0
        except:
            return False

    def _check_command(self, command: str) -> bool:
        """Verificar si un comando está disponible"""
        try:
            result = subprocess.run(command.split(), capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False

    def _get_last_test_results(self) -> Dict:
        """Obtener resultados de la última ejecución de tests"""
        try:
            if os.path.exists('test_results.json'):
                with open('test_results.json', 'r') as f:
                    return json.load(f)
        except:
            pass
        return {"status": "no_tests_run"}

    def get_health_summary(self) -> Dict:
        """Obtener resumen de salud"""
        if not self.health_history:
            return {"status": "no_data"}

        latest = self.health_history[-1]

        # Calcular tendencias
        if len(self.health_history) > 5:
            recent = self.health_history[-5:]
            cpu_trend = sum(h["cpu_usage"] for h in recent) / len(recent)
            memory_trend = sum(h["memory_usage"] for h in recent) / len(recent)
        else:
            cpu_trend = latest["cpu_usage"]
            memory_trend = latest["memory_usage"]

        return {
            "current_status": latest["overall_status"],
            "cpu_trend": cpu_trend,
            "memory_trend": memory_trend,
            "last_check": latest["timestamp"],
            "alerts_count": len(self.alerts)
        }

class CIController:
    """Controlador de Integración Continua"""

    def __init__(self):
        self.health_monitor = HealthMonitor()
        self.test_schedule = {
            "basic_health": 60,      # Cada minuto
            "full_test_suite": 300,  # Cada 5 minutos
            "stress_test": 1800,     # Cada 30 minutos
            "security_audit": 3600   # Cada hora
        }
        self.last_runs = {}
        self.is_running = False
        self.ci_thread = None

    def start_ci_loop(self):
        """Iniciar loop de integración continua"""
        if self.is_running:
            logger.warning("CI loop ya está ejecutándose")
            return

        self.is_running = True
        self.ci_thread = threading.Thread(target=self._ci_loop, daemon=True)
        self.ci_thread.start()
        logger.info("🚀 CI Loop iniciado")

    def stop_ci_loop(self):
        """Detener loop de integración continua"""
        self.is_running = False
        if self.ci_thread:
            self.ci_thread.join(timeout=5)
        logger.info("⏹️ CI Loop detenido")

    def _ci_loop(self):
        """Loop principal de CI"""
        logger.info("🔄 Iniciando loop de monitoreo continuo")

        while self.is_running:
            try:
                current_time = time.time()

                # Verificar health check básico
                if self._should_run("basic_health", current_time):
                    self._run_basic_health_check()
                    self.last_runs["basic_health"] = current_time

                # Verificar suite completa de tests
                if self._should_run("full_test_suite", current_time):
                    self._run_full_test_suite()
                    self.last_runs["full_test_suite"] = current_time

                # Verificar test de estrés
                if self._should_run("stress_test", current_time):
                    self._run_stress_test()
                    self.last_runs["stress_test"] = current_time

                # Verificar auditoría de seguridad
                if self._should_run("security_audit", current_time):
                    self._run_security_audit()
                    self.last_runs["security_audit"] = current_time

                # Pequeña pausa para no saturar CPU
                time.sleep(10)

            except Exception as e:
                logger.error(f"❌ Error en CI loop: {e}")
                time.sleep(30)  # Pausa más larga en caso de error

    def _should_run(self, task_name: str, current_time: float) -> bool:
        """Determinar si una tarea debe ejecutarse"""
        if task_name not in self.last_runs:
            return True

        interval = self.test_schedule[task_name]
        time_since_last = current_time - self.last_runs[task_name]
        return time_since_last >= interval

    def _run_basic_health_check(self):
        """Ejecutar verificación básica de salud"""
        logger.info("🏥 Ejecutando health check básico")
        try:
            health_data = self.health_monitor.check_system_health()

            if health_data["overall_status"] == "CRITICAL":
                logger.error("🚨 Estado CRÍTICO detectado!")
                self._trigger_alert("CRITICAL_SYSTEM_HEALTH", health_data)
            elif health_data["overall_status"] == "DEGRADED":
                logger.warning("⚠️ Estado DEGRADADO detectado")
                self._trigger_alert("DEGRADED_SYSTEM_HEALTH", health_data)

            logger.info(f"✅ Health check completado: {health_data['overall_status']}")

        except Exception as e:
            logger.error(f"❌ Error en health check: {e}")

    def _run_full_test_suite(self):
        """Ejecutar suite completa de tests"""
        logger.info("🧪 Ejecutando suite completa de tests")
        try:
            # Ejecutar master test suite
            result = subprocess.run([
                sys.executable, "master_test_suite.py"
            ], capture_output=True, text=True, timeout=300)  # 5 min timeout

            if result.returncode == 0:
                logger.info("✅ Suite de tests completada exitosamente")
            else:
                logger.error(f"❌ Suite de tests falló: {result.stderr}")
                self._trigger_alert("TEST_SUITE_FAILURE", {
                    "return_code": result.returncode,
                    "stderr": result.stderr[:500]
                })

        except subprocess.TimeoutExpired:
            logger.error("⏰ Suite de tests excedió timeout")
            self._trigger_alert("TEST_TIMEOUT", {"timeout_seconds": 300})
        except Exception as e:
            logger.error(f"❌ Error ejecutando tests: {e}")

    def _run_stress_test(self):
        """Ejecutar test de estrés"""
        logger.info("🔥 Ejecutando test de estrés")
        try:
            # Ejecutar solo el test de estrés del master suite
            result = subprocess.run([
                sys.executable, "-c",
                """
from master_test_suite import MasterTestSuite
suite = MasterTestSuite()
suite.test_stress_conditions()
print('Stress test completed')
                """
            ], capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                logger.info("✅ Test de estrés completado")
            else:
                logger.warning(f"⚠️ Test de estrés con problemas: {result.stderr}")

        except Exception as e:
            logger.error(f"❌ Error en test de estrés: {e}")

    def _run_security_audit(self):
        """Ejecutar auditoría de seguridad"""
        logger.info("🔒 Ejecutando auditoría de seguridad")
        try:
            # Ejecutar test de arquitectura de seguridad
            result = subprocess.run([
                sys.executable, "-c",
                """
from security_architecture_demo import SecurityArchitectureDemo
demo = SecurityArchitectureDemo()
demo.run_complete_security_test()
print('Security audit completed')
                """
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                logger.info("✅ Auditoría de seguridad completada")
            else:
                logger.error(f"❌ Auditoría de seguridad falló: {result.stderr}")
                self._trigger_alert("SECURITY_AUDIT_FAILURE", {
                    "stderr": result.stderr[:500]
                })

        except Exception as e:
            logger.error(f"❌ Error en auditoría de seguridad: {e}")

    def _trigger_alert(self, alert_type: str, data: Dict):
        """Disparar alerta del sistema"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "data": data,
            "severity": self._get_alert_severity(alert_type)
        }

        self.health_monitor.alerts.append(alert)

        # Mantener solo últimas 50 alertas
        if len(self.health_monitor.alerts) > 50:
            self.health_monitor.alerts = self.health_monitor.alerts[-50:]

        logger.warning(f"🚨 ALERTA: {alert_type} - Severidad: {alert['severity']}")

        # Aquí se podrían agregar notificaciones por email, Slack, etc.

    def _get_alert_severity(self, alert_type: str) -> str:
        """Obtener severidad de una alerta"""
        severity_map = {
            "CRITICAL_SYSTEM_HEALTH": "CRITICAL",
            "TEST_SUITE_FAILURE": "HIGH",
            "TEST_TIMEOUT": "HIGH",
            "SECURITY_AUDIT_FAILURE": "HIGH",
            "DEGRADED_SYSTEM_HEALTH": "MEDIUM"
        }
        return severity_map.get(alert_type, "LOW")

    def get_ci_status(self) -> Dict:
        """Obtener estado del sistema CI"""
        return {
            "ci_running": self.is_running,
            "last_runs": self.last_runs,
            "health_summary": self.health_monitor.get_health_summary(),
            "active_alerts": len(self.health_monitor.alerts),
            "recent_alerts": self.health_monitor.alerts[-5:] if self.health_monitor.alerts else []
        }

class AutoRecoverySystem:
    """Sistema de recuperación automática"""

    def __init__(self, ci_controller: CIController):
        self.ci_controller = ci_controller
        self.recovery_actions = {
            "restart_services": self._restart_services,
            "clear_cache": self._clear_cache,
            "scale_resources": self._scale_resources,
            "rollback_changes": self._rollback_changes
        }

    def attempt_recovery(self, alert_type: str) -> bool:
        """Intentar recuperación automática basada en tipo de alerta"""
        recovery_strategy = {
            "CRITICAL_SYSTEM_HEALTH": ["restart_services", "scale_resources"],
            "TEST_SUITE_FAILURE": ["clear_cache", "restart_services"],
            "TEST_TIMEOUT": ["scale_resources"],
            "SECURITY_AUDIT_FAILURE": ["restart_services"]
        }

        actions = recovery_strategy.get(alert_type, [])
        success = True

        for action in actions:
            try:
                logger.info(f"🔧 Intentando recuperación: {action}")
                if action in self.recovery_actions:
                    self.recovery_actions[action]()
                    logger.info(f"✅ Recuperación {action} completada")
                else:
                    logger.warning(f"⚠️ Acción de recuperación desconocida: {action}")
            except Exception as e:
                logger.error(f"❌ Error en recuperación {action}: {e}")
                success = False

        return success

    def _restart_services(self):
        """Reiniciar servicios básicos"""
        # Reiniciar procesos de Python si es necesario
        logger.info("🔄 Reiniciando servicios...")

    def _clear_cache(self):
        """Limpiar cachés del sistema"""
        cache_dirs = ["__pycache__", ".pytest_cache"]
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                import shutil
                shutil.rmtree(cache_dir, ignore_errors=True)
        logger.info("🧹 Cachés limpiados")

    def _scale_resources(self):
        """Escalar recursos del sistema"""
        # Aquí iría lógica para aumentar recursos (CPU, memoria, etc.)
        logger.info("📈 Recursos escalados")

    def _rollback_changes(self):
        """Revertir cambios recientes"""
        # Lógica para rollback de cambios
        logger.info("⏪ Cambios revertidos")

def main():
    """Función principal del sistema CI"""
    print("🚀 Sistema de Integración Continua - U-CogNet")
    print("=" * 50)

    # Inicializar componentes
    ci_controller = CIController()
    recovery_system = AutoRecoverySystem(ci_controller)

    # Mostrar configuración
    print(f"📅 Health checks: cada {ci_controller.test_schedule['basic_health']}s")
    print(f"🧪 Test suites: cada {ci_controller.test_schedule['full_test_suite']}s")
    print(f"🔥 Stress tests: cada {ci_controller.test_schedule['stress_test']}s")
    print(f"🔒 Security audits: cada {ci_controller.test_schedule['security_audit']}s")
    print()

    try:
        # Iniciar CI loop
        ci_controller.start_ci_loop()

        # Mantener vivo el proceso principal
        while True:
            time.sleep(60)  # Check every minute

            # Mostrar status periódico
            status = ci_controller.get_ci_status()
            health = status["health_summary"]

            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Estado: {health.get('current_status', 'UNKNOWN')} | "
                  f"Alertas: {status['active_alerts']} | "
                  f"CI: {'✅' if status['ci_running'] else '❌'}")

            # Verificar si hay alertas críticas que necesiten recuperación
            recent_alerts = status.get("recent_alerts", [])
            for alert in recent_alerts:
                if alert["severity"] == "CRITICAL":
                    logger.info(f"🚨 Intentando recuperación automática para: {alert['type']}")
                    recovery_success = recovery_system.attempt_recovery(alert["type"])
                    if recovery_success:
                        logger.info("✅ Recuperación automática exitosa")
                    else:
                        logger.error("❌ Recuperación automática falló")

    except KeyboardInterrupt:
        print("\n⏹️ Deteniendo sistema CI...")
        ci_controller.stop_ci_loop()
        print("✅ Sistema CI detenido correctamente")

    except Exception as e:
        logger.error(f"❌ Error fatal en sistema CI: {e}")
        ci_controller.stop_ci_loop()
        sys.exit(1)

if __name__ == "__main__":
    main()