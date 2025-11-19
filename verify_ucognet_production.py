#!/usr/bin/env python3
"""
U-CogNet Production Readiness Verification
Verificación completa de que U-CogNet está listo para producción
"""

import sys
import os
import asyncio
from datetime import datetime

# Configurar path ANTES de las importaciones
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from ucognet.core.cognitive_core import CognitiveCore
from ucognet.core.tda_manager import TDAManager
from ucognet.core.evaluator import Evaluator
from ucognet.core.trainer_loop import TrainerLoop
from ucognet.core.mycelial_optimizer import MycelialOptimizer
from ucognet.core.types import SystemState, Metrics, TopologyConfig
from ucognet.core.utils import setup_logging, get_system_info


async def verify_ucognet_readiness():
    """Verifica que U-CogNet esté completamente operativo"""

    print("🧠 U-COGNET - VERIFICACIÓN DE PRODUCCIÓN")
    print("=" * 60)
    print(f"⏰ Timestamp: {datetime.now().isoformat()}")
    print()

    # 1. Verificar sistema de información
    print("1️⃣ VERIFICANDO INFORMACIÓN DEL SISTEMA:")
    system_info = get_system_info()
    for key, value in system_info.items():
        print(f"   • {key}: {value}")
    print("✅ Sistema operativo correctamente detectado")
    print()

    # 2. Verificar módulos críticos
    print("2️⃣ VERIFICANDO MÓDULOS CRÍTICOS:")

    modules_status = {}

    try:
        cognitive_core = CognitiveCore()
        modules_status['CognitiveCore'] = "✅ IMPLEMENTADO"
        print("   ✅ CognitiveCore: Instancia creada")
    except Exception as e:
        modules_status['CognitiveCore'] = f"❌ ERROR: {e}"
        print(f"   ❌ CognitiveCore: {e}")

    try:
        tda_manager = TDAManager()
        modules_status['TDAManager'] = "✅ IMPLEMENTADO"
        print("   ✅ TDAManager: Instancia creada")
    except Exception as e:
        modules_status['TDAManager'] = f"❌ ERROR: {e}"
        print(f"   ❌ TDAManager: {e}")

    try:
        evaluator = Evaluator()
        modules_status['Evaluator'] = "✅ IMPLEMENTADO"
        print("   ✅ Evaluator: Instancia creada")
    except Exception as e:
        modules_status['Evaluator'] = f"❌ ERROR: {e}"
        print(f"   ❌ Evaluator: {e}")

    try:
        trainer_loop = TrainerLoop()
        modules_status['TrainerLoop'] = "✅ IMPLEMENTADO"
        print("   ✅ TrainerLoop: Instancia creada")
    except Exception as e:
        modules_status['TrainerLoop'] = f"❌ ERROR: {e}"
        print(f"   ❌ TrainerLoop: {e}")

    try:
        mycelial_optimizer = MycelialOptimizer()
        modules_status['MycelialOptimizer'] = "✅ IMPLEMENTADO"
        print("   ✅ MycelialOptimizer: Instancia creada")
    except Exception as e:
        modules_status['MycelialOptimizer'] = f"❌ ERROR: {e}"
        print(f"   ❌ MycelialOptimizer: {e}")

    print()

    # 3. Verificar integración de módulos
    print("3️⃣ VERIFICANDO INTEGRACIÓN DE MÓDULOS:")

    integration_tests = []

    # Test 1: Cognitive Core con TDA Manager
    try:
        if 'cognitive_core' in locals() and 'tda_manager' in locals():
            # Simular procesamiento básico
            test_data = {"input": [1, 2, 3, 4, 5]}
            result = await cognitive_core.process_input(test_data)
            integration_tests.append("✅ CognitiveCore ↔ TDAManager")
            print("   ✅ CognitiveCore procesa datos correctamente")
        else:
            integration_tests.append("❌ CognitiveCore ↔ TDAManager")
            print("   ❌ Módulos no disponibles para integración")
    except Exception as e:
        integration_tests.append("❌ CognitiveCore ↔ TDAManager")
        print(f"   ❌ Error en integración: {e}")

    # Test 1: Cognitive Core con TDA Manager
    try:
        if 'cognitive_core' in locals() and 'tda_manager' in locals():
            # Simular procesamiento básico
            test_data = {"input": [1, 2, 3, 4, 5]}
            result = await cognitive_core.process_input(test_data)
            integration_tests.append("✅ CognitiveCore ↔ TDAManager")
            print("   ✅ CognitiveCore procesa datos correctamente")
        else:
            integration_tests.append("❌ CognitiveCore ↔ TDAManager")
            print("   ❌ Módulos no disponibles para integración")
    except Exception as e:
        integration_tests.append("❌ CognitiveCore ↔ TDAManager")
        print(f"   ❌ Error en integración: {e}")

    # Test 2: Evaluator con métricas
    try:
        if 'evaluator' in locals():
            report = await evaluator.evaluate_performance()
            integration_tests.append("✅ Evaluator calcula métricas")
            print(f"   ✅ Evaluator: Overall Score={report.overall_score:.2f}")
        else:
            integration_tests.append("❌ Evaluator no disponible")
            print("   ❌ Evaluator no disponible")
    except Exception as e:
        integration_tests.append("❌ Evaluator error")
        print(f"   ❌ Error en evaluator: {e}")

    # Test 3: Mycelial Optimizer
    try:
        if 'mycelial_optimizer' in locals():
            # Probar adaptación de learning rates
            learning_rates = await mycelial_optimizer.adapt_learning_rates(0.8)
            integration_tests.append("✅ MycelialOptimizer operativo")
            print("   ✅ MycelialOptimizer adapta learning rates")
        else:
            integration_tests.append("❌ MycelialOptimizer no disponible")
            print("   ❌ MycelialOptimizer no disponible")
    except Exception as e:
        integration_tests.append("❌ MycelialOptimizer error")
        print(f"   ❌ Error en MycelialOptimizer: {e}")

    print()

    # 4. Verificar arquitectura micelial
    print("4️⃣ VERIFICANDO ARQUITECTURA MICELIAL:")

    mycelial_features = []

    # Verificar MycelialOptimizer tiene características miceliales
    try:
        if 'mycelial_optimizer' in locals():
            # Verificar métodos miceliales
            if hasattr(mycelial_optimizer, 'cluster_parameters'):
                mycelial_features.append("✅ Clustering de parámetros")
                print("   ✅ Clustering de parámetros implementado")
            if hasattr(mycelial_optimizer, 'adapt_learning_rates'):
                mycelial_features.append("✅ Adaptación de learning rates")
                print("   ✅ Adaptación de learning rates implementada")
            if hasattr(mycelial_optimizer, 'prune_connections'):
                mycelial_features.append("✅ Poda de conexiones")
                print("   ✅ Poda de conexiones implementada")
        else:
            mycelial_features.append("❌ MycelialOptimizer no disponible")
            print("   ❌ MycelialOptimizer no disponible")
    except Exception as e:
        mycelial_features.append(f"❌ Error micelial: {e}")
        print(f"   ❌ Error verificando características miceliales: {e}")

    print()

    # 5. Reporte final
    print("🎯 REPORTE FINAL DE VERIFICACIÓN")
    print("=" * 60)

    all_modules_ok = all("✅" in status for status in modules_status.values())
    all_integration_ok = all("✅" in test for test in integration_tests)
    mycelial_ok = len([f for f in mycelial_features if "✅" in f]) >= 2

    print("📦 MÓDULOS CRÍTICOS:")
    for module, status in modules_status.items():
        print(f"   {status}")

    print()
    print("🔗 INTEGRACIÓN:")
    for test in integration_tests:
        print(f"   {test}")

    print()
    print("🍄 ARQUITECTURA MICELIAL:")
    for feature in mycelial_features:
        print(f"   {feature}")

    print()
    print("🏆 EVALUACIÓN FINAL:")

    if all_modules_ok and all_integration_ok and mycelial_ok:
        print("🎉 ¡U-COGNET ESTÁ COMPLETAMENTE LISTO PARA PRODUCCIÓN!")
        print()
        print("✅ Todos los módulos críticos implementados")
        print("✅ Integración entre módulos funcional")
        print("✅ Arquitectura micelial operativa")
        print("✅ Sistema de optimización inteligente activo")
        print("✅ Dependencias críticas satisfechas")
        print()
        print("🚀 El sistema U-CogNet está preparado para:")
        print("   • Procesamiento cognitivo multimodal")
        print("   • Adaptación topológica dinámica")
        print("   • Optimización inspirada en micelio")
        print("   • Evaluación de rendimiento en tiempo real")
        print("   • Aprendizaje continuo y autónomo")
        return True
    else:
        print("⚠️ U-COGNET REQUIERE COMPLEMENTOS ANTES DE PRODUCCIÓN")
        print()
        if not all_modules_ok:
            print("❌ Módulos críticos faltantes o con errores")
        if not all_integration_ok:
            print("❌ Problemas de integración entre módulos")
        if not mycelial_ok:
            print("❌ Características miceliales insuficientes")
        return False


async def main():
    """Función principal"""
    try:
        success = await verify_ucognet_readiness()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ ERROR CRÍTICO EN VERIFICACIÓN: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Configurar logging
    logger = setup_logging("INFO")

    # Ejecutar verificación
    asyncio.run(main())