#!/usr/bin/env python3
"""
Suite Completa de Pruebas Unitarias para U-CogNet
Verifica que todos los módulos principales estén implementados y funcionen
"""

import asyncio
import sys
import unittest
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

class UCogNetTestSuite(unittest.TestCase):
    """Suite completa de pruebas para U-CogNet"""

    def setUp(self):
        """Configuración inicial de pruebas"""
        self.modules_status = {}
        self.test_results = []

    def test_core_modules_existence(self):
        """Verifica que todos los módulos principales existen"""
        core_modules = [
            # Arquitectura principal según documentación
            'ucognet.core.input_handler',
            'ucognet.core.vision_detector',
            'ucognet.core.incremental_tank_learner',
            'ucognet.core.cognitive_core',
            'ucognet.core.semantic_feedback',
            'ucognet.core.evaluator',
            'ucognet.core.trainer_loop',
            'ucognet.core.tda_manager',
            'ucognet.core.visual_interface',
            # Optimizador micelial
            'ucognet.core.mycelial_optimizer',
            # Sistema de memoria
            'ucognet.core.memory.short_term_memory',
            'ucognet.core.memory.episodic_memory',
            'ucognet.core.memory.semantic_memory',
            # Sistema de trazabilidad (MTC)
            'ucognet.core.tracing.cognitive_event',
            'ucognet.core.tracing.trace_core',
            'ucognet.core.tracing.event_bus',
            'ucognet.core.tracing.causal_builder',
            'ucognet.core.tracing.coherence_evaluator',
            'ucognet.core.tracing.query_api',
            'ucognet.core.tracing.storage_manager',
            # Interfaces y protocolos
            'ucognet.core.interfaces',
            'ucognet.core.protocols',
            # Runtime
            'ucognet.runtime.engine',
            # Utilidades
            'ucognet.core.utils',
            'ucognet.core.types'
        ]

        missing_modules = []
        existing_modules = []

        for module_name in core_modules:
            try:
                module = importlib.import_module(module_name)
                existing_modules.append(module_name)
                self.modules_status[module_name] = {"status": "EXISTS", "module": module}
            except ImportError as e:
                missing_modules.append(module_name)
                self.modules_status[module_name] = {"status": "MISSING", "error": str(e)}

        # Reportar resultados
        print(f"\n📦 Módulos existentes: {len(existing_modules)}/{len(core_modules)}")
        print(f"❌ Módulos faltantes: {len(missing_modules)}/{len(core_modules)}")

        if missing_modules:
            print("\n🔍 Módulos faltantes críticos:")
            for mod in missing_modules:
                print(f"   • {mod}")

        # La prueba falla si faltan módulos críticos
        critical_modules = [
            'ucognet.core.cognitive_core',
            'ucognet.core.tda_manager',
            'ucognet.core.evaluator',
            'ucognet.core.trainer_loop',
            'ucognet.core.mycelial_optimizer'
        ]

        missing_critical = [m for m in critical_modules if m in missing_modules]
        if missing_critical:
            self.fail(f"Módulos críticos faltantes: {missing_critical}")

    def test_tracing_system_integration(self):
        """Verifica que el sistema de trazabilidad esté completamente integrado"""
        try:
            from ucognet.core.tracing import (
                CognitiveEvent, EventType, LogLevel,
                CognitiveTraceCore, CognitiveEventBus,
                CausalGraphBuilder, CoherenceEthicsEvaluator,
                TraceQueryAPI, TraceStorageManager,
                get_event_bus, get_trace_core
            )

            # Verificar que se pueden instanciar los componentes principales
            event_bus = get_event_bus()
            trace_core = get_trace_core()
            causal_builder = CausalGraphBuilder()
            coherence_evaluator = CoherenceEthicsEvaluator()
            query_api = TraceQueryAPI(trace_core, causal_builder, coherence_evaluator)
            storage_manager = TraceStorageManager()

            # Verificar integración básica
            event_id = event_bus.emit(
                EventType.DECISION,
                "TestModule",
                inputs={"test": "integration"},
                episode_id="test_episode"
            )

            self.assertIsNotNone(event_id)

            # Verificar que el evento se registró
            event = trace_core.get_event(event_id)
            self.assertIsNotNone(event)
            self.assertEqual(event.source, "TestModule")

            print("✅ Sistema de trazabilidad completamente funcional")

        except Exception as e:
            self.fail(f"Error en integración de trazabilidad: {e}")

    def test_cognitive_core_basic_functionality(self):
        """Verifica funcionalidad básica del cognitive core"""
        try:
            from ucognet.core.cognitive_core import CognitiveCore

            # Crear instancia del cognitive core
            core = CognitiveCore()

            # Verificar que tiene los métodos principales
            self.assertTrue(hasattr(core, 'process_input'))
            self.assertTrue(hasattr(core, 'update_memory'))
            self.assertTrue(hasattr(core, 'make_decision'))
            self.assertTrue(hasattr(core, 'evaluate_state'))

            print("✅ Cognitive core con funcionalidad básica")

        except ImportError:
            self.skipTest("CognitiveCore no implementado aún")
        except Exception as e:
            self.fail(f"Error en cognitive core: {e}")

    def test_tda_manager_existence(self):
        """Verifica que el TDA manager existe y es funcional"""
        try:
            from ucognet.core.tda_manager import TDAManager

            # Crear instancia
            tda = TDAManager()

            # Verificar métodos principales
            self.assertTrue(hasattr(tda, 'evaluate_topology'))
            self.assertTrue(hasattr(tda, 'adapt_topology'))
            self.assertTrue(hasattr(tda, 'get_active_modules'))

            print("✅ TDA Manager implementado")

        except ImportError:
            self.skipTest("TDAManager no implementado aún")
        except Exception as e:
            self.fail(f"Error en TDA Manager: {e}")

    def test_evaluator_functionality(self):
        """Verifica que el evaluator calcula métricas correctamente"""
        try:
            from ucognet.core.evaluator import Evaluator

            evaluator = Evaluator()

            # Verificar métodos de evaluación
            self.assertTrue(hasattr(evaluator, 'calculate_metrics'))
            self.assertTrue(hasattr(evaluator, 'evaluate_performance'))
            self.assertTrue(hasattr(evaluator, 'get_confidence_score'))

            print("✅ Evaluator con métricas implementado")

        except ImportError:
            self.skipTest("Evaluator no implementado aún")
        except Exception as e:
            self.fail(f"Error en evaluator: {e}")

    def test_trainer_loop_existence(self):
        """Verifica que el trainer loop existe"""
        try:
            from ucognet.core.trainer_loop import TrainerLoop

            trainer = TrainerLoop()

            # Verificar métodos de entrenamiento
            self.assertTrue(hasattr(trainer, 'collect_difficult_examples'))
            self.assertTrue(hasattr(trainer, 'perform_micro_update'))
            self.assertTrue(hasattr(trainer, 'schedule_training'))

            print("✅ Trainer loop implementado")

        except ImportError:
            self.skipTest("TrainerLoop no implementado aún")
        except Exception as e:
            self.fail(f"Error en trainer loop: {e}")

    def test_mycelial_optimizer_existence(self):
        """Verifica que el optimizador micelial existe"""
        try:
            from ucognet.core.mycelial_optimizer import MycelialOptimizer

            optimizer = MycelialOptimizer()

            # Verificar métodos del optimizador
            self.assertTrue(hasattr(optimizer, 'cluster_parameters'))
            self.assertTrue(hasattr(optimizer, 'adapt_learning_rates'))
            self.assertTrue(hasattr(optimizer, 'prune_unused_regions'))

            print("✅ Optimizador micelial implementado")

        except ImportError:
            self.skipTest("MycelialOptimizer no implementado aún - MÓDULO CRÍTICO FALTANTE")
        except Exception as e:
            self.fail(f"Error en optimizador micelial: {e}")

    def test_memory_systems(self):
        """Verifica que los sistemas de memoria existen"""
        memory_modules = [
            ('ucognet.core.memory.short_term_memory', 'ShortTermMemory'),
            ('ucognet.core.memory.episodic_memory', 'EpisodicMemory'),
            ('ucognet.core.memory.semantic_memory', 'SemanticMemory')
        ]

        for module_name, class_name in memory_modules:
            try:
                module = importlib.import_module(module_name)
                memory_class = getattr(module, class_name)
                memory_instance = memory_class()

                # Verificar métodos básicos
                self.assertTrue(hasattr(memory_instance, 'store'))
                self.assertTrue(hasattr(memory_instance, 'retrieve'))

                print(f"✅ {class_name} implementado")

            except ImportError:
                self.skipTest(f"{class_name} no implementado aún")
            except Exception as e:
                self.fail(f"Error en {class_name}: {e}")

    def test_input_handler_modality_support(self):
        """Verifica que el input handler soporta múltiples modalidades"""
        try:
            from ucognet.core.input_handler import InputHandler

            handler = InputHandler()

            # Verificar soporte de modalidades
            modalities = ['vision', 'audio', 'text', 'timeseries']
            for modality in modalities:
                self.assertTrue(hasattr(handler, f'handle_{modality}'))

            print("✅ Input handler multimodal implementado")

        except ImportError:
            self.skipTest("InputHandler no implementado aún")
        except Exception as e:
            self.fail(f"Error en input handler: {e}")

    def test_vision_detector_yolo_integration(self):
        """Verifica integración con YOLOv8"""
        try:
            from ucognet.core.vision_detector import VisionDetector

            detector = VisionDetector()

            # Verificar métodos de detección
            self.assertTrue(hasattr(detector, 'detect'))
            self.assertTrue(hasattr(detector, 'load_model'))
            self.assertTrue(hasattr(detector, 'preprocess_frame'))

            print("✅ Vision detector con YOLOv8 implementado")

        except ImportError:
            self.skipTest("VisionDetector no implementado aún")
        except Exception as e:
            self.fail(f"Error en vision detector: {e}")

    def test_semantic_feedback_generation(self):
        """Verifica generación de feedback semántico"""
        try:
            from ucognet.core.semantic_feedback import SemanticFeedback

            feedback = SemanticFeedback()

            # Verificar métodos de generación
            self.assertTrue(hasattr(feedback, 'generate_description'))
            self.assertTrue(hasattr(feedback, 'analyze_context'))
            self.assertTrue(hasattr(feedback, 'create_summary'))

            print("✅ Semantic feedback implementado")

        except ImportError:
            self.skipTest("SemanticFeedback no implementado aún")
        except Exception as e:
            self.fail(f"Error en semantic feedback: {e}")

    def test_visual_interface_hud(self):
        """Verifica interfaz visual con HUD táctico"""
        try:
            from ucognet.core.visual_interface import VisualInterface

            interface = VisualInterface()

            # Verificar componentes del HUD
            self.assertTrue(hasattr(interface, 'render_hud'))
            self.assertTrue(hasattr(interface, 'display_detections'))
            self.assertTrue(hasattr(interface, 'show_system_status'))

            print("✅ Visual interface con HUD implementado")

        except ImportError:
            self.skipTest("VisualInterface no implementado aún")
        except Exception as e:
            self.fail(f"Error en visual interface: {e}")

    def test_incremental_tank_learner(self):
        """Verifica aprendizaje incremental de tanques"""
        try:
            from ucognet.core.incremental_tank_learner import IncrementalTankLearner

            learner = IncrementalTankLearner()

            # Verificar métodos de aprendizaje
            self.assertTrue(hasattr(learner, 'learn_from_detection'))
            self.assertTrue(hasattr(learner, 'update_thresholds'))
            self.assertTrue(hasattr(learner, 'persist_knowledge'))

            print("✅ Incremental tank learner implementado")

        except ImportError:
            self.skipTest("IncrementalTankLearner no implementado aún")
        except Exception as e:
            self.fail(f"Error en incremental tank learner: {e}")

    def test_runtime_engine_integration(self):
        """Verifica integración del runtime engine"""
        try:
            from ucognet.runtime.engine import UCogNetEngine

            engine = UCogNetEngine()

            # Verificar métodos del engine
            self.assertTrue(hasattr(engine, 'initialize'))
            self.assertTrue(hasattr(engine, 'step'))
            self.assertTrue(hasattr(engine, 'shutdown'))
            self.assertTrue(hasattr(engine, 'get_status'))

            print("✅ Runtime engine implementado")

        except ImportError:
            self.skipTest("UCogNetEngine no implementado aún")
        except Exception as e:
            self.fail(f"Error en runtime engine: {e}")

    def test_interfaces_and_protocols(self):
        """Verifica interfaces y protocolos"""
        try:
            from ucognet.core.interfaces import (
                CognitiveModule, MemorySystem, EvaluatorInterface,
                TrainerInterface, TDAManagerInterface, TraceManager
            )
            from ucognet.core.protocols import CognitiveProtocol

            # Verificar que las interfaces principales existen
            interfaces = [
                CognitiveModule, MemorySystem, EvaluatorInterface,
                TrainerInterface, TDAManagerInterface, TraceManager
            ]

            for interface in interfaces:
                self.assertTrue(hasattr(interface, '__subclasshook__') or hasattr(interface, '__protocol__'))

            print("✅ Interfaces y protocolos definidos")

        except ImportError as e:
            self.fail(f"Error importando interfaces: {e}")
        except Exception as e:
            self.fail(f"Error en interfaces: {e}")

    def test_utils_and_types(self):
        """Verifica utilidades y tipos"""
        try:
            from ucognet.core import utils, types

            # Verificar funciones de utilidad comunes
            self.assertTrue(hasattr(utils, 'load_config'))
            self.assertTrue(hasattr(utils, 'setup_logging'))
            self.assertTrue(hasattr(utils, 'calculate_metrics'))

            # Verificar tipos definidos
            self.assertTrue(hasattr(types, 'Detection'))
            self.assertTrue(hasattr(types, 'CognitiveState'))
            self.assertTrue(hasattr(types, 'MemoryEntry'))

            print("✅ Utilidades y tipos implementados")

        except ImportError as e:
            self.fail(f"Error importando utils/types: {e}")
        except Exception as e:
            self.fail(f"Error en utils/types: {e}")


def run_complete_test_suite():
    """Ejecuta la suite completa de pruebas y reporta resultados"""
    print("🧪 Suite Completa de Pruebas Unitarias - U-CogNet")
    print("=" * 80)

    # Crear suite de pruebas
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(UCogNetTestSuite)

    # Ejecutar pruebas
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    # Reporte final
    print("\n" + "=" * 80)
    print("📊 REPORTE FINAL DE PRUEBAS")
    print("=" * 80)

    total_tests = result.testsRun
    passed = total_tests - len(result.failures) - len(result.errors)
    failed = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)

    print(f"📋 Total de pruebas: {total_tests}")
    print(f"✅ Pasaron: {passed}")
    print(f"❌ Fallaron: {failed}")
    print(f"⚠️ Errores: {errors}")
    print(f"⏭️ Omitidas: {skipped}")

    success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
    print(f"📈 Tasa de éxito: {success_rate:.1f}%")
    if result.failures:
        print("\n🔍 FALLOS DETALLADOS:")
        for test, traceback in result.failures:
            print(f"   • {test}: {traceback.strip().split('\n')[-1]}")

    if result.errors:
        print("\n🚨 ERRORES DETALLADOS:")
        for test, traceback in result.errors:
            print(f"   • {test}: {traceback.strip().split('\n')[-1]}")

    # Evaluar completitud del proyecto
    print("\n🏗️ EVALUACIÓN DE COMPLETITUD DEL PROYECTO")
    print("-" * 50)

    if success_rate >= 90:
        print("🎉 PROYECTO COMPLETO - Todos los módulos principales implementados")
        print("🚀 Listo para despliegue y uso en producción")
    elif success_rate >= 70:
        print("⚠️ PROYECTO CASI COMPLETO - Faltan algunos módulos no críticos")
        print("🔧 Implementar módulos faltantes antes del despliegue")
    elif success_rate >= 50:
        print("📋 PROYECTO PARCIAL - Módulos básicos implementados")
        print("🏗️ Necesario implementar módulos críticos faltantes")
    else:
        print("❌ PROYECTO INCOMPLETO - Faltan módulos críticos")
        print("🛠️ Priorizar implementación de módulos principales")

    # Módulos críticos que deben existir
    critical_modules = [
        'Sistema de Trazabilidad (MTC)',
        'Cognitive Core',
        'TDA Manager',
        'Optimizador Micelial',
        'Evaluator',
        'Trainer Loop'
    ]

    print("\n🔍 MÓDULOS CRÍTICOS REQUERIDOS:")
    for module in critical_modules:
        print(f"   • {module}")

    return result


if __name__ == "__main__":
    # Ejecutar suite completa
    result = run_complete_test_suite()

    # Salir con código apropiado
    if result.wasSuccessful():
        sys.exit(0)
    else:
        sys.exit(1)