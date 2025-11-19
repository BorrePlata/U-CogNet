#!/usr/bin/env python3
"""
Validación Completa del Sistema de Trazabilidad Cognitiva (MTC)
Ejecuta todos los componentes y valida la integración completa
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

# Importar todos los componentes del MTC
from ucognet.core.tracing import (
    CognitiveEvent, EventType, LogLevel,
    CognitiveTraceCore, CognitiveEventBus,
    CausalGraphBuilder, CoherenceEthicsEvaluator,
    TraceQueryAPI, TraceStorageManager,
    get_event_bus, get_trace_core
)

class ValidationSuite:
    """Suite completa de validación del MTC"""

    def __init__(self):
        self.event_bus = get_event_bus()
        self.trace_core = get_trace_core()
        self.causal_builder = CausalGraphBuilder()
        self.coherence_evaluator = CoherenceEthicsEvaluator()
        self.query_api = TraceQueryAPI(
            self.trace_core,
            self.causal_builder,
            self.coherence_evaluator
        )
        self.storage_manager = TraceStorageManager()

        self.test_results = {}
        self.episode_id = f"validation_{int(time.time())}"

    async def run_full_validation(self) -> Dict[str, Any]:
        """Ejecuta validación completa del sistema"""
        print("🧪 Iniciando Validación Completa del MTC")
        print("=" * 60)

        # 1. Validar esquema de eventos
        await self._validate_event_schema()

        # 2. Validar bus de eventos
        await self._validate_event_bus()

        # 3. Validar núcleo de trazas
        await self._validate_trace_core()

        # 4. Validar constructor causal
        await self._validate_causal_builder()

        # 5. Validar evaluador de coherencia/ética
        await self._validate_coherence_evaluator()

        # 6. Validar API de consultas
        await self._validate_query_api()

        # 7. Validar gestor de almacenamiento
        await self._validate_storage_manager()

        # 8. Validar integración completa
        await self._validate_full_integration()

        # 9. Generar reporte final
        return self._generate_validation_report()

    async def _validate_event_schema(self):
        """Valida el esquema de eventos cognitivos"""
        print("1. Validando esquema de eventos...")

        try:
            # Crear evento básico
            event = CognitiveEvent(
                event_type=EventType.DECISION,
                source="ValidationSuite",
                inputs={"test": "data"},
                outputs={"result": "success"},
                episode_id=self.episode_id,
                step_id=0
            )

            # Validar estructura
            assert event.event_id is not None
            assert event.timestamp is not None
            assert event.trace_id is not None
            assert event.span_id is not None

            # Validar serialización
            event_dict = event.to_dict()
            reconstructed = CognitiveEvent.from_dict(event_dict)
            assert reconstructed.event_type == event.event_type
            assert reconstructed.source == event.source

            self.test_results["event_schema"] = {"status": "PASS", "details": "Esquema válido"}
            print("   ✅ Esquema de eventos validado")

        except Exception as e:
            self.test_results["event_schema"] = {"status": "FAIL", "error": str(e)}
            print(f"   ❌ Error en esquema: {e}")

    async def _validate_event_bus(self):
        """Valida el bus de eventos"""
        print("2. Validando bus de eventos...")

        try:
            # Emitir evento de prueba
            event_id = self.event_bus.emit(
                EventType.DECISION,
                "ValidationSuite",
                inputs={"test": "bus"},
                episode_id=self.episode_id,
                step_id=1
            )

            # Verificar que se registró
            assert event_id is not None

            # Verificar que el evento existe en el núcleo
            event = self.trace_core.get_event(event_id)
            assert event is not None
            assert event.source == "ValidationSuite"

            self.test_results["event_bus"] = {"status": "PASS", "details": "Bus funcional"}
            print("   ✅ Bus de eventos validado")

        except Exception as e:
            self.test_results["event_bus"] = {"status": "FAIL", "error": str(e)}
            print(f"   ❌ Error en bus: {e}")

    async def _validate_trace_core(self):
        """Valida el núcleo de trazas"""
        print("3. Validando núcleo de trazas...")

        try:
            # Obtener eventos del episodio
            events = self.trace_core.get_events_by_episode(self.episode_id)
            assert len(events) >= 2  # Al menos los eventos emitidos

            # Verificar trazas
            traces = self.trace_core.get_traces_by_episode(self.episode_id)
            assert len(traces) > 0

            # Verificar estadísticas
            stats = self.trace_core.get_episode_stats(self.episode_id)
            assert "total_events" in stats
            assert "event_types" in stats

            self.test_results["trace_core"] = {"status": "PASS", "details": f"{len(events)} eventos procesados"}
            print("   ✅ Núcleo de trazas validado")

        except Exception as e:
            self.test_results["trace_core"] = {"status": "FAIL", "error": str(e)}
            print(f"   ❌ Error en núcleo: {e}")

    async def _validate_causal_builder(self):
        """Valida el constructor de grafos causales"""
        print("4. Validando constructor causal...")

        try:
            # Construir grafo causal
            causal_graph = self.causal_builder.build_causal_graph(self.episode_id)

            # Verificar estructura del grafo
            assert hasattr(causal_graph, 'nodes')
            assert hasattr(causal_graph, 'edges')

            # Verificar análisis causal
            analysis = self.causal_builder.analyze_causal_relationships(self.episode_id)
            assert "causal_links" in analysis
            assert "confidence_scores" in analysis

            self.test_results["causal_builder"] = {
                "status": "PASS",
                "details": f"Grafo con {len(causal_graph.nodes)} nodos, {len(causal_graph.edges)} aristas"
            }
            print("   ✅ Constructor causal validado")

        except Exception as e:
            self.test_results["causal_builder"] = {"status": "FAIL", "error": str(e)}
            print(f"   ❌ Error en constructor causal: {e}")

    async def _validate_coherence_evaluator(self):
        """Valida el evaluador de coherencia y ética"""
        print("5. Validando evaluador de coherencia/ética...")

        try:
            # Evaluar coherencia del episodio
            coherence_score = self.coherence_evaluator.evaluate_coherence(self.episode_id)
            assert isinstance(coherence_score, dict)
            assert "overall_coherence" in coherence_score

            # Evaluar ética
            ethics_score = self.coherence_evaluator.evaluate_ethics(self.episode_id)
            assert isinstance(ethics_score, dict)
            assert "overall_ethics" in ethics_score

            # Verificar métricas detalladas
            detailed_metrics = self.coherence_evaluator.get_detailed_metrics(self.episode_id)
            assert "coherence_metrics" in detailed_metrics
            assert "ethics_criteria" in detailed_metrics

            self.test_results["coherence_evaluator"] = {
                "status": "PASS",
                "details": f"Coherencia: {coherence_score['overall_coherence']:.2f}, Ética: {ethics_score['overall_ethics']:.2f}"
            }
            print("   ✅ Evaluador de coherencia/ética validado")

        except Exception as e:
            self.test_results["coherence_evaluator"] = {"status": "FAIL", "error": str(e)}
            print(f"   ❌ Error en evaluador: {e}")

    async def _validate_query_api(self):
        """Valida la API de consultas"""
        print("6. Validando API de consultas...")

        try:
            # Obtener resumen del episodio
            summary = self.query_api.get_episode_summary(self.episode_id)
            assert "total_events" in summary
            assert "episode_id" in summary

            # Obtener análisis causal
            causal_analysis = self.query_api.get_causal_analysis(self.episode_id)
            assert "causal_links" in causal_analysis

            # Obtener reporte de salud
            health_report = self.query_api.get_system_health_report(hours=1)
            assert "total_events" in health_report

            # Realizar consulta temporal
            time_query = self.query_api.query_events_by_time_range(
                start_time=time.time() - 3600,
                end_time=time.time()
            )
            assert isinstance(time_query, list)

            self.test_results["query_api"] = {"status": "PASS", "details": "API funcional"}
            print("   ✅ API de consultas validada")

        except Exception as e:
            self.test_results["query_api"] = {"status": "FAIL", "error": str(e)}
            print(f"   ❌ Error en API: {e}")

    async def _validate_storage_manager(self):
        """Valida el gestor de almacenamiento"""
        print("7. Validando gestor de almacenamiento...")

        try:
            # Forzar guardado
            await self.storage_manager.save_events_async()

            # Verificar archivos creados
            storage_stats = self.storage_manager.get_storage_stats()
            assert "total_events_stored" in storage_stats

            # Verificar compresión si está habilitada
            if self.storage_manager.compression_enabled:
                assert storage_stats.get("compression_ratio", 0) > 0

            # Verificar políticas de retención
            retention_info = self.storage_manager.get_retention_info()
            assert "retention_days" in retention_info

            self.test_results["storage_manager"] = {
                "status": "PASS",
                "details": f"{storage_stats['total_events_stored']} eventos almacenados"
            }
            print("   ✅ Gestor de almacenamiento validado")

        except Exception as e:
            self.test_results["storage_manager"] = {"status": "FAIL", "error": str(e)}
            print(f"   ❌ Error en almacenamiento: {e}")

    async def _validate_full_integration(self):
        """Valida la integración completa del sistema"""
        print("8. Validando integración completa...")

        try:
            # Crear un flujo completo de trazabilidad
            integration_episode = f"integration_test_{int(time.time())}"

            # 1. Emitir evento inicial
            event1_id = self.event_bus.emit(
                EventType.DECISION,
                "IntegrationTest",
                inputs={"phase": "start"},
                episode_id=integration_episode,
                step_id=0
            )

            # 2. Simular procesamiento
            await asyncio.sleep(0.1)

            # 3. Emitir evento de resultado
            event2_id = self.event_bus.emit(
                EventType.REWARD,
                "IntegrationTest",
                outputs={"result": "success"},
                episode_id=integration_episode,
                step_id=1,
                trace_id=event1_id
            )

            # 4. Esperar procesamiento asíncrono
            await asyncio.sleep(0.2)

            # 5. Verificar trazabilidad completa
            summary = self.query_api.get_episode_summary(integration_episode)
            assert summary["total_events"] >= 2

            causal = self.query_api.get_causal_analysis(integration_episode)
            assert len(causal.get("causal_links", [])) > 0

            coherence = self.coherence_evaluator.evaluate_coherence(integration_episode)
            assert coherence["overall_coherence"] >= 0

            ethics = self.coherence_evaluator.evaluate_ethics(integration_episode)
            assert ethics["overall_ethics"] >= 0

            self.test_results["full_integration"] = {
                "status": "PASS",
                "details": "Flujo completo validado"
            }
            print("   ✅ Integración completa validada")

        except Exception as e:
            self.test_results["full_integration"] = {"status": "FAIL", "error": str(e)}
            print(f"   ❌ Error en integración: {e}")

    def _generate_validation_report(self) -> Dict[str, Any]:
        """Genera reporte completo de validación"""
        print("\n📊 Reporte de Validación del MTC")
        print("=" * 60)

        passed = 0
        failed = 0

        for component, result in self.test_results.items():
            status = result["status"]
            if status == "PASS":
                passed += 1
                print(f"✅ {component}: {result.get('details', 'OK')}")
            else:
                failed += 1
                print(f"❌ {component}: {result.get('error', 'Error desconocido')}")

        total_tests = len(self.test_results)
        success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0

        print(f"\n📈 Resultados: {passed}/{total_tests} pruebas pasaron ({success_rate:.1f}%)")

        # Resumen del sistema
        system_summary = {
            "validation_complete": failed == 0,
            "total_components": total_tests,
            "passed_components": passed,
            "failed_components": failed,
            "success_rate": success_rate,
            "episode_id": self.episode_id,
            "timestamp": time.time(),
            "component_results": self.test_results
        }

        if failed == 0:
            print("🎉 ¡Validación completa exitosa!")
            print("🧠 El MTC está listo para producción")
        else:
            print("⚠️ Algunas validaciones fallaron")
            print("🔧 Revisa los componentes con error")

        return system_summary

async def main():
    """Función principal de validación"""
    print("🧪 Suite de Validación Completa - Sistema de Trazabilidad Cognitiva")
    print("=" * 80)

    try:
        # Ejecutar validación completa
        validator = ValidationSuite()
        report = await validator.run_full_validation()

        # Mostrar resumen final
        print("\n" + "=" * 80)
        if report["validation_complete"]:
            print("🎯 VALIDACIÓN EXITOSA")
            print("📦 El MTC está completamente funcional y listo para uso")
            print("🚀 Puede integrarse en cualquier aplicación cognitiva")
        else:
            print("⚠️ VALIDACIÓN CON ERRORES")
            print("🔧 Revisa los componentes fallidos antes de usar en producción")

        print(f"📊 Tasa de éxito: {report['success_rate']:.1f}%")
        print(f"🎮 Episodio de validación: {report['episode_id']}")

    except Exception as e:
        print(f"❌ Error crítico en validación: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())