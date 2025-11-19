"""
Topología Dinámica Adaptativa (TDA) Manager para U-CogNet
Orquesta la reorganización automática de la arquitectura del sistema
"""

import asyncio
import time
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .interfaces import TDAManagerInterface
from .tracing import get_event_bus, EventType


class TopologyState(Enum):
    """Estados de la topología del sistema"""
    STABLE = "stable"
    ADAPTING = "adapting"
    OPTIMIZING = "optimizing"
    RECOVERING = "recovering"


@dataclass
class ModuleMetrics:
    """Métricas de rendimiento de un módulo"""
    name: str
    performance_score: float
    resource_usage: float
    error_rate: float
    activation_frequency: float
    last_used: float
    dependencies: Set[str]
    capabilities: Set[str]


@dataclass
class TopologyConfig:
    """Configuración de topología"""
    adaptation_threshold: float = 0.7
    optimization_interval: int = 100
    max_modules_per_path: int = 5
    min_performance_threshold: float = 0.6
    resource_limit: float = 0.8


class TDAManager(TDAManagerInterface):
    """
    Gestor de Topología Dinámica Adaptativa
    Monitorea rendimiento y reorganiza automáticamente la arquitectura del sistema
    """

    def __init__(self, config: Optional[TopologyConfig] = None):
        self.config = config or TopologyConfig()
        self.event_bus = get_event_bus()

        # Estado de la topología
        self.topology_graph = nx.DiGraph()
        self.module_metrics: Dict[str, ModuleMetrics] = {}
        self.active_modules: Set[str] = set()
        self.topology_state = TopologyState.STABLE

        # Historial y análisis
        self.performance_history: List[Dict[str, Any]] = []
        self.topology_changes: List[Dict[str, Any]] = []
        self.adaptation_cycles = 0

        # Control de adaptación
        self.last_adaptation = time.time()
        self.adaptation_cooldown = 60  # segundos
        self.performance_window = 50  # mediciones para análisis

        # Sistema de predicción
        self.performance_predictor = None

        print("🧠 TDA Manager inicializado")

    async def initialize(self) -> bool:
        """Inicializa el sistema TDA"""
        try:
            # Registrar módulos iniciales
            await self._register_core_modules()

            # Construir grafo inicial
            await self._build_initial_topology()

            # Iniciar monitoreo continuo
            asyncio.create_task(self._continuous_monitoring())

            # Emitir evento de inicialización
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "TDAManager",
                outputs={"initialized": True, "modules_registered": len(self.module_metrics)},
                explanation="Inicialización del TDA Manager"
            )

            print("✅ TDA Manager inicializado correctamente")
            return True

        except Exception as e:
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "TDAManager",
                outputs={"initialized": False, "error": str(e)},
                log_level=2
            )
            print(f"❌ Error inicializando TDA Manager: {e}")
            return False

    async def _register_core_modules(self):
        """Registra los módulos principales del sistema"""
        core_modules = [
            {
                "name": "input_handler",
                "capabilities": {"vision", "audio", "text", "timeseries"},
                "dependencies": set()
            },
            {
                "name": "vision_detector",
                "capabilities": {"object_detection", "scene_analysis"},
                "dependencies": {"input_handler"}
            },
            {
                "name": "cognitive_core",
                "capabilities": {"reasoning", "decision_making", "memory"},
                "dependencies": {"vision_detector", "input_handler"}
            },
            {
                "name": "evaluator",
                "capabilities": {"performance_analysis", "metrics_calculation"},
                "dependencies": {"cognitive_core"}
            },
            {
                "name": "trainer_loop",
                "capabilities": {"model_training", "parameter_update"},
                "dependencies": {"evaluator", "cognitive_core"}
            },
            {
                "name": "semantic_feedback",
                "capabilities": {"natural_language", "explanation_generation"},
                "dependencies": {"cognitive_core"}
            },
            {
                "name": "visual_interface",
                "capabilities": {"display", "hud_rendering"},
                "dependencies": {"vision_detector", "semantic_feedback"}
            },
            {
                "name": "mycelial_optimizer",
                "capabilities": {"parameter_optimization", "topology_adaptation"},
                "dependencies": {"evaluator", "trainer_loop"}
            }
        ]

        for module_info in core_modules:
            metrics = ModuleMetrics(
                name=module_info["name"],
                performance_score=0.8,  # puntuación inicial
                resource_usage=0.1,
                error_rate=0.0,
                activation_frequency=0.0,
                last_used=time.time(),
                dependencies=module_info["dependencies"],
                capabilities=module_info["capabilities"]
            )

            self.module_metrics[module_info["name"]] = metrics
            self.topology_graph.add_node(module_info["name"], **module_info)

    async def _build_initial_topology(self):
        """Construye la topología inicial del sistema"""
        # Crear conexiones basadas en dependencias
        for module_name, metrics in self.module_metrics.items():
            for dependency in metrics.dependencies:
                if dependency in self.module_metrics:
                    self.topology_graph.add_edge(dependency, module_name, weight=1.0)

        # Activar módulos iniciales
        initial_modules = {"input_handler", "vision_detector", "cognitive_core"}
        self.active_modules.update(initial_modules)

        print(f"📊 Topología inicial construida con {len(self.topology_graph.nodes)} nodos y {len(self.topology_graph.edges)} conexiones")

    async def _continuous_monitoring(self):
        """Monitoreo continuo del rendimiento del sistema"""
        while True:
            try:
                # Recopilar métricas de rendimiento
                await self._collect_performance_metrics()

                # Analizar necesidad de adaptación
                needs_adaptation = await self._analyze_adaptation_need()

                if needs_adaptation:
                    await self.adapt_topology()

                # Optimizar periódicamente
                if self.adaptation_cycles % self.config.optimization_interval == 0:
                    await self._optimize_topology()

                await asyncio.sleep(10)  # monitoreo cada 10 segundos

            except Exception as e:
                print(f"⚠️ Error en monitoreo continuo: {e}")
                await asyncio.sleep(30)

    async def _collect_performance_metrics(self):
        """Recopila métricas de rendimiento de todos los módulos"""
        current_time = time.time()

        for module_name in self.active_modules:
            if module_name in self.module_metrics:
                # Simular recopilación de métricas (en implementación real vendrían de los módulos)
                metrics = self.module_metrics[module_name]

                # Actualizar métricas simuladas
                metrics.performance_score = np.random.normal(0.8, 0.1)
                metrics.resource_usage = np.random.normal(0.3, 0.1)
                metrics.error_rate = max(0, np.random.normal(0.02, 0.01))
                metrics.last_used = current_time

                # Calcular frecuencia de activación
                recent_activations = sum(1 for entry in self.performance_history[-20:]
                                       if entry.get("module") == module_name)
                metrics.activation_frequency = recent_activations / 20.0

    async def _analyze_adaptation_need(self) -> bool:
        """Analiza si es necesaria una adaptación de topología"""
        if time.time() - self.last_adaptation < self.adaptation_cooldown:
            return False

        # Calcular métricas globales
        active_metrics = [self.module_metrics[name] for name in self.active_modules
                         if name in self.module_metrics]

        if not active_metrics:
            return False

        avg_performance = np.mean([m.performance_score for m in active_metrics])
        avg_resource_usage = np.mean([m.resource_usage for m in active_metrics])
        max_error_rate = max([m.error_rate for m in active_metrics])

        # Criterios para adaptación
        performance_degraded = avg_performance < self.config.min_performance_threshold
        resource_overloaded = avg_resource_usage > self.config.resource_limit
        high_error_rate = max_error_rate > 0.1

        needs_adaptation = performance_degraded or resource_overloaded or high_error_rate

        if needs_adaptation:
            print(f"🔄 Adaptación necesaria - Performance: {avg_performance:.2f}, Recursos: {avg_resource_usage:.2f}, Error máximo: {max_error_rate:.2f}")

        return needs_adaptation

    async def evaluate_topology(self) -> Dict[str, Any]:
        """
        Evalúa el estado actual de la topología

        Returns:
            Evaluación completa de la topología
        """
        evaluation = {
            "topology_state": self.topology_state.value,
            "active_modules": len(self.active_modules),
            "total_modules": len(self.module_metrics),
            "graph_density": nx.density(self.topology_graph),
            "average_clustering": nx.average_clustering(self.topology_graph),
            "adaptation_cycles": self.adaptation_cycles,
            "module_performance": {},
            "topology_metrics": {}
        }

        # Evaluar rendimiento de módulos
        for name, metrics in self.module_metrics.items():
            evaluation["module_performance"][name] = {
                "performance_score": metrics.performance_score,
                "resource_usage": metrics.resource_usage,
                "error_rate": metrics.error_rate,
                "activation_frequency": metrics.activation_frequency,
                "is_active": name in self.active_modules
            }

        # Métricas de topología
        evaluation["topology_metrics"] = {
            "num_nodes": len(self.topology_graph.nodes),
            "num_edges": len(self.topology_graph.edges),
            "is_connected": nx.is_weakly_connected(self.topology_graph) if self.topology_graph.edges else False,
            "average_path_length": nx.average_shortest_path_length(self.topology_graph) if nx.is_weakly_connected(self.topology_graph) and self.topology_graph.edges else 0,
            "degree_centrality": nx.degree_centrality(self.topology_graph)
        }

        return evaluation

    async def adapt_topology(self) -> bool:
        """
        Adapta la topología del sistema basado en rendimiento

        Returns:
            True si la adaptación fue exitosa
        """
        self.topology_state = TopologyState.ADAPTING
        self.last_adaptation = time.time()

        try:
            # Emitir evento de adaptación
            self.event_bus.emit(
                EventType.LEARNING_STEP,
                "TDAManager",
                inputs={"adaptation_reason": "performance_optimization"},
                explanation="Inicio de adaptación de topología"
            )

            # Identificar módulos problemáticos
            problematic_modules = self._identify_problematic_modules()
            underutilized_modules = self._identify_underutilized_modules()

            # Desactivar módulos problemáticos
            for module in problematic_modules:
                if module in self.active_modules:
                    self.active_modules.remove(module)
                    print(f"🔽 Desactivando módulo problemático: {module}")

            # Activar módulos útiles
            for module in underutilized_modules[:2]:  # máximo 2 a la vez
                if module not in self.active_modules:
                    self.active_modules.add(module)
                    print(f"🔼 Activando módulo útil: {module}")

            # Reorganizar conexiones
            await self._reorganize_connections()

            # Registrar cambio
            change_record = {
                "timestamp": time.time(),
                "cycle": self.adaptation_cycles,
                "deactivated": list(problematic_modules),
                "activated": list(underutilized_modules),
                "topology_snapshot": self._get_topology_snapshot()
            }

            self.topology_changes.append(change_record)
            self.adaptation_cycles += 1

            self.topology_state = TopologyState.STABLE

            # Emitir evento de finalización
            self.event_bus.emit(
                EventType.LEARNING_STEP,
                "TDAManager",
                outputs={"adaptation_success": True, "modules_changed": len(problematic_modules) + len(underutilized_modules)},
                explanation="Adaptación de topología completada"
            )

            print(f"✅ Adaptación completada - Ciclo {self.adaptation_cycles}")
            return True

        except Exception as e:
            self.topology_state = TopologyState.RECOVERING
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "TDAManager",
                outputs={"adaptation_success": False, "error": str(e)},
                log_level=2
            )
            print(f"❌ Error en adaptación: {e}")
            return False

    def _identify_problematic_modules(self) -> Set[str]:
        """Identifica módulos con bajo rendimiento"""
        problematic = set()

        for name, metrics in self.module_metrics.items():
            if (metrics.performance_score < self.config.min_performance_threshold or
                metrics.error_rate > 0.05 or
                metrics.resource_usage > self.config.resource_limit):
                problematic.add(name)

        return problematic

    def _identify_underutilized_modules(self) -> List[str]:
        """Identifica módulos útiles pero inactivos"""
        candidates = []

        for name, metrics in self.module_metrics.items():
            if (name not in self.active_modules and
                metrics.performance_score > self.config.adaptation_threshold and
                metrics.error_rate < 0.02 and
                metrics.resource_usage < 0.5):
                candidates.append((name, metrics.performance_score))

        # Ordenar por rendimiento
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in candidates]

    async def _reorganize_connections(self):
        """Reorganiza las conexiones del grafo de topología"""
        # Simplificar: remover conexiones redundantes y agregar conexiones útiles
        edges_to_remove = []
        edges_to_add = []

        # Identificar conexiones redundantes (múltiples caminos)
        for node in self.topology_graph.nodes():
            predecessors = list(self.topology_graph.predecessors(node))
            if len(predecessors) > 3:  # máximo 3 conexiones entrantes
                # Remover las conexiones más débiles
                edge_weights = [(pred, self.topology_graph[pred][node].get('weight', 1.0))
                              for pred in predecessors]
                edge_weights.sort(key=lambda x: x[1])

                # Remover las 2 más débiles
                for pred, _ in edge_weights[:2]:
                    edges_to_remove.append((pred, node))

        # Agregar conexiones entre módulos complementarios
        for module1 in self.active_modules:
            for module2 in self.active_modules:
                if (module1 != module2 and
                    not self.topology_graph.has_edge(module1, module2) and
                    self._are_modules_complementary(module1, module2)):
                    edges_to_add.append((module1, module2, 0.5))  # peso inicial

        # Aplicar cambios
        for edge in edges_to_remove:
            if self.topology_graph.has_edge(*edge):
                self.topology_graph.remove_edge(*edge)

        for src, dst, weight in edges_to_add:
            self.topology_graph.add_edge(src, dst, weight=weight)

        print(f"🔄 Reorganización: -{len(edges_to_remove)} aristas, +{len(edges_to_add)} aristas")

    def _are_modules_complementary(self, module1: str, module2: str) -> bool:
        """Determina si dos módulos son complementarios"""
        if module1 not in self.module_metrics or module2 not in self.module_metrics:
            return False

        caps1 = self.module_metrics[module1].capabilities
        caps2 = self.module_metrics[module2].capabilities

        # Complementarios si tienen capacidades diferentes pero relacionadas
        intersection = caps1 & caps2
        union = caps1 | caps2

        if not union:
            return False

        overlap_ratio = len(intersection) / len(union)
        return overlap_ratio < 0.3  # baja superposición = complementarios

    async def _optimize_topology(self):
        """Optimiza la topología global del sistema"""
        self.topology_state = TopologyState.OPTIMIZING

        try:
            print("🔧 Iniciando optimización de topología global...")

            # Análisis de componentes conectados
            if not nx.is_weakly_connected(self.topology_graph):
                # Reconectar componentes desconectados
                components = list(nx.weakly_connected_components(self.topology_graph))
                if len(components) > 1:
                    await self._reconnect_components(components)

            # Optimizar pesos de aristas basado en uso
            await self._optimize_edge_weights()

            # Identificar y eliminar bottlenecks
            await self._remove_bottlenecks()

            print("✅ Optimización de topología completada")

        except Exception as e:
            print(f"⚠️ Error en optimización: {e}")
        finally:
            self.topology_state = TopologyState.STABLE

    async def _reconnect_components(self, components: List[Set[str]]):
        """Reconecta componentes desconectados"""
        for i in range(len(components) - 1):
            # Conectar el último nodo de un componente con el primero del siguiente
            comp1_nodes = list(components[i])
            comp2_nodes = list(components[i + 1])

            if comp1_nodes and comp2_nodes:
                src = comp1_nodes[-1]
                dst = comp2_nodes[0]
                self.topology_graph.add_edge(src, dst, weight=0.3)
                print(f"🔗 Reconectando componentes: {src} -> {dst}")

    async def _optimize_edge_weights(self):
        """Optimiza pesos de aristas basado en rendimiento"""
        for src, dst, data in self.topology_graph.edges(data=True):
            # Aumentar peso si ambos módulos están activos y funcionando bien
            if (src in self.active_modules and dst in self.active_modules and
                src in self.module_metrics and dst in self.module_metrics):

                src_perf = self.module_metrics[src].performance_score
                dst_perf = self.module_metrics[dst].performance_score
                avg_perf = (src_perf + dst_perf) / 2

                # Ajustar peso basado en rendimiento conjunto
                current_weight = data.get('weight', 1.0)
                new_weight = current_weight * (0.9 + 0.2 * avg_perf)  # ajuste suave
                data['weight'] = np.clip(new_weight, 0.1, 2.0)

    async def _remove_bottlenecks(self):
        """Identifica y elimina bottlenecks en la topología"""
        # Calcular betweenness centrality
        centrality = nx.betweenness_centrality(self.topology_graph)

        # Identificar nodos con alta centralidad pero bajo rendimiento
        bottlenecks = []
        for node, cent in centrality.items():
            if (node in self.module_metrics and
                cent > 0.5 and  # alta centralidad
                self.module_metrics[node].performance_score < 0.7):  # bajo rendimiento
                bottlenecks.append(node)

        # Crear rutas alternativas para bottlenecks
        for bottleneck in bottlenecks[:2]:  # máximo 2 a la vez
            await self._create_alternative_path(bottleneck)

    async def _create_alternative_path(self, bottleneck: str):
        """Crea una ruta alternativa para evitar un bottleneck"""
        # Encontrar nodos conectados al bottleneck
        predecessors = list(self.topology_graph.predecessors(bottleneck))
        successors = list(self.topology_graph.successors(bottleneck))

        # Crear conexiones directas entre predecesores y sucesores
        for pred in predecessors[:2]:  # limitar a 2
            for succ in successors[:2]:
                if not self.topology_graph.has_edge(pred, succ):
                    self.topology_graph.add_edge(pred, succ, weight=0.4)
                    print(f"🔄 Ruta alternativa creada: {pred} -> {succ}")

    def _get_topology_snapshot(self) -> Dict[str, Any]:
        """Obtiene un snapshot del estado actual de la topología"""
        return {
            "nodes": list(self.topology_graph.nodes()),
            "edges": list(self.topology_graph.edges(data=True)),
            "active_modules": list(self.active_modules),
            "topology_state": self.topology_state.value
        }

    def get_active_modules(self) -> List[str]:
        """Obtiene la lista de módulos activos actualmente"""
        return list(self.active_modules)

    def get_topology_status(self) -> Dict[str, Any]:
        """Obtiene el estado completo de la topología"""
        return {
            "state": self.topology_state.value,
            "active_modules": self.get_active_modules(),
            "total_modules": len(self.module_metrics),
            "graph_metrics": {
                "nodes": len(self.topology_graph.nodes),
                "edges": len(self.topology_graph.edges),
                "density": nx.density(self.topology_graph),
                "is_connected": nx.is_weakly_connected(self.topology_graph) if self.topology_graph.edges else False
            },
            "adaptation_info": {
                "cycles": self.adaptation_cycles,
                "last_adaptation": self.last_adaptation,
                "changes_count": len(self.topology_changes)
            }
        }