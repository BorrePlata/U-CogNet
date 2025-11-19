"""
Mycelial Optimizer - Optimizador Inspirado en Redes de Hongos
Implementa optimización adaptativa basada en el concepto micelial
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging

from .interfaces import CognitiveModule
from .tracing import get_event_bus, EventType


class MycelialState(Enum):
    """Estados del optimizador micelial"""
    EXPLORING = "exploring"
    EXPLOITING = "exploiting"
    ADAPTING = "adapting"
    PRUNING = "pruning"


@dataclass
class MycelialNode:
    """Nodo en la red de parámetros micelial"""
    parameter_name: str
    current_value: float
    learning_rate: float
    nutrient_level: float  # "nutrientes" = gradientes/actualizaciones útiles
    connections: Set[str]  # conexiones con otros parámetros
    last_update: float
    contribution_score: float  # contribución al rendimiento global


@dataclass
class MycelialCluster:
    """Cluster de parámetros relacionados"""
    name: str
    parameters: Set[str]
    performance_score: float
    resource_allocation: float
    growth_rate: float
    last_optimization: float


class MycelialOptimizer(CognitiveModule):
    """
    Optimizador inspirado en micelio de hongos
    Distribuye recursos de aprendizaje basándose en utilidad y conectividad
    """

    def __init__(self):
        self.event_bus = get_event_bus()

        # Red micelial de parámetros
        self.mycelial_nodes: Dict[str, MycelialNode] = {}
        self.clusters: Dict[str, MycelialCluster] = {}

        # Estado del optimizador
        self.state = MycelialState.EXPLORING
        self.global_nutrient_level = 1.0
        self.adaptation_rate = 0.05
        self.pruning_threshold = 0.1

        # Control de optimización
        self.optimization_interval = 100  # pasos
        self.step_counter = 0
        self.last_optimization = time.time()

        # Historial de rendimiento
        self.performance_history: List[float] = []
        self.parameter_contributions: Dict[str, List[float]] = {}

        print("🍄 Mycelial Optimizer inicializado")

    async def initialize(self) -> bool:
        """Inicializa el optimizador micelial"""
        try:
            # Crear clusters iniciales de parámetros
            await self._initialize_parameter_clusters()

            # Establecer conexiones iniciales
            await self._establish_initial_connections()

            # Iniciar optimización continua
            # asyncio.create_task(self._continuous_optimization())

            # Emitir evento de inicialización
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "MycelialOptimizer",
                outputs={
                    "initialized": True,
                    "clusters_created": len(self.clusters),
                    "nodes_created": len(self.mycelial_nodes)
                },
                explanation="Inicialización del Mycelial Optimizer"
            )

            print("✅ Mycelial Optimizer inicializado correctamente")
            return True

        except Exception as e:
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "MycelialOptimizer",
                outputs={"initialized": False, "error": str(e)},
                log_level=2
            )
            print(f"❌ Error inicializando Mycelial Optimizer: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del optimizador micelial"""
        return {
            "module": "MycelialOptimizer",
            "state": self.state.value if self.state else "uninitialized",
            "clusters": len(self.clusters),
            "nodes": len(self.mycelial_nodes),
            "total_connections": sum(len(node.connections) for node in self.mycelial_nodes.values()),
            "optimization_cycles": self.optimization_cycles,
            "last_optimization": self.last_optimization,
            "performance_score": self.performance_score,
            "active_adaptations": len(self.active_adaptations) if self.active_adaptations else 0
        }

    async def _initialize_parameter_clusters(self):
        """Inicializa clusters de parámetros por capas/funciones"""
        # Clusters típicos en una red neuronal
        cluster_definitions = [
            {
                "name": "vision_encoder",
                "parameters": ["conv1_weight", "conv1_bias", "conv2_weight", "conv2_bias"],
                "layer_type": "convolutional"
            },
            {
                "name": "feature_extractor",
                "parameters": ["fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"],
                "layer_type": "fully_connected"
            },
            {
                "name": "attention_mechanism",
                "parameters": ["attn_query", "attn_key", "attn_value", "attn_output"],
                "layer_type": "attention"
            },
            {
                "name": "output_layer",
                "parameters": ["output_weight", "output_bias"],
                "layer_type": "classification"
            }
        ]

        for cluster_def in cluster_definitions:
            # Crear nodos para cada parámetro
            for param_name in cluster_def["parameters"]:
                node = MycelialNode(
                    parameter_name=param_name,
                    current_value=0.0,  # será actualizado durante optimización
                    learning_rate=0.001,
                    nutrient_level=1.0,
                    connections=set(),
                    last_update=time.time(),
                    contribution_score=0.5
                )
                self.mycelial_nodes[param_name] = node

            # Crear cluster
            cluster = MycelialCluster(
                name=cluster_def["name"],
                parameters=set(cluster_def["parameters"]),
                performance_score=0.5,
                resource_allocation=1.0 / len(cluster_definitions),  # distribución equitativa inicial
                growth_rate=0.0,
                last_optimization=time.time()
            )
            self.clusters[cluster_def["name"]] = cluster

    async def _establish_initial_connections(self):
        """Establece conexiones iniciales entre parámetros"""
        # Conexiones intra-cluster (fuertes)
        for cluster in self.clusters.values():
            params_list = list(cluster.parameters)
            for i, param1 in enumerate(params_list):
                for param2 in params_list[i+1:]:
                    if param1 in self.mycelial_nodes and param2 in self.mycelial_nodes:
                        self.mycelial_nodes[param1].connections.add(param2)
                        self.mycelial_nodes[param2].connections.add(param1)

        # Conexiones inter-cluster (débilies iniciales)
        cluster_names = list(self.clusters.keys())
        for i, cluster1_name in enumerate(cluster_names):
            for cluster2_name in cluster_names[i+1:]:
                cluster1 = self.clusters[cluster1_name]
                cluster2 = self.clusters[cluster2_name]

                # Conectar algunos parámetros entre clusters
                params1 = list(cluster1.parameters)[:2]  # primeros 2 parámetros
                params2 = list(cluster2.parameters)[:2]

                for p1 in params1:
                    for p2 in params2:
                        if np.random.rand() < 0.3:  # 30% probabilidad de conexión
                            if p1 in self.mycelial_nodes and p2 in self.mycelial_nodes:
                                self.mycelial_nodes[p1].connections.add(p2)
                                self.mycelial_nodes[p2].connections.add(p1)

    async def cluster_parameters(self, parameter_gradients: Dict[str, np.ndarray]) -> Dict[str, MycelialCluster]:
        """
        Organiza parámetros en clusters basándose en gradientes y conexiones

        Args:
            parameter_gradients: Gradientes de parámetros

        Returns:
            Clusters actualizados
        """
        # Actualizar nutrientes basados en gradientes
        for param_name, gradient in parameter_gradients.items():
            if param_name in self.mycelial_nodes:
                node = self.mycelial_nodes[param_name]
                # Magnitud del gradiente como indicador de "nutrientes"
                nutrient_influx = np.linalg.norm(gradient)
                node.nutrient_level = 0.9 * node.nutrient_level + 0.1 * nutrient_influx
                node.last_update = time.time()

        # Actualizar puntuaciones de contribución
        await self._update_contribution_scores(parameter_gradients)

        # Reorganizar clusters si es necesario
        await self._reorganize_clusters()

        return self.clusters.copy()

    async def _update_contribution_scores(self, gradients: Dict[str, np.ndarray]):
        """Actualiza puntuaciones de contribución de parámetros"""
        total_nutrients = sum(node.nutrient_level for node in self.mycelial_nodes.values())

        if total_nutrients == 0:
            return

        for param_name, node in self.mycelial_nodes.items():
            # Contribución = nutrientes propios + influencia en conexiones
            own_contribution = node.nutrient_level / total_nutrients

            connection_contribution = 0.0
            if node.connections:
                connected_nutrients = sum(self.mycelial_nodes[conn].nutrient_level
                                        for conn in node.connections if conn in self.mycelial_nodes)
                connection_contribution = connected_nutrients / (len(node.connections) * total_nutrients)

            node.contribution_score = 0.7 * own_contribution + 0.3 * connection_contribution

            # Registrar en historial
            if param_name not in self.parameter_contributions:
                self.parameter_contributions[param_name] = []
            self.parameter_contributions[param_name].append(node.contribution_score)

            # Mantener historial limitado
            if len(self.parameter_contributions[param_name]) > 50:
                self.parameter_contributions[param_name] = self.parameter_contributions[param_name][-50:]

    async def _reorganize_clusters(self):
        """Reorganiza clusters basándose en patrones de contribución"""
        # Identificar parámetros con baja contribución
        low_contribution_params = [
            name for name, node in self.mycelial_nodes.items()
            if node.contribution_score < self.pruning_threshold
        ]

        # Podar parámetros poco útiles
        for param_name in low_contribution_params:
            # Remover conexiones
            node = self.mycelial_nodes[param_name]
            for connected_param in node.connections.copy():
                if connected_param in self.mycelial_nodes:
                    self.mycelial_nodes[connected_param].connections.discard(param_name)

            # Marcar para eliminación (no eliminar inmediatamente para estabilidad)
            node.nutrient_level *= 0.5  # reducir nutrientes gradualmente

        # Fomentar crecimiento de parámetros útiles
        high_contribution_params = [
            name for name, node in self.mycelial_nodes.items()
            if node.contribution_score > 0.8
        ]

        for param_name in high_contribution_params:
            node = self.mycelial_nodes[param_name]
            # Aumentar nutrientes y crear nuevas conexiones
            node.nutrient_level *= 1.2

            # Posiblemente crear nuevas conexiones con parámetros similares
            await self._grow_new_connections(param_name)

    async def _grow_new_connections(self, param_name: str):
        """Crea nuevas conexiones para parámetros de alto rendimiento"""
        if param_name not in self.mycelial_nodes:
            return

        node = self.mycelial_nodes[param_name]

        # Encontrar parámetros candidatos para conexión
        candidates = []
        for other_name, other_node in self.mycelial_nodes.items():
            if (other_name != param_name and
                other_name not in node.connections and
                other_node.contribution_score > 0.3):  # umbral mínimo

                # Calcular similitud (basada en contribución y nutrientes)
                similarity = (node.contribution_score * other_node.contribution_score +
                            0.5 * (node.nutrient_level + other_node.nutrient_level) / 2)
                candidates.append((other_name, similarity))

        # Ordenar por similitud y conectar los mejores
        candidates.sort(key=lambda x: x[1], reverse=True)
        connections_to_create = min(2, len(candidates))  # máximo 2 nuevas conexiones

        for other_name, _ in candidates[:connections_to_create]:
            node.connections.add(other_name)
            self.mycelial_nodes[other_name].connections.add(param_name)

    async def adapt_learning_rates(self, performance_feedback: float) -> Dict[str, float]:
        """
        Adapta tasas de aprendizaje basándose en rendimiento global

        Args:
            performance_feedback: Feedback de rendimiento (0-1)

        Returns:
            Nuevas tasas de aprendizaje por parámetro
        """
        new_rates = {}

        # Ajuste global basado en rendimiento
        if performance_feedback > 0.8:
            # Buen rendimiento: mantener o ligeramente aumentar
            global_adjustment = 1.05
        elif performance_feedback > 0.6:
            # Rendimiento aceptable: mantener
            global_adjustment = 1.0
        else:
            # Mal rendimiento: reducir
            global_adjustment = 0.9

        for param_name, node in self.mycelial_nodes.items():
            # Ajuste local basado en contribución
            local_adjustment = 0.8 + 0.4 * node.contribution_score  # 0.8 - 1.2

            # Combinar ajustes
            new_rate = node.learning_rate * global_adjustment * local_adjustment

            # Limitar rango
            new_rate = np.clip(new_rate, 1e-6, 1e-2)

            node.learning_rate = new_rate
            new_rates[param_name] = new_rate

        # Emitir evento de adaptación
        self.event_bus.emit(
            EventType.LEARNING_STEP,
            "MycelialOptimizer",
            outputs={
                "learning_rates_adapted": len(new_rates),
                "global_adjustment": global_adjustment,
                "performance_feedback": performance_feedback
            },
            explanation="Tasas de aprendizaje adaptadas"
        )

        return new_rates

    async def prune_unused_regions(self) -> Dict[str, bool]:
        """
        Poda regiones de parámetros poco utilizados

        Returns:
            Diccionario indicando qué parámetros fueron podados
        """
        pruned_params = {}

        # Identificar parámetros para podar
        for param_name, node in self.mycelial_nodes.items():
            should_prune = (
                node.contribution_score < self.pruning_threshold and
                node.nutrient_level < 0.2 and
                len(node.connections) < 2
            )

            if should_prune:
                # Marcar como inactivo (no eliminar completamente para posibles re-activaciones)
                node.nutrient_level = 0.0
                node.learning_rate *= 0.1  # reducir drásticamente
                pruned_params[param_name] = True

                # Emitir evento de poda
                self.event_bus.emit(
                    EventType.LEARNING_STEP,
                    "MycelialOptimizer",
                    outputs={"parameter_pruned": param_name, "contribution_score": node.contribution_score},
                    explanation="Parámetro podado por baja contribución"
                )
            else:
                pruned_params[param_name] = False

        pruned_count = sum(pruned_params.values())
        if pruned_count > 0:
            print(f"✂️ Podados {pruned_count} parámetros poco utilizados")

        return pruned_params

    def optimize_step(self, loss: float, parameter_gradients: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Realiza un paso de optimización micelial

        Args:
            loss: Valor de pérdida actual
            parameter_gradients: Gradientes de parámetros

        Returns:
            Información del paso de optimización
        """
        self.step_counter += 1

        # Actualizar estado basado en pérdida
        performance_feedback = max(0, 1.0 - loss)  # simplificación
        self.performance_history.append(performance_feedback)

        # Mantener historial limitado
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

        # Clustering de parámetros
        clusters = asyncio.run(self.cluster_parameters(parameter_gradients))

        # Adaptación de learning rates
        new_rates = asyncio.run(self.adapt_learning_rates(performance_feedback))

        # Poda periódica
        if self.step_counter % 50 == 0:
            asyncio.run(self.prune_unused_regions())

        # Calcular actualizaciones de parámetros
        parameter_updates = {}
        for param_name, gradient in parameter_gradients.items():
            if param_name in self.mycelial_nodes:
                node = self.mycelial_nodes[param_name]
                # Actualización micelial: gradiente * learning_rate * nutrient_factor
                nutrient_factor = 0.5 + node.nutrient_level  # 0.5 - 1.5
                update = -gradient * node.learning_rate * nutrient_factor
                parameter_updates[param_name] = update

        # Emitir evento de paso de optimización
        self.event_bus.emit(
            EventType.LEARNING_STEP,
            "MycelialOptimizer",
            outputs={
                "optimization_step": self.step_counter,
                "loss": loss,
                "performance_feedback": performance_feedback,
                "parameters_updated": len(parameter_updates)
            },
            explanation="Paso de optimización micelial completado"
        )

        return {
            "parameter_updates": parameter_updates,
            "learning_rates": new_rates,
            "clusters": clusters,
            "performance_feedback": performance_feedback,
            "mycelial_state": self.state.value
        }

    def get_mycelial_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual de la red micelial"""
        return {
            "state": self.state.value,
            "nodes_count": len(self.mycelial_nodes),
            "clusters_count": len(self.clusters),
            "total_connections": sum(len(node.connections) for node in self.mycelial_nodes.values()) // 2,
            "global_nutrient_level": self.global_nutrient_level,
            "step_counter": self.step_counter,
            "average_contribution": np.mean([n.contribution_score for n in self.mycelial_nodes.values()]),
            "performance_trend": np.mean(self.performance_history[-10:]) if self.performance_history else 0.0
        }

    def get_cluster_info(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene información detallada de clusters"""
        cluster_info = {}
        for name, cluster in self.clusters.items():
            cluster_info[name] = {
                "parameters_count": len(cluster.parameters),
                "performance_score": cluster.performance_score,
                "resource_allocation": cluster.resource_allocation,
                "growth_rate": cluster.growth_rate,
                "active_parameters": sum(1 for p in cluster.parameters
                                       if p in self.mycelial_nodes and
                                       self.mycelial_nodes[p].contribution_score > 0.2)
            }
        return cluster_info