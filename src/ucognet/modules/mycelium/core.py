# /mnt/c/Users/desar/Documents/Science/UCogNet/src/ucognet/modules/mycelium/core.py
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from .types import MycoContext, MycoSignal, MycoPath, MycoMetrics

logger = logging.getLogger(__name__)

class MycoNode:
    """
    Nodo en la red micelial que representa un módulo cognitivo.

    Cada nodo mantiene estado local, capacidades y métricas de rendimiento.
    """

    def __init__(self, name: str, capabilities: Dict[str, float],
                 max_load: float = 1.0, recovery_rate: float = 0.1):
        self.name = name
        self.capabilities = capabilities  # ej: {"vision": 0.9, "audio": 0.1}
        self.max_load = max_load
        self.recovery_rate = recovery_rate

        # Estado dinámico
        self.current_load = 0.0
        self.activation_history: List[float] = []
        self.performance_metrics: Dict[str, float] = {}
        self.last_activation = 0.0
        self.adaptation_score = 0.5  # capacidad de adaptación [0,1]

    def update_state(self, context: MycoContext, activation_level: float = 0.0):
        """Actualizar estado del nodo basado en contexto y activación"""
        self.current_load = min(self.max_load, self.current_load + activation_level)

        # Recuperación natural
        self.current_load = max(0, self.current_load - self.recovery_rate)

        # Registrar activación
        self.activation_history.append(activation_level)
        if len(self.activation_history) > 100:  # mantener historial limitado
            self.activation_history.pop(0)

        self.last_activation = context.timestamp

        # Actualizar métricas de rendimiento
        if context.metrics:
            for key, value in context.metrics.items():
                if key in self.capabilities:
                    self.performance_metrics[key] = value

    def get_effective_capability(self, task_type: str) -> float:
        """Obtener capacidad efectiva considerando carga actual"""
        base_capability = self.capabilities.get(task_type, 0.0)
        load_penalty = self.current_load / self.max_load
        return base_capability * (1.0 - load_penalty * 0.5)

    def can_handle_task(self, task_type: str, required_capability: float = 0.1) -> bool:
        """Verificar si el nodo puede manejar un tipo de tarea"""
        return self.get_effective_capability(task_type) >= required_capability

    def get_health_score(self) -> float:
        """Obtener puntuación de salud del nodo [0,1]"""
        # Basado en carga, historial de activación, y métricas
        load_factor = 1.0 - (self.current_load / self.max_load)
        activation_stability = 1.0 - np.std(self.activation_history[-20:]) if self.activation_history else 1.0
        return (load_factor + activation_stability) / 2.0


class MycoEdge:
    """
    Conexión entre nodos con feromonas y pesos dinámicos.

    Las aristas evolucionan basado en experiencia y seguridad.
    """

    def __init__(self, src: str, dst: str, base_weight: float = 1.0,
                 evaporation_rate: float = 0.01, max_pheromone: float = 10.0):
        self.src = src
        self.dst = dst
        self.base_weight = base_weight
        self.evaporation_rate = evaporation_rate
        self.max_pheromone = max_pheromone

        # Estado dinámico
        self.pheromone = 0.0
        self.last_update = 0.0
        self.usage_count = 0
        self.success_rate = 0.5
        self.safety_history: List[float] = []

    def effective_weight(self, safety_score: float = 1.0) -> float:
        """Calcular peso efectivo considerando feromonas y seguridad"""
        pheromone_contribution = min(self.pheromone, self.max_pheromone)
        safety_modulation = safety_score * (1.0 + self.success_rate)
        return self.base_weight + pheromone_contribution * safety_modulation

    def update_pheromone(self, reward: float, safety_score: float, timestamp: float):
        """Actualizar nivel de feromona basado en recompensa"""
        # Solo reforzar si es seguro
        if safety_score > 0.1:
            delta = reward * safety_score * (1.0 + self.success_rate)
            self.pheromone = min(self.max_pheromone, self.pheromone + delta)

        self.last_update = timestamp
        self.usage_count += 1

        # Actualizar tasa de éxito
        if reward > 0:
            self.success_rate = 0.9 * self.success_rate + 0.1 * 1.0
        else:
            self.success_rate = 0.9 * self.success_rate + 0.1 * 0.0

        # Registrar seguridad
        self.safety_history.append(safety_score)
        if len(self.safety_history) > 50:
            self.safety_history.pop(0)

    def evaporate(self, dt: float):
        """Evaporar feromonas con el tiempo"""
        evaporation = self.pheromone * self.evaporation_rate * dt
        self.pheromone = max(0, self.pheromone - evaporation)

    def get_reliability_score(self) -> float:
        """Obtener puntuación de confiabilidad de la conexión"""
        safety_avg = np.mean(self.safety_history) if self.safety_history else 0.5
        return (self.success_rate + safety_avg) / 2.0


class MycoNet:
    """
    Red micelial principal que coordina módulos cognitivos.

    Gestiona rutas, recursos, atención y evolución controlada del sistema.
    """

    def __init__(self, safety_module=None, exploration_rate: float = 0.1):
        self.nodes: Dict[str, MycoNode] = {}
        self.edges: Dict[Tuple[str, str], MycoEdge] = {}
        self.safety_module = safety_module
        self.exploration_rate = exploration_rate

        # Estado global
        self.global_context: Optional[MycoContext] = None
        self.active_paths: List[MycoPath] = []
        self.emergent_behaviors: List[str] = []
        self.performance_history: List[MycoMetrics] = []

        logger.info("MycoNet initialized with safety integration")

    def register_node(self, node: MycoNode):
        """Registrar un nuevo nodo en la red"""
        self.nodes[node.name] = node
        logger.info(f"Registered node: {node.name} with capabilities: {node.capabilities}")

    def connect(self, src: str, dst: str, base_weight: float = 1.0):
        """Crear conexión entre nodos"""
        if src not in self.nodes or dst not in self.nodes:
            logger.warning(f"Cannot connect {src} -> {dst}: nodes not found")
            return

        edge = MycoEdge(src, dst, base_weight)
        self.edges[(src, dst)] = edge
        logger.info(f"Connected {src} -> {dst} with weight {base_weight}")

    def route(self, context: MycoContext, max_depth: int = 5) -> MycoPath:
        """
        Calcular ruta óptima a través de la red micelial.

        Considera seguridad, capacidades, carga y feromonas.
        """
        self.global_context = context

        # Comenzar desde nodo de entrada
        start_node = self._find_start_node(context)
        if not start_node:
            return self._create_empty_path(context)

        path = self._find_optimal_path(start_node, context, max_depth)
        return path

    def _find_start_node(self, context: MycoContext) -> Optional[str]:
        """Encontrar nodo inicial apropiado para el contexto"""
        # Buscar nodos de entrada o con alta capacidad para la tarea
        candidates = []
        for name, node in self.nodes.items():
            if "input" in name.lower() or "gateway" in name.lower():
                candidates.append((name, 1.0))
            else:
                task_capability = max(node.capabilities.values()) if node.capabilities else 0.0
                candidates.append((name, task_capability))

        if not candidates:
            return None

        # Seleccionar basado en capacidad y exploración
        if np.random.rand() < self.exploration_rate:
            return np.random.choice([c[0] for c in candidates])

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _find_optimal_path(self, start: str, context: MycoContext, max_depth: int) -> MycoPath:
        """Encontrar camino óptimo usando algoritmo de colonia de hormigas modificado"""
        current = start
        path_nodes = [current]
        path_edges = []
        total_weight = 0.0
        total_safety = 1.0

        for depth in range(max_depth):
            candidates = self._get_candidate_edges(current, context)

            if not candidates:
                break

            # Seleccionar siguiente nodo
            next_node, edge_weight, safety_score = self._select_next_node(candidates, context)

            if not next_node:
                break

            # Actualizar camino
            path_nodes.append(next_node)
            path_edges.append((current, next_node))
            total_weight += edge_weight
            total_safety *= safety_score

            current = next_node

            # Criterios de parada
            if self._should_stop_path(current, context):
                break

        expected_reward = self._estimate_path_reward(path_nodes, context)

        return MycoPath(
            nodes=path_nodes,
            edges=path_edges,
            total_weight=total_weight,
            safety_score=total_safety,
            expected_reward=expected_reward,
            context=context
        )

    def _get_candidate_edges(self, current: str, context: MycoContext) -> List[Tuple[str, MycoEdge, float]]:
        """Obtener aristas candidatas desde el nodo actual"""
        candidates = []

        for (src, dst), edge in self.edges.items():
            if src != current:
                continue

            # Verificar que el nodo destino existe y puede manejar la tarea
            if dst not in self.nodes:
                continue

            target_node = self.nodes[dst]
            task_type = context.task_id.split('_')[0]  # simplificación
            if not target_node.can_handle_task(task_type):
                continue

            # Obtener puntuación de seguridad
            safety_score = 1.0
            if self.safety_module:
                safety_score = self.safety_module.evaluate_transition(src, dst, context)

            if safety_score <= 0.1:  # umbral mínimo de seguridad
                continue

            candidates.append((dst, edge, safety_score))

        return candidates

    def _select_next_node(self, candidates: List[Tuple[str, MycoEdge, float]],
                         context: MycoContext) -> Tuple[Optional[str], float, float]:
        """Seleccionar siguiente nodo basado en pesos y exploración"""
        if not candidates:
            return None, 0.0, 0.0

        # Calcular probabilidades
        weights = []
        for dst, edge, safety in candidates:
            edge_weight = edge.effective_weight(safety)
            node_health = self.nodes[dst].get_health_score()
            combined_weight = edge_weight * node_health * safety
            weights.append(combined_weight)

        # Exploración vs explotación
        if np.random.rand() < self.exploration_rate:
            # Exploración: selección aleatoria ponderada
            total_weight = sum(weights)
            if total_weight > 0:
                probs = [w / total_weight for w in weights]
                idx = np.random.choice(len(candidates), p=probs)
            else:
                idx = np.random.choice(len(candidates))
        else:
            # Explotación: seleccionar el mejor
            idx = np.argmax(weights)

        dst, edge, safety = candidates[idx]
        return dst, edge.effective_weight(safety), safety

    def _should_stop_path(self, current: str, context: MycoContext) -> bool:
        """Determinar si el camino debería detenerse"""
        # Criterios de parada
        if "output" in current.lower() or "final" in current.lower():
            return True

        if len(self.active_paths) > 10:  # límite de caminos activos
            return True

        # Verificar si hemos alcanzado un nodo de alto nivel
        node = self.nodes.get(current)
        if node and any(cap > 0.8 for cap in node.capabilities.values()):
            return True

        return False

    def _estimate_path_reward(self, path_nodes: List[str], context: MycoContext) -> float:
        """Estimar recompensa esperada del camino"""
        if not path_nodes:
            return 0.0

        # Basado en capacidades de los nodos y métricas del contexto
        total_capability = 0.0
        for node_name in path_nodes:
            node = self.nodes.get(node_name)
            if node:
                task_type = context.task_id.split('_')[0]
                total_capability += node.get_effective_capability(task_type)

        # Modulado por métricas del contexto
        reward_modifier = 1.0
        if context.metrics:
            reward_modifier = np.mean(list(context.metrics.values()))

        return total_capability * reward_modifier / len(path_nodes)

    def _create_empty_path(self, context: MycoContext) -> MycoPath:
        """Crear camino vacío para casos de error"""
        return MycoPath(
            nodes=[],
            edges=[],
            total_weight=0.0,
            safety_score=0.0,
            expected_reward=0.0,
            context=context
        )

    def reinforce_path(self, path: MycoPath, actual_reward: float):
        """
        Reforzar feromonas en el camino basado en recompensa real.

        Solo refuerza caminos seguros y exitosos.
        """
        if not path.edges or path.safety_score <= 0.1:
            return

        # Reforzar cada arista del camino
        for src, dst in path.edges:
            edge = self.edges.get((src, dst))
            if edge:
                edge.update_pheromone(actual_reward, path.safety_score, path.context.timestamp)

        # Actualizar estado de nodos
        for node_name in path.nodes:
            node = self.nodes.get(node_name)
            if node:
                activation_level = actual_reward * path.safety_score
                node.update_state(path.context, activation_level)

        logger.info(f"Reinforced path with reward {actual_reward:.3f}, safety {path.safety_score:.3f}")

    def evaporate(self, dt: float = 1.0):
        """Evaporar feromonas en todas las aristas"""
        for edge in self.edges.values():
            edge.evaporate(dt)

    def get_system_metrics(self) -> MycoMetrics:
        """Obtener métricas globales del sistema micelial"""
        if not self.edges:
            return MycoMetrics(0, 0, 0, {}, [], 0, 0)

        # Calcular eficiencia de caminos
        path_efficiency = np.mean([p.total_weight / len(p.nodes) if p.nodes else 0
                                  for p in self.active_paths]) if self.active_paths else 0

        # Cumplimiento de seguridad
        safety_scores = [p.safety_score for p in self.active_paths]
        safety_compliance = np.mean(safety_scores) if safety_scores else 1.0

        # Tasa de adaptación
        adaptation_rate = np.mean([n.adaptation_score for n in self.nodes.values()])

        # Distribución de recursos
        resource_distribution = {}
        for name, node in self.nodes.items():
            resource_distribution[name] = node.current_load / node.max_load

        # Entropía de feromonas
        pheromones = [e.pheromone for e in self.edges.values()]
        pheromone_entropy = -np.sum([p * np.log(p + 1e-10) for p in pheromones]) if pheromones else 0

        # Velocidad de convergencia (basada en cambios recientes)
        convergence_speed = 0.5  # placeholder

        return MycoMetrics(
            path_efficiency=path_efficiency,
            safety_compliance=safety_compliance,
            adaptation_rate=adaptation_rate,
            resource_distribution=resource_distribution,
            emergent_behaviors=self.emergent_behaviors.copy(),
            pheromone_entropy=pheromone_entropy,
            convergence_speed=convergence_speed
        )

    def detect_emergent_behaviors(self):
        """Detectar comportamientos emergentes en la red"""
        # Análisis de patrones en caminos activos
        if len(self.active_paths) < 3:
            return

        # Buscar caminos recurrentes
        path_patterns = {}
        for path in self.active_paths[-10:]:  # últimos 10 caminos
            pattern = tuple(path.nodes)
            path_patterns[pattern] = path_patterns.get(pattern, 0) + 1

        # Identificar patrones emergentes
        for pattern, count in path_patterns.items():
            if count >= 3:  # patrón recurrente
                behavior_name = f"pattern_{len(pattern)}nodes_{count}occurrences"
                if behavior_name not in self.emergent_behaviors:
                    self.emergent_behaviors.append(behavior_name)
                    logger.info(f"Emergent behavior detected: {behavior_name}")

    def optimize_topology(self):
        """Optimizar topología de la red basado en uso y rendimiento"""
        # Identificar aristas poco usadas
        unused_edges = []
        for (src, dst), edge in self.edges.items():
            if edge.usage_count < 5:  # umbral arbitrario
                unused_edges.append((src, dst))

        # Podar conexiones poco útiles (con cuidado)
        for src, dst in unused_edges[:2]:  # máximo 2 por optimización
            if (src, dst) in self.edges:
                del self.edges[(src, dst)]
                logger.info(f"Pruned unused connection: {src} -> {dst}")

        # Sugerir nuevas conexiones basadas en capacidades complementarias
        self._suggest_new_connections()

    def _suggest_new_connections(self):
        """Sugerir nuevas conexiones entre nodos complementarios"""
        suggestions = []

        for name1, node1 in self.nodes.items():
            for name2, node2 in self.nodes.items():
                if name1 == name2 or (name1, name2) in self.edges:
                    continue

                # Calcular complementariedad
                overlap = 0
                total = 0
                for cap in set(node1.capabilities.keys()) | set(node2.capabilities.keys()):
                    val1 = node1.capabilities.get(cap, 0)
                    val2 = node2.capabilities.get(cap, 0)
                    overlap += min(val1, val2)
                    total += max(val1, val2)

                if total > 0:
                    complementarity = 1.0 - (overlap / total)
                    if complementarity > 0.7:  # alta complementariedad
                        suggestions.append((name1, name2, complementarity))

        # Crear conexiones sugeridas (limitado)
        for src, dst, comp in suggestions[:3]:  # máximo 3 nuevas conexiones
            self.connect(src, dst, base_weight=comp)
            logger.info(f"Suggested new connection: {src} -> {dst} (complementarity: {comp:.2f})")