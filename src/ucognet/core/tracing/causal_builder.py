"""
Causal Graph Builder
Construye grafos causales a partir de eventos cognitivos
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
import networkx as nx
from datetime import datetime, timedelta

from .cognitive_event import CognitiveEvent, EventType

class CausalLink:
    """Representa un enlace causal entre eventos"""

    def __init__(self, source_event: CognitiveEvent, target_event: CognitiveEvent,
                 link_type: str, confidence: float, explanation: str = None):
        self.source_event = source_event
        self.target_event = target_event
        self.link_type = link_type  # "causes", "enables", "prevents", "correlates"
        self.confidence = confidence
        self.explanation = explanation
        self.timestamp = datetime.now()

class CausalGraphBuilder:
    """
    Constructor de grafos causales a partir de secuencias de eventos.
    Identifica relaciones causales entre decisiones, acciones y resultados.
    """

    def __init__(self):
        self.causal_graph = nx.DiGraph()
        self.event_nodes: Dict[str, CognitiveEvent] = {}
        self.causal_links: List[CausalLink] = []

        # Reglas de causalidad por defecto
        self.causality_rules = {
            'temporal_proximity': self._rule_temporal_proximity,
            'trace_id_sharing': self._rule_trace_id_sharing,
            'decision_to_reward': self._rule_decision_to_reward,
            'gating_to_performance': self._rule_gating_to_performance,
            'learning_to_improvement': self._rule_learning_to_improvement
        }

    def add_event(self, event: CognitiveEvent) -> None:
        """Agrega un evento al grafo causal"""
        event_id = event.event_id

        # Agregar nodo al grafo
        self.event_nodes[event_id] = event
        self.causal_graph.add_node(event_id,
                                 event_type=event.event_type.value,
                                 source_module=event.source_module,
                                 timestamp=event.timestamp)

        # Intentar encontrar enlaces causales con eventos existentes
        self._find_causal_links(event)

    def add_events_batch(self, events: List[CognitiveEvent]) -> None:
        """Agrega múltiples eventos y construye enlaces causales"""
        # Ordenar por timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        for event in sorted_events:
            self.add_event(event)

    def _find_causal_links(self, new_event: CognitiveEvent) -> None:
        """Encuentra enlaces causales para un evento nuevo"""
        # Buscar en ventana de tiempo reciente (últimos 30 segundos)
        recent_cutoff = new_event.timestamp - timedelta(seconds=30)

        recent_events = [
            event for event in self.event_nodes.values()
            if event.timestamp >= recent_cutoff and event.event_id != new_event.event_id
        ]

        # Aplicar reglas de causalidad
        for rule_name, rule_func in self.causality_rules.items():
            links = rule_func(new_event, recent_events)
            self.causal_links.extend(links)

            # Agregar aristas al grafo
            for link in links:
                self.causal_graph.add_edge(
                    link.source_event.event_id,
                    link.target_event.event_id,
                    link_type=link.link_type,
                    confidence=link.confidence,
                    explanation=link.explanation
                )

    def _rule_temporal_proximity(self, new_event: CognitiveEvent,
                               recent_events: List[CognitiveEvent]) -> List[CausalLink]:
        """Regla: eventos cercanos en tiempo pueden estar relacionados"""
        links = []

        for event in recent_events:
            time_diff = (new_event.timestamp - event.timestamp).total_seconds()

            # Si están muy cerca (menos de 1 segundo), alta probabilidad de relación
            if time_diff < 1.0:
                confidence = 0.8
                link_type = "temporal_proximity"
                explanation = f"Eventos separados por {time_diff:.3f}s"

                links.append(CausalLink(event, new_event, link_type, confidence, explanation))

        return links

    def _rule_trace_id_sharing(self, new_event: CognitiveEvent,
                             recent_events: List[CognitiveEvent]) -> List[CausalLink]:
        """Regla: eventos con mismo trace_id están causalmente relacionados"""
        links = []

        if new_event.trace_id:
            for event in recent_events:
                if event.trace_id == new_event.trace_id:
                    confidence = 0.95
                    link_type = "trace_sharing"
                    explanation = f"Mismo trace_id: {new_event.trace_id}"

                    links.append(CausalLink(event, new_event, link_type, confidence, explanation))

        return links

    def _rule_decision_to_reward(self, new_event: CognitiveEvent,
                               recent_events: List[CognitiveEvent]) -> List[CausalLink]:
        """Regla: decisiones pueden causar rewards"""
        links = []

        if new_event.event_type == EventType.REWARD:
            # Buscar decisiones recientes
            decision_events = [
                e for e in recent_events
                if e.event_type == EventType.DECISION and e.episode_id == new_event.episode_id
            ]

            for decision in decision_events:
                confidence = 0.7
                link_type = "decision_to_reward"
                explanation = f"Decisión {decision.outputs.get('decision')} llevó a reward {new_event.outputs.get('reward')}"

                links.append(CausalLink(decision, new_event, link_type, confidence, explanation))

        return links

    def _rule_gating_to_performance(self, new_event: CognitiveEvent,
                                  recent_events: List[CognitiveEvent]) -> List[CausalLink]:
        """Regla: cambios de gating afectan performance"""
        links = []

        if new_event.event_type == EventType.EVALUATION_METRIC:
            # Buscar cambios de gating recientes
            gating_events = [
                e for e in recent_events
                if e.event_type == EventType.GATING_CHANGE
            ]

            for gating in gating_events:
                confidence = 0.6
                link_type = "gating_to_performance"
                gate_name = gating.outputs.get('gate_name')
                explanation = f"Cambio en gate '{gate_name}' afectó métricas de rendimiento"

                links.append(CausalLink(gating, new_event, link_type, confidence, explanation))

        return links

    def _rule_learning_to_improvement(self, new_event: CognitiveEvent,
                                    recent_events: List[CognitiveEvent]) -> List[CausalLink]:
        """Regla: eventos de aprendizaje pueden causar mejoras"""
        links = []

        if new_event.event_type == EventType.EVALUATION_METRIC:
            # Buscar eventos de aprendizaje recientes
            learning_events = [
                e for e in recent_events
                if e.event_type == EventType.LEARNING_STEP
            ]

            for learning in learning_events:
                # Verificar si las métricas mejoraron
                old_metrics = learning.metrics
                new_metrics = new_event.metrics

                # Comparar métricas clave (simplificado)
                improvement_detected = any(
                    new_metrics.get(key, 0) > old_metrics.get(key, 0)
                    for key in ['precision', 'recall', 'f1_score']
                )

                if improvement_detected:
                    confidence = 0.75
                    link_type = "learning_to_improvement"
                    explanation = "Aprendizaje resultó en mejora de métricas"

                    links.append(CausalLink(learning, new_event, link_type, confidence, explanation))

        return links

    def get_causal_chain(self, start_event_id: str, max_depth: int = 5) -> List[CognitiveEvent]:
        """Obtiene la cadena causal empezando desde un evento"""
        if start_event_id not in self.causal_graph:
            return []

        # Usar BFS para encontrar la cadena causal
        chain = []
        visited = set()
        queue = [(start_event_id, 0)]  # (event_id, depth)

        while queue and len(chain) < max_depth:
            current_id, depth = queue.pop(0)

            if current_id in visited:
                continue

            visited.add(current_id)
            chain.append(self.event_nodes[current_id])

            # Agregar sucesores (eventos que este causa)
            if depth < max_depth:
                for successor in self.causal_graph.successors(current_id):
                    queue.append((successor, depth + 1))

        return chain

    def get_root_causes(self, target_event_id: str, max_depth: int = 3) -> List[CognitiveEvent]:
        """Encuentra las causas raíz de un evento"""
        if target_event_id not in self.causal_graph:
            return []

        # Usar BFS reverso para encontrar causas raíz
        root_causes = []
        visited = set()
        queue = [(target_event_id, 0)]  # (event_id, depth)

        while queue:
            current_id, depth = queue.pop(0)

            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)

            predecessors = list(self.causal_graph.predecessors(current_id))

            if not predecessors:
                # Es causa raíz
                root_causes.append(self.event_nodes[current_id])
            else:
                # Continuar búsqueda
                for pred in predecessors:
                    queue.append((pred, depth + 1))

        return root_causes

    def analyze_episode(self, episode_id: str) -> Dict[str, Any]:
        """Analiza un episodio completo y sus relaciones causales"""
        episode_events = [
            event for event in self.event_nodes.values()
            if event.episode_id == episode_id
        ]

        if not episode_events:
            return {}

        # Construir subgrafo del episodio
        episode_nodes = {e.event_id for e in episode_events}
        episode_graph = self.causal_graph.subgraph(episode_nodes)

        # Análisis básico
        analysis = {
            'episode_id': episode_id,
            'total_events': len(episode_events),
            'causal_links': len(episode_graph.edges()),
            'event_types': {},
            'critical_path': [],
            'root_causes': [],
            'key_decisions': []
        }

        # Contar tipos de eventos
        for event in episode_events:
            event_type = event.event_type.value
            analysis['event_types'][event_type] = analysis['event_types'].get(event_type, 0) + 1

        # Encontrar camino crítico (simplificado: camino más largo)
        try:
            longest_path = nx.dag_longest_path(episode_graph)
            analysis['critical_path'] = [self.event_nodes[eid] for eid in longest_path]
        except:
            analysis['critical_path'] = []

        # Encontrar decisiones clave
        decisions = [e for e in episode_events if e.event_type == EventType.DECISION]
        analysis['key_decisions'] = decisions[:5]  # Top 5 decisiones

        return analysis

    def get_graph_summary(self) -> Dict[str, Any]:
        """Obtiene resumen del grafo causal"""
        return {
            'total_nodes': len(self.causal_graph.nodes()),
            'total_edges': len(self.causal_graph.edges()),
            'causal_links_count': len(self.causal_links),
            'connected_components': nx.number_weakly_connected_components(self.causal_graph),
            'average_degree': sum(dict(self.causal_graph.degree()).values()) / len(self.causal_graph) if self.causal_graph else 0
        }

    def export_graph(self, filepath: str, format: str = 'graphml') -> None:
        """Exporta el grafo causal a archivo"""
        if format == 'graphml':
            nx.write_graphml(self.causal_graph, filepath)
        elif format == 'json':
            # Exportar como JSON serializable
            graph_data = {
                'nodes': [
                    {
                        'id': node_id,
                        'event_type': self.causal_graph.nodes[node_id]['event_type'],
                        'source_module': self.causal_graph.nodes[node_id]['source_module'],
                        'timestamp': self.causal_graph.nodes[node_id]['timestamp'].isoformat()
                    }
                    for node_id in self.causal_graph.nodes()
                ],
                'edges': [
                    {
                        'source': source,
                        'target': target,
                        'link_type': self.causal_graph.edges[source, target]['link_type'],
                        'confidence': self.causal_graph.edges[source, target]['confidence']
                    }
                    for source, target in self.causal_graph.edges()
                ]
            }

            import json
            with open(filepath, 'w') as f:
                json.dump(graph_data, f, indent=2)