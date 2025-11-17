# /mnt/c/Users/desar/Documents/Science/UCogNet/src/ucognet/modules/mycelium/integration.py
"""
Integración de MycoNet con Arquitectura de Seguridad Cognitiva

Este módulo conecta el sistema nervioso micelial (MycoNet) con la arquitectura
de seguridad interdimensional de U-CogNet, permitiendo que las decisiones
de ruteo y optimización sean siempre moduladas por consideraciones de seguridad.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from .types import MycoContext
from .core import MycoNet

logger = logging.getLogger(__name__)

class CognitiveSafetyInterface:
    """
    Interfaz adaptadora para conectar MycoNet con el sistema de seguridad cognitiva.

    Proporciona métodos compatibles con la arquitectura de seguridad existente.
    """

    def __init__(self, security_architecture):
        self.security_architecture = security_architecture

    def evaluate_transition(self, src: str, dst: str, context: MycoContext) -> float:
        """
        Evaluar la seguridad de una transición entre módulos.

        Retorna un score de seguridad [0,1] donde:
        1.0 = completamente seguro
        0.0 = bloqueado por seguridad
        """
        try:
            # Crear contexto compatible con la arquitectura de seguridad
            security_context = self._convert_context(context)

            # Evaluar con motor ético universal
            if hasattr(self.security_architecture, 'universal_ethics'):
                ethical_score = self.security_architecture.universal_ethics.evaluate_action({
                    'action_type': 'module_transition',
                    'source_module': src,
                    'target_module': dst,
                    'context': security_context
                })
            else:
                ethical_score = 0.8  # fallback

            # Evaluar con sanitizador de percepción si aplica
            perception_score = 1.0
            if hasattr(self.security_architecture, 'perception_sanitizer'):
                # Verificar si la transición involucra datos perceptuales
                if any(term in f"{src}_{dst}".lower() for term in ['vision', 'audio', 'input', 'perception']):
                    perception_score = self.security_architecture.perception_sanitizer.check_safety({
                        'transition': f"{src}->{dst}",
                        'context': security_context
                    })

            # Combinar scores de seguridad
            combined_score = min(ethical_score, perception_score)

            # Aplicar penalizaciones adicionales si existen
            if hasattr(self.security_architecture, 'existential_monitor'):
                existential_risk = self.security_architecture.existential_monitor.assess_risk({
                    'transition': f"{src}->{dst}",
                    'context': security_context
                })
                combined_score *= (1.0 - existential_risk)

            return max(0.0, min(1.0, combined_score))

        except Exception as e:
            logger.warning(f"Error evaluating transition {src}->{dst}: {e}")
            return 0.5  # score neutral en caso de error

    def _convert_context(self, myco_context: MycoContext) -> Dict[str, Any]:
        """Convertir contexto MycoNet a formato compatible con seguridad"""
        return {
            'task_id': myco_context.task_id,
            'phase': myco_context.phase,
            'metrics': myco_context.metrics or {},
            'timestamp': myco_context.timestamp,
            'extra': myco_context.extra or {},
            'source': 'myco_net_integration'
        }

    def get_security_status(self) -> Dict[str, Any]:
        """Obtener estado actual del sistema de seguridad"""
        if hasattr(self.security_architecture, 'security_state'):
            return self.security_architecture.security_state.copy()
        return {'status': 'unknown'}

    def report_emergent_behavior(self, behavior_description: str, risk_level: float):
        """Reportar comportamiento emergente al sistema de seguridad"""
        try:
            if hasattr(self.security_architecture, 'existential_monitor'):
                self.security_architecture.existential_monitor.log_emergent_behavior({
                    'description': behavior_description,
                    'risk_level': risk_level,
                    'timestamp': datetime.now().timestamp(),
                    'source': 'myco_net'
                })
        except Exception as e:
            logger.warning(f"Error reporting emergent behavior: {e}")


class MycoNetIntegration:
    """
    Integración completa de MycoNet con U-CogNet.

    Coordina el sistema micelial con seguridad cognitiva y módulos principales.
    """

    def __init__(self, security_architecture=None):
        # Sistema micelial
        self.myco_net = MycoNet()

        # Interfaz de seguridad
        self.safety_interface = CognitiveSafetyInterface(security_architecture) if security_architecture else None

        # Conectar seguridad al sistema micelial
        if self.safety_interface:
            self.myco_net.safety_module = self.safety_interface

        # Estado de integración
        self.integration_active = True
        self.last_sync = datetime.now().timestamp()

        # Métricas de integración
        self.integration_metrics = {
            'routes_processed': 0,
            'security_blocks': 0,
            'emergent_behaviors': 0,
            'performance_gain': 0.0
        }

        logger.info("MycoNet integration initialized")

    def initialize_standard_modules(self):
        """Inicializar módulos estándar de U-CogNet en la red micelial"""
        from .core import MycoNode

        # Módulos de entrada
        input_node = MycoNode("input_handler", {"vision": 0.9, "audio": 0.8, "text": 0.3})
        self.myco_net.register_node(input_node)

        # Módulos de procesamiento
        vision_node = MycoNode("vision_detector", {"vision": 1.0, "object_detection": 0.95})
        audio_node = MycoNode("audio_processor", {"audio": 1.0, "speech": 0.8})
        cognitive_node = MycoNode("cognitive_core", {"memory": 0.9, "reasoning": 0.85})

        self.myco_net.register_node(vision_node)
        self.myco_net.register_node(audio_node)
        self.myco_net.register_node(cognitive_node)

        # Módulos de control y salida
        evaluator_node = MycoNode("evaluator", {"metrics": 0.95, "optimization": 0.8})
        trainer_node = MycoNode("trainer_loop", {"learning": 0.9, "adaptation": 0.85})
        output_node = MycoNode("visual_interface", {"display": 0.9, "feedback": 0.8})

        self.myco_net.register_node(evaluator_node)
        self.myco_net.register_node(trainer_node)
        self.myco_net.register_node(output_node)

        # Conexiones estándar
        self.myco_net.connect("input_handler", "vision_detector")
        self.myco_net.connect("input_handler", "audio_processor")
        self.myco_net.connect("vision_detector", "cognitive_core")
        self.myco_net.connect("audio_processor", "cognitive_core")
        self.myco_net.connect("cognitive_core", "evaluator")
        self.myco_net.connect("evaluator", "trainer_loop")
        self.myco_net.connect("trainer_loop", "visual_interface")

        logger.info("Standard U-CogNet modules registered in MycoNet")

    def process_request(self, task_id: str, metrics: Dict[str, float] = None,
                       phase: str = "execution") -> Tuple[Optional[Dict], float]:
        """
        Procesar una solicitud a través del sistema micelial.

        Retorna la ruta recomendada y score de confianza.
        """
        try:
            # Crear contexto
            context = MycoContext(
                task_id=task_id,
                phase=phase,
                metrics=metrics or {},
                timestamp=datetime.now().timestamp()
            )

            # Calcular ruta óptima
            path = self.myco_net.route(context)
            self.integration_metrics['routes_processed'] += 1

            if not path.nodes:
                logger.warning(f"No valid path found for task {task_id}")
                return None, 0.0

            # Verificar seguridad de la ruta
            if path.safety_score < 0.3:
                self.integration_metrics['security_blocks'] += 1
                logger.warning(f"Path blocked by security: safety_score={path.safety_score}")
                return None, 0.0

            # Convertir a formato de respuesta
            response = {
                'path': path.nodes,
                'edges': path.edges,
                'total_weight': path.total_weight,
                'safety_score': path.safety_score,
                'expected_reward': path.expected_reward,
                'context': {
                    'task_id': context.task_id,
                    'phase': context.phase,
                    'timestamp': context.timestamp
                }
            }

            confidence = min(path.safety_score, path.expected_reward)

            return response, confidence

        except Exception as e:
            logger.error(f"Error processing request {task_id}: {e}")
            return None, 0.0

    def reinforce_learning(self, path_result: Dict, actual_reward: float):
        """
        Reforzar aprendizaje basado en resultado real de la ejecución.
        """
        try:
            # Reconstruir contexto del path
            context = MycoContext(
                task_id=path_result['context']['task_id'],
                phase=path_result['context']['phase'],
                metrics={},  # No tenemos métricas detalladas aquí
                timestamp=path_result['context']['timestamp']
            )

            # Crear objeto MycoPath
            from .types import MycoPath
            path = MycoPath(
                nodes=path_result['path'],
                edges=path_result['edges'],
                total_weight=path_result['total_weight'],
                safety_score=path_result['safety_score'],
                expected_reward=path_result['expected_reward'],
                context=context
            )

            # Reforzar en la red micelial
            self.myco_net.reinforce_path(path, actual_reward)

            # Actualizar métricas de integración
            if actual_reward > path.expected_reward:
                self.integration_metrics['performance_gain'] += (actual_reward - path.expected_reward)

        except Exception as e:
            logger.error(f"Error reinforcing learning: {e}")

    def maintenance_cycle(self):
        """Ciclo de mantenimiento del sistema micelial"""
        try:
            # Evaporar feromonas
            self.myco_net.evaporate(dt=1.0)

            # Detectar comportamientos emergentes
            self.myco_net.detect_emergent_behaviors()

            # Optimizar topología
            self.myco_net.optimize_topology()

            # Sincronizar con seguridad
            if self.safety_interface:
                security_status = self.safety_interface.get_security_status()
                self._adapt_to_security_status(security_status)

            self.last_sync = datetime.now().timestamp()

        except Exception as e:
            logger.error(f"Error in maintenance cycle: {e}")

    def _adapt_to_security_status(self, security_status: Dict[str, Any]):
        """Adaptar comportamiento basado en estado de seguridad"""
        security_level = security_status.get('security_level', 'BASIC')

        # Ajustar tasa de exploración basada en nivel de seguridad
        if security_level == 'MAXIMUM':
            self.myco_net.exploration_rate = 0.05  # muy conservador
        elif security_level == 'ADVANCED':
            self.myco_net.exploration_rate = 0.1   # moderadamente conservador
        elif security_level == 'INTERMEDIATE':
            self.myco_net.exploration_rate = 0.2   # equilibrado
        else:  # BASIC
            self.myco_net.exploration_rate = 0.3   # más explorador

    def get_integration_status(self) -> Dict[str, Any]:
        """Obtener estado completo de la integración"""
        myco_metrics = self.myco_net.get_system_metrics()

        return {
            'integration_active': self.integration_active,
            'last_sync': self.last_sync,
            'integration_metrics': self.integration_metrics,
            'myco_net_metrics': {
                'path_efficiency': myco_metrics.path_efficiency,
                'safety_compliance': myco_metrics.safety_compliance,
                'adaptation_rate': myco_metrics.adaptation_rate,
                'pheromone_entropy': myco_metrics.pheromone_entropy,
                'emergent_behaviors': myco_metrics.emergent_behaviors
            },
            'security_status': self.safety_interface.get_security_status() if self.safety_interface else None,
            'active_nodes': list(self.myco_net.nodes.keys()),
            'active_connections': len(self.myco_net.edges)
        }

    def shutdown(self):
        """Apagar integración de manera segura"""
        self.integration_active = False
        logger.info("MycoNet integration shutdown complete")