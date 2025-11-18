"""
Generador de feedback semántico para U-CogNet.
Versión inicial: Reglas simbólicas simples.
"""

from typing import List, Dict, Any
from ..common.types import Detection, Context
from ..common.logging import logger

class SemanticFeedback:
    """
    Generador de explicaciones semánticas.
    Versión inicial: Reglas basadas en patrones de detección.
    """

    def __init__(self):
        self.rules = {
            'convoy': lambda dets: any(d.class_name in ['vehicle', 'tank'] for d in dets) and len(dets) >= 2,
            'crowd': lambda dets: sum(1 for d in dets if d.class_name == 'person') >= 3,
            'military_activity': lambda dets: any(d.class_name == 'tank' for d in dets),
            'empty_scene': lambda dets: len(dets) == 0
        }
        logger.info("SemanticFeedback inicializado con reglas simbólicas")

    def generate(self, context: Context, detections: List[Detection]) -> str:
        """
        Genera una explicación textual basada en contexto y detecciones.
        """
        # Aplicar reglas
        active_rules = []
        for rule_name, rule_func in self.rules.items():
            if rule_func(detections):
                active_rules.append(rule_name)

        if not active_rules:
            description = f"Escena con {len(detections)} objetos detectados"
        else:
            description = f"Escena identificada como: {', '.join(active_rules)}"

        # Añadir detalles
        if detections:
            classes = [d.class_name for d in detections]
            unique_classes = list(set(classes))
            description += f". Objetos: {', '.join(unique_classes)}"

        # Contexto temporal
        recent_count = len(context.recent_events)
        description += f". Eventos recientes: {recent_count}"

        logger.debug(f"Feedback generado: {description}")

        return description