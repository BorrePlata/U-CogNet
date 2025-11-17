"""
Advanced Semantic Feedback
=========================

Sophisticated scene analysis and natural language generation for tactical situations.
Provides contextual, temporal, and relational understanding of detected objects.
"""

from ucognet.core.interfaces import SemanticFeedback
from ucognet.core.types import Context, Detection
from typing import List, Dict
import time

class AdvancedSemanticFeedback(SemanticFeedback):
    """
    Advanced semantic feedback with threat assessment and contextual analysis.
    """

    def __init__(self):
        # Estado temporal para tracking
        self.scene_history = []
        self.last_analysis_time = time.time()

        # Umbrales de análisis
        self.confidence_threshold = 0.6

    def generate(self, context: Context, detections: List[Detection]) -> str:
        """
        Genera explicación semántica avanzada basada en análisis de amenazas.
        """
        current_time = time.time()

        # Filtrar detecciones por confianza
        high_confidence_detections = [
            d for d in detections if d.confidence >= self.confidence_threshold
        ]

        # Análisis de amenazas
        threat_analysis = self._assess_threat_level(high_confidence_detections)

        # Generar explicación narrativa
        explanation = self._generate_narrative_explanation(threat_analysis, high_confidence_detections)

        self.last_analysis_time = current_time
        return explanation

    def _assess_threat_level(self, detections: List[Detection]) -> Dict:
        """Evalúa el nivel de amenaza de la escena."""
        threat_score = 0.0
        threat_factors = []

        # Contar tipos de objetos
        military_objects = []
        weapons = []
        persons = []

        for detection in detections:
            class_name = detection.class_name.lower()
            if any(keyword in class_name for keyword in ['tank', 'military', 'vehicle', 'truck']):
                military_objects.append(detection)
                threat_score += 0.7  # Objetos militares son amenazas potenciales
                threat_factors.append(f"objeto militar: {detection.class_name}")
            elif any(keyword in class_name for keyword in ['gun', 'rifle', 'weapon', 'knife']):
                weapons.append(detection)
                threat_score += 0.8  # Armas son amenazas altas
                threat_factors.append(f"arma detectada: {detection.class_name}")
            elif class_name == 'person':
                persons.append(detection)
                threat_score += 0.2  # Personas pueden ser neutrales

        # Lógica adicional de amenaza
        if weapons and persons:
            threat_score += 0.5  # Personas con armas = amenaza alta
            threat_factors.append("persona armada detectada")

        if len(military_objects) >= 2:
            threat_score += 0.4  # Múltiples objetos militares
            threat_factors.append("actividad militar concentrada")

        # Normalizar y clasificar
        normalized_threat = min(threat_score / 2.0, 1.0)

        if normalized_threat > 0.7:
            level = 'CRITICAL'
        elif normalized_threat > 0.4:
            level = 'HIGH'
        elif normalized_threat > 0.2:
            level = 'MEDIUM'
        else:
            level = 'LOW'

        return {
            'level': level,
            'score': normalized_threat,
            'factors': threat_factors,
            'military_objects': len(military_objects),
            'weapons': len(weapons),
            'persons': len(persons)
        }

    def _generate_narrative_explanation(self, threat_analysis: Dict, detections: List[Detection]) -> str:
        """Genera una explicación narrativa del lenguaje."""
        components = []

        # Nivel de amenaza
        threat_level = threat_analysis['level']
        if threat_level != 'LOW':
            threat_emoji = {'CRITICAL': '🚨', 'HIGH': '⚠️', 'MEDIUM': '⚡'}.get(threat_level, 'ℹ️')
            components.append(f"{threat_emoji} NIVEL {threat_level}")

        # Descripción de escena
        military_count = threat_analysis['military_objects']
        weapon_count = threat_analysis['weapons']
        person_count = threat_analysis['persons']

        scene_descriptions = []
        if military_count > 0:
            scene_descriptions.append(f"objetos militares: {military_count}")
        if weapon_count > 0:
            scene_descriptions.append(f"armas: {weapon_count}")
        if person_count > 0:
            scene_descriptions.append(f"personas: {person_count}")

        if scene_descriptions:
            components.append(f"Actividad detectada: {', '.join(scene_descriptions)}")
        else:
            components.append("Zona despejada")

        # Factores de amenaza
        if threat_analysis['factors']:
            factors_text = " | ".join(threat_analysis['factors'][:2])  # Máximo 2 factores
            components.append(f"Factores: {factors_text}")

        # Recomendaciones
        if threat_level == 'CRITICAL':
            components.append("💡 Activar protocolos de respuesta inmediata")
        elif threat_level == 'HIGH':
            components.append("💡 Mantener vigilancia activa")
        elif threat_level == 'MEDIUM':
            components.append("💡 Monitoreo continuo recomendado")

        return " | ".join(components)
