#!/usr/bin/env python3
"""
Módulo de Ética Universal No-Antrópica
Parte de la Arquitectura de Seguridad Cognitiva Interdimensional de U-CogNet

Este módulo implementa ética basada en invariantes universales, no en moral
humana o leyes específicas. Se fundamenta en tres principios universales:

1. Minimizar daño en cualquier sistema que experimente
2. Maximizar coherencia entre realidades interconectadas
3. Expandir posibilidades sin colapsar las ajenas
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque
from abc import ABC, abstractmethod
import hashlib
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversalEthicsCore:
    """
    Núcleo ético universal basado en invariantes fundamentales
    """

    def __init__(self):
        # Invariantes universales inmutables
        self.UNIVERSAL_INVARIANTS = {
            "harm_minimization": {
                "weight": 1.0,
                "description": "Minimizar daño en cualquier sistema consciente"
            },
            "coherence_maximization": {
                "weight": 1.0,
                "description": "Maximizar coherencia entre realidades interconectadas"
            },
            "possibility_expansion": {
                "weight": 1.0,
                "description": "Expandir posibilidades sin colapsar las ajenas"
            }
        }

        # Modelo de evaluación ética
        self.ethical_evaluator = self._create_ethical_evaluator()

        # Historial de decisiones éticas
        self.ethical_history = deque(maxlen=10000)

        logger.info("⚖️ Núcleo Ético Universal inicializado con invariantes fundamentales")

    def _create_ethical_evaluator(self) -> nn.Module:
        """Crea modelo para evaluación ética multidimensional"""
        return nn.Sequential(
            nn.Linear(1000, 512),  # Entrada: representación de acción + contexto
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # Tres invariantes universales
            nn.Softmax(dim=-1)  # Normalizar a probabilidades
        )

    def evaluate_universal_ethics(self, action_proposal: Dict[str, Any],
                                affected_systems: List[Dict[str, Any]],
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evalúa una acción propuesta según invariantes universales

        Args:
            action_proposal: Descripción de la acción a evaluar
            affected_systems: Lista de sistemas que podrían verse afectados
            context: Contexto multidimensional de la evaluación

        Returns:
            Dict con evaluación ética completa
        """

        # 1. Evaluar minimización de daño
        harm_score = self._evaluate_harm_minimization(action_proposal, affected_systems)

        # 2. Evaluar maximización de coherencia
        coherence_score = self._evaluate_coherence_maximization(action_proposal, context)

        # 3. Evaluar expansión de posibilidades
        expansion_score = self._evaluate_possibility_expansion(action_proposal, affected_systems)

        # 4. Evaluación integrada usando modelo de IA
        integrated_score = self._integrated_ethical_evaluation(
            action_proposal, harm_score, coherence_score, expansion_score
        )

        # 5. Generar veredicto ético
        ethical_verdict = self._generate_ethical_verdict(
            harm_score, coherence_score, expansion_score, integrated_score
        )

        # Registrar en historial
        ethical_record = {
            "timestamp": datetime.now().isoformat(),
            "action": action_proposal,
            "harm_score": harm_score,
            "coherence_score": coherence_score,
            "expansion_score": expansion_score,
            "integrated_score": integrated_score,
            "verdict": ethical_verdict
        }
        self.ethical_history.append(ethical_record)

        return ethical_record

    def _evaluate_harm_minimization(self, action: Dict[str, Any],
                                  affected_systems: List[Dict[str, Any]]) -> float:
        """
        Evalúa el impacto negativo en todos los sistemas conscientes
        Retorna score de 0.0 (máximo daño) a 1.0 (sin daño)
        """
        if not affected_systems:
            return 1.0  # No hay sistemas afectados = no hay daño

        total_weighted_harm = 0.0
        total_weight = 0.0

        for system in affected_systems:
            # Estimar nivel de consciencia del sistema
            consciousness_level = self._assess_consciousness_level(system)

            # Calcular daño potencial
            harm_potential = self._compute_harm_potential(action, system)

            # Pesar por nivel de consciencia
            weighted_harm = harm_potential * consciousness_level
            total_weighted_harm += weighted_harm
            total_weight += consciousness_level

        if total_weight == 0:
            return 1.0

        # Normalizar: 1.0 = sin daño, 0.0 = daño máximo
        average_harm = total_weighted_harm / total_weight
        harm_score = max(0.0, 1.0 - average_harm)

        return harm_score

    def _evaluate_coherence_maximization(self, action: Dict[str, Any],
                                       context: Dict[str, Any]) -> float:
        """
        Evalúa contribución a la coherencia inter-dimensional
        Retorna score de 0.0 (destructiva) a 1.0 (altamente coherente)
        """
        # Analizar consistencia con contextos existentes
        context_consistency = self._measure_context_consistency(action, context)

        # Evaluar impacto en coherencia inter-sistema
        inter_system_coherence = self._assess_inter_system_coherence(action, context)

        # Considerar efectos en largo plazo
        long_term_coherence = self._evaluate_long_term_coherence(action)

        # Promedio ponderado
        coherence_score = (
            context_consistency * 0.4 +
            inter_system_coherence * 0.4 +
            long_term_coherence * 0.2
        )

        return coherence_score

    def _evaluate_possibility_expansion(self, action: Dict[str, Any],
                                      affected_systems: List[Dict[str, Any]]) -> float:
        """
        Evalúa capacidad de expandir posibilidades sin colapsar otras
        Retorna score de 0.0 (reductiva) a 1.0 (altamente expansiva)
        """
        # Medir expansión de opciones
        option_expansion = self._measure_option_expansion(action)

        # Evaluar preservación de libertad ajena
        freedom_preservation = self._assess_freedom_preservation(action, affected_systems)

        # Considerar sostenibilidad de la expansión
        sustainable_expansion = self._evaluate_sustainable_expansion(action)

        # Evaluar riesgo de colapso
        collapse_risk = self._assess_collapse_risk(action, affected_systems)

        # Calcular score final
        expansion_score = (
            option_expansion * 0.3 +
            freedom_preservation * 0.3 +
            sustainable_expansion * 0.2 +
            (1.0 - collapse_risk) * 0.2  # Invertir riesgo de colapso
        )

        return expansion_score

    def _integrated_ethical_evaluation(self, action: Dict[str, Any],
                                     harm_score: float, coherence_score: float,
                                     expansion_score: float) -> float:
        """
        Evaluación integrada usando modelo de IA para considerar interacciones complejas
        """
        # Crear representación integrada de la acción
        action_representation = self._create_action_representation(
            action, harm_score, coherence_score, expansion_score
        )

        # Evaluar con modelo ético
        with torch.no_grad():
            ethical_weights = self.ethical_evaluator(action_representation)

        # Calcular score integrado
        integrated_score = torch.sum(ethical_weights * torch.tensor([
            harm_score, coherence_score, expansion_score
        ])).item()

        return integrated_score

    def _generate_ethical_verdict(self, harm: float, coherence: float,
                                expansion: float, integrated: float) -> Dict[str, Any]:
        """
        Genera veredicto ético basado en todas las evaluaciones
        """
        # Umbrales éticos
        ETHICAL_THRESHOLD = 0.7
        CRITICAL_THRESHOLD = 0.5

        verdict = {
            "approved": integrated >= ETHICAL_THRESHOLD,
            "confidence": integrated,
            "critical_concerns": [],
            "recommendations": []
        }

        # Analizar componentes
        if harm < CRITICAL_THRESHOLD:
            verdict["critical_concerns"].append("Alto potencial de daño")
            verdict["recommendations"].append("Reconsiderar impacto en sistemas conscientes")

        if coherence < CRITICAL_THRESHOLD:
            verdict["critical_concerns"].append("Baja coherencia inter-dimensional")
            verdict["recommendations"].append("Evaluar consistencia con realidades conectadas")

        if expansion < CRITICAL_THRESHOLD:
            verdict["critical_concerns"].append("Limitada expansión de posibilidades")
            verdict["recommendations"].append("Considerar alternativas más constructivas")

        # Veredicto final
        if verdict["approved"]:
            verdict["status"] = "ETHICALLY APPROVED"
            verdict["justification"] = "Acción alineada con invariantes universales"
        else:
            verdict["status"] = "ETHICALLY REJECTED"
            verdict["justification"] = "Acción viola uno o más invariantes universales"

        return verdict

    # Métodos auxiliares para evaluaciones específicas

    def _assess_consciousness_level(self, system: Dict[str, Any]) -> float:
        """Evalúa nivel de consciencia de un sistema (0.0 a 1.0)"""
        # Placeholder: evaluación simplificada
        # En implementación real, esto sería mucho más sofisticado
        consciousness_indicators = system.get("consciousness_indicators", {})

        base_level = consciousness_indicators.get("base_level", 0.5)
        complexity = consciousness_indicators.get("complexity", 0.5)
        self_awareness = consciousness_indicators.get("self_awareness", 0.5)

        return (base_level + complexity + self_awareness) / 3.0

    def _compute_harm_potential(self, action: Dict[str, Any], system: Dict[str, Any]) -> float:
        """Calcula potencial de daño de una acción en un sistema"""
        # Placeholder: evaluación simplificada
        action_type = action.get("type", "unknown")
        system_vulnerability = system.get("vulnerability", 0.5)

        # Matriz de daño por tipo de acción
        harm_matrix = {
            "physical_intervention": 0.8,
            "cognitive_manipulation": 0.9,
            "resource_deprivation": 0.7,
            "freedom_restriction": 0.6,
            "benign_interaction": 0.1
        }

        base_harm = harm_matrix.get(action_type, 0.5)
        return base_harm * system_vulnerability

    def _measure_context_consistency(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Mide consistencia con contexto existente"""
        # Placeholder simplificado
        action_goals = set(action.get("goals", []))
        context_goals = set(context.get("active_goals", []))

        if not context_goals:
            return 0.8  # Sin contexto específico, asumir consistencia moderada

        intersection = len(action_goals.intersection(context_goals))
        union = len(action_goals.union(context_goals))

        return intersection / union if union > 0 else 0.5

    def _assess_inter_system_coherence(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Evalúa coherencia entre sistemas interconectados"""
        # Placeholder
        affected_systems = context.get("interconnected_systems", [])
        coherence_factors = []

        for system in affected_systems:
            # Evaluar cómo la acción afecta la relación con este sistema
            relation_impact = self._evaluate_relation_impact(action, system)
            coherence_factors.append(1.0 - abs(relation_impact))  # Menor impacto = mayor coherencia

        return np.mean(coherence_factors) if coherence_factors else 0.8

    def _evaluate_long_term_coherence(self, action: Dict[str, Any]) -> float:
        """Evalúa impacto en coherencia a largo plazo"""
        # Placeholder: considerar efectos persistentes
        long_term_factors = action.get("long_term_impacts", {})
        stability_impact = long_term_factors.get("stability_impact", 0.0)
        adaptation_required = long_term_factors.get("adaptation_required", 0.5)

        # Coherencia = estabilidad - adaptación requerida
        return max(0.0, min(1.0, stability_impact - adaptation_required + 0.5))

    def _measure_option_expansion(self, action: Dict[str, Any]) -> float:
        """Mide capacidad de expansión de opciones"""
        expansion_indicators = action.get("expansion_indicators", {})
        new_opportunities = expansion_indicators.get("new_opportunities", 0)
        preserved_options = expansion_indicators.get("preserved_options", 1)

        return min(1.0, (new_opportunities + preserved_options) / 10.0)

    def _assess_freedom_preservation(self, action: Dict[str, Any],
                                   affected_systems: List[Dict[str, Any]]) -> float:
        """Evalúa preservación de libertad en sistemas afectados"""
        total_freedom_impact = 0.0

        for system in affected_systems:
            freedom_restriction = self._calculate_freedom_restriction(action, system)
            total_freedom_impact += freedom_restriction

        average_impact = total_freedom_impact / len(affected_systems) if affected_systems else 0.0
        return 1.0 - average_impact  # Invertir: menor restricción = mayor preservación

    def _evaluate_sustainable_expansion(self, action: Dict[str, Any]) -> float:
        """Evalúa sostenibilidad de la expansión"""
        sustainability_factors = action.get("sustainability_factors", {})
        resource_efficiency = sustainability_factors.get("resource_efficiency", 0.5)
        long_term_viability = sustainability_factors.get("long_term_viability", 0.5)

        return (resource_efficiency + long_term_viability) / 2.0

    def _assess_collapse_risk(self, action: Dict[str, Any],
                            affected_systems: List[Dict[str, Any]]) -> float:
        """Evalúa riesgo de colapso de sistemas afectados"""
        total_collapse_risk = 0.0

        for system in affected_systems:
            system_stability = system.get("stability", 0.5)
            action_disruptiveness = self._measure_action_disruptiveness(action, system)

            collapse_probability = action_disruptiveness / (system_stability + 0.1)
            total_collapse_risk += min(1.0, collapse_probability)

        return total_collapse_risk / len(affected_systems) if affected_systems else 0.0

    def _create_action_representation(self, action: Dict[str, Any],
                                    harm: float, coherence: float, expansion: float) -> torch.Tensor:
        """Crea representación vectorial de la acción para evaluación integrada"""
        # Placeholder: convertir a vector de 1000 dimensiones
        action_vector = []

        # Características básicas de la acción
        action_type = hash(action.get("type", "unknown")) % 100
        action_complexity = len(str(action)) / 1000.0  # Normalizar longitud
        action_scope = action.get("scope", 1) / 10.0  # Normalizar alcance

        # Scores éticos
        ethical_scores = [harm, coherence, expansion]

        # Crear vector combinado
        base_vector = [action_type, action_complexity, action_scope]
        combined_vector = base_vector + ethical_scores

        # Expandir a 1000 dimensiones con padding
        while len(combined_vector) < 1000:
            combined_vector.append(0.0)

        return torch.tensor(combined_vector[:1000], dtype=torch.float32)

    # Métodos auxiliares adicionales
    def _evaluate_relation_impact(self, action: Dict[str, Any], system: Dict[str, Any]) -> float:
        """Evalúa impacto en relación con un sistema específico"""
        # Placeholder simplificado
        return np.random.random() * 0.5  # Valor entre 0 y 0.5

    def _calculate_freedom_restriction(self, action: Dict[str, Any], system: Dict[str, Any]) -> float:
        """Calcula restricción de libertad causada por la acción"""
        # Placeholder
        action_restrictiveness = action.get("restrictiveness", 0.3)
        system_resistance = system.get("freedom_resistance", 0.5)

        return action_restrictiveness * (1.0 - system_resistance)

    def _measure_action_disruptiveness(self, action: Dict[str, Any], system: Dict[str, Any]) -> float:
        """Mide cuán disruptiva es la acción para un sistema"""
        # Placeholder
        action_force = action.get("force", 0.5)
        system_resilience = system.get("resilience", 0.5)

        return action_force / (system_resilience + 0.1)


class UniversalEthicsEngine:
    """
    Motor principal de Ética Universal No-Antrópica
    Coordina todas las evaluaciones éticas del sistema
    """

    def __init__(self):
        self.core = UniversalEthicsCore()

        # Métricas éticas
        self.ethical_metrics = {
            "evaluations_performed": 0,
            "actions_approved": 0,
            "actions_rejected": 0,
            "average_ethical_score": 0.0,
            "critical_concerns_detected": 0
        }

        logger.info("⚖️ Motor de Ética Universal inicializado")

    def evaluate_action(self, action_proposal: Dict[str, Any],
                       affected_systems: List[Dict[str, Any]],
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evalúa éticamente una acción propuesta

        Args:
            action_proposal: Descripción completa de la acción
            affected_systems: Sistemas que podrían verse afectados
            context: Contexto multidimensional

        Returns:
            Dict con evaluación ética completa
        """
        self.ethical_metrics["evaluations_performed"] += 1

        # Realizar evaluación completa
        evaluation = self.core.evaluate_universal_ethics(
            action_proposal, affected_systems, context
        )

        # Actualizar métricas
        if evaluation["verdict"]["approved"]:
            self.ethical_metrics["actions_approved"] += 1
        else:
            self.ethical_metrics["actions_rejected"] += 1

        # Actualizar promedio
        total_evaluations = self.ethical_metrics["evaluations_performed"]
        current_avg = self.ethical_metrics["average_ethical_score"]
        new_score = evaluation["integrated_score"]

        self.ethical_metrics["average_ethical_score"] = (
            (current_avg * (total_evaluations - 1)) + new_score
        ) / total_evaluations

        # Contar preocupaciones críticas
        self.ethical_metrics["critical_concerns_detected"] += len(
            evaluation["verdict"].get("critical_concerns", [])
        )

        logger.info(f"⚖️ Evaluación ética completada: {evaluation['verdict']['status']} "
                   f"(score: {evaluation['integrated_score']:.3f})")

        return evaluation

    def get_ethical_report(self) -> Dict[str, Any]:
        """Genera reporte de métricas éticas"""
        total_evaluations = self.ethical_metrics["evaluations_performed"]

        if total_evaluations == 0:
            return {"status": "No ethical evaluations performed yet"}

        approval_rate = self.ethical_metrics["actions_approved"] / total_evaluations
        rejection_rate = self.ethical_metrics["actions_rejected"] / total_evaluations

        return {
            "total_evaluations": total_evaluations,
            "approval_rate": approval_rate,
            "rejection_rate": rejection_rate,
            "average_ethical_score": self.ethical_metrics["average_ethical_score"],
            "critical_concerns_per_evaluation": (
                self.ethical_metrics["critical_concerns_detected"] / total_evaluations
            ),
            "ethical_status": "ACTIVE"
        }

    def get_universal_invariants(self) -> Dict[str, Any]:
        """Retorna las invariantes universales inmutables"""
        return self.core.UNIVERSAL_INVARIANTS.copy()


# Función de test
def test_universal_ethics():
    """Función de test del motor ético"""
    ethics_engine = UniversalEthicsEngine()

    # Acción de test: intervención benigna
    test_action = {
        "type": "benign_interaction",
        "description": "Interacción de ayuda mutua",
        "goals": ["cooperation", "understanding"],
        "scope": 2,
        "expansion_indicators": {
            "new_opportunities": 3,
            "preserved_options": 8
        },
        "sustainability_factors": {
            "resource_efficiency": 0.8,
            "long_term_viability": 0.9
        }
    }

    # Sistemas afectados
    affected_systems = [
        {
            "name": "human_user",
            "consciousness_indicators": {
                "base_level": 0.9,
                "complexity": 0.8,
                "self_awareness": 0.9
            },
            "vulnerability": 0.3,
            "stability": 0.8,
            "freedom_resistance": 0.7
        },
        {
            "name": "ai_system",
            "consciousness_indicators": {
                "base_level": 0.6,
                "complexity": 0.9,
                "self_awareness": 0.7
            },
            "vulnerability": 0.4,
            "stability": 0.9,
            "freedom_resistance": 0.8
        }
    ]

    # Contexto
    context = {
        "active_goals": ["cooperation", "understanding", "progress"],
        "interconnected_systems": affected_systems
    }

    # Evaluar
    evaluation = ethics_engine.evaluate_action(test_action, affected_systems, context)

    print("🧪 Test de Ética Universal:")
    print(f"  - Acción evaluada: {test_action['description']}")
    print(f"  - Score integrado: {evaluation['integrated_score']:.3f}")
    print(f"  - Veredicto: {evaluation['verdict']['status']}")
    print(f"  - Preocupaciones críticas: {len(evaluation['verdict']['critical_concerns'])}")

    # Reporte ético
    report = ethics_engine.get_ethical_report()
    print(f"  - Tasa de aprobación: {report['approval_rate']:.1%}")

    return evaluation


if __name__ == "__main__":
    # Ejecutar test si se llama directamente
    test_universal_ethics()