#!/usr/bin/env python3
"""
U-CogNet Chemistry Module - Organic Chemistry Prediction and Design
Implements prediction, retrosynthesis, molecular design, and creative thinking capabilities.
"""

import sys
import os
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np

# Add src to path for U-CogNet integration
sys.path.insert(0, str(Path(__file__).parent / "src"))

class OrganicChemistryPredictor:
    """
    Basic organic chemistry prediction module.
    Implements simple reaction prediction using rule-based approach.
    """

    def __init__(self):
        # Basic reaction templates (simplified)
        self.reaction_rules = {
            'alkene_hydrohalogenation': {
                'reactants': ['alkene', 'HX'],
                'products': ['alkyl_halide'],
                'description': 'HX adds to alkene following Markovnikov rule'
            },
            'alkene_hydration': {
                'reactants': ['alkene', 'H2O'],
                'products': ['alcohol'],
                'description': 'Water adds to alkene with acid catalysis'
            },
            'alcohol_dehydration': {
                'reactants': ['alcohol'],
                'products': ['alkene'],
                'description': 'Alcohol loses water to form alkene'
            },
            'sn2_substitution': {
                'reactants': ['alkyl_halide', 'nucleophile'],
                'products': ['substituted_product'],
                'description': 'Nucleophilic substitution'
            }
        }

        # Learning history
        self.prediction_history = []
        self.accuracy_history = []

    def predict_reaction_product(self, reactants: List[str], conditions: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Predict the product of an organic reaction.

        Args:
            reactants: List of reactant names/types
            conditions: Reaction conditions (temperature, catalyst, etc.)

        Returns:
            Dictionary with prediction results
        """
        reactants_sorted = sorted(reactants)

        # Simple rule matching
        for reaction_name, rule in self.reaction_rules.items():
            if self._match_reactants(reactants_sorted, rule['reactants']):
                product = self._generate_product(reactants_sorted, rule)
                confidence = self._calculate_confidence(rule, conditions)

                result = {
                    'reaction_type': reaction_name,
                    'predicted_products': product,
                    'confidence': confidence,
                    'explanation': rule['description'],
                    'reactants_used': reactants_sorted
                }

                # Store in history
                self.prediction_history.append(result)
                return result

        # No matching rule found
        return {
            'reaction_type': 'unknown',
            'predicted_products': ['unable_to_predict'],
            'confidence': 0.0,
            'explanation': 'No matching reaction rule found',
            'reactants_used': reactants_sorted
        }

    def _match_reactants(self, reactants: List[str], rule_reactants: List[str]) -> bool:
        """Check if reactants match a reaction rule."""
        if len(reactants) != len(rule_reactants):
            return False

        # Flexible string matching with synonyms
        synonyms = {
            'alkene': ['alkene', 'ethene', 'ethylene', 'propene', 'alkene'],
            'HX': ['HX', 'HCl', 'HBr', 'HI', 'hydrohalic acid'],
            'H2O': ['H2O', 'water'],
            'alcohol': ['alcohol', 'ethanol', 'propanol', 'alcohol'],
            'alkyl_halide': ['alkyl_halide', 'ethyl_bromide', 'alkyl halide'],
            'nucleophile': ['nucleophile', 'NaOH', 'hydroxide']
        }

        for rule_reactant in rule_reactants:
            matched = False
            for reactant in reactants:
                reactant_lower = reactant.lower()
                if rule_reactant in synonyms:
                    if any(syn.lower() in reactant_lower or reactant_lower in syn.lower() for syn in synonyms[rule_reactant]):
                        matched = True
                        break
                elif rule_reactant.lower() in reactant_lower:
                    matched = True
                    break
            if not matched:
                return False

        return True

    def _generate_product(self, reactants: List[str], rule: Dict) -> List[str]:
        """Generate product names based on reaction rule."""
        # Simplified product generation
        products = []
        for product_type in rule['products']:
            if product_type == 'alkyl_halide':
                products.append('alkyl halide')
            elif product_type == 'alcohol':
                products.append('alcohol')
            elif product_type == 'alkene':
                products.append('alkene')
            elif product_type == 'vinyl_halide':
                products.append('vinyl halide')
            elif product_type == 'substituted_product':
                products.append('substituted product')
            else:
                products.append(product_type)

        return products

    def _calculate_confidence(self, rule: Dict, conditions: Dict = None) -> float:
        """Calculate prediction confidence."""
        base_confidence = 0.7  # Base confidence for known reactions

        if conditions:
            # Adjust confidence based on conditions
            if 'temperature' in conditions and conditions['temperature'] > 100:
                base_confidence += 0.1
            if 'catalyst' in conditions:
                base_confidence += 0.1

        return min(1.0, base_confidence)

class RetrosynthesisPlanner:
    """
    Retrosynthesis planning module.
    Proposes synthetic routes for target molecules.
    """

    def __init__(self):
        # Simplified retrosynthetic rules
        self.retro_rules = {
            'alcohol_from_alkene': {
                'target': 'alcohol',
                'precursors': ['alkene', 'H2O'],
                'conditions': 'acid-catalyzed hydration',
                'yield_estimate': 0.8
            },
            'alkene_from_alcohol': {
                'target': 'alkene',
                'precursors': ['alcohol'],
                'conditions': 'acid-catalyzed dehydration',
                'yield_estimate': 0.7
            },
            'alkyl_halide_from_alcohol': {
                'target': 'alkyl_halide',
                'precursors': ['alcohol', 'HX'],
                'conditions': 'halogenation',
                'yield_estimate': 0.9
            }
        }

    def plan_retrosynthesis(self, target_molecule: str, max_steps: int = 3) -> Dict[str, Any]:
        """
        Plan retrosynthetic route for target molecule.

        Args:
            target_molecule: Target compound name
            max_steps: Maximum retrosynthetic steps

        Returns:
            Retrosynthetic plan
        """
        plan = {
            'target': target_molecule,
            'routes': [],
            'total_steps': 0,
            'estimated_yield': 1.0
        }

        # Generate possible routes
        routes = self._generate_routes(target_molecule, max_steps)

        for route in routes[:3]:  # Limit to top 3 routes
            plan['routes'].append(route)

        return plan

    def _generate_routes(self, target: str, max_steps: int) -> List[Dict]:
        """Generate possible retrosynthetic routes."""
        routes = []

        # Simple rule-based route generation
        for rule_name, rule in self.retro_rules.items():
            if rule['target'] in target.lower():
                route = {
                    'steps': [rule],
                    'precursors': rule['precursors'],
                    'estimated_yield': rule['yield_estimate'],
                    'complexity': 1
                }
                routes.append(route)

        return routes

class MolecularDesigner:
    """
    Molecular design module for property optimization.
    """

    def __init__(self):
        self.property_ranges = {
            'toxicity': {'low': 0.0, 'high': 1.0},
            'solubility': {'low': 0.0, 'high': 10.0},
            'bioavailability': {'low': 0.0, 'high': 1.0}
        }

    def design_molecule(self, target_properties: Dict[str, str]) -> Dict[str, Any]:
        """
        Design a molecule with specified properties.

        Args:
            target_properties: Dict of property -> target level (e.g., {'toxicity': 'low', 'solubility': 'high'})

        Returns:
            Designed molecule information
        """
        design = {
            'target_properties': target_properties,
            'proposed_structure': self._generate_structure(target_properties),
            'estimated_properties': self._estimate_properties(target_properties),
            'synthesis_complexity': 'medium',
            'confidence': 0.6
        }

        return design

    def _generate_structure(self, properties: Dict[str, str]) -> str:
        """Generate a molecular structure description."""
        # Simplified structure generation
        base_structures = {
            'low_toxicity': 'phenol derivative',
            'high_solubility': 'carboxylic acid',
            'balanced': 'aromatic amine'
        }

        structure_parts = []
        for prop, level in properties.items():
            if prop == 'toxicity' and level == 'low':
                structure_parts.append(base_structures['low_toxicity'])
            elif prop == 'solubility' and level == 'high':
                structure_parts.append(base_structures['high_solubility'])

        if not structure_parts:
            structure_parts.append(base_structures['balanced'])

        return ' + '.join(structure_parts)

    def _estimate_properties(self, target_properties: Dict[str, str]) -> Dict[str, float]:
        """Estimate properties of designed molecule."""
        estimates = {}
        for prop, level in target_properties.items():
            if level == 'low':
                estimates[prop] = 0.2
            elif level == 'high':
                estimates[prop] = 0.8
            else:
                estimates[prop] = 0.5
        return estimates

class CreativeChemistryThinker:
    """
    Creative thinking module for interdisciplinary chemical concepts.
    """

    def __init__(self):
        self.concept_domains = {
            'pharmacology': ['drug delivery', 'targeting', 'metabolism'],
            'nanotechnology': ['nanoparticles', 'surface modification', 'controlled release'],
            'materials_science': ['polymers', 'biomaterials', 'smart materials']
        }

    def generate_creative_idea(self, domains: List[str]) -> Dict[str, Any]:
        """
        Generate creative interdisciplinary idea.

        Args:
            domains: List of domains to combine

        Returns:
            Creative idea description
        """
        if len(domains) < 2:
            domains = ['pharmacology', 'nanotechnology']  # Default combination

        idea = {
            'combined_domains': domains,
            'concept': self._combine_concepts(domains),
            'potential_applications': self._generate_applications(domains),
            'novelty_score': random.uniform(0.6, 0.9),
            'feasibility': random.choice(['high', 'medium', 'low'])
        }

        return idea

    def _combine_concepts(self, domains: List[str]) -> str:
        """Combine concepts from different domains."""
        concepts = []
        for domain in domains:
            if domain in self.concept_domains:
                concepts.extend(self.concept_domains[domain])

        # Create a creative combination
        selected = random.sample(concepts, min(3, len(concepts)))
        return f"Integration of {', '.join(selected)} for advanced therapeutic systems"

    def _generate_applications(self, domains: List[str]) -> List[str]:
        """Generate potential applications."""
        applications = [
            "Targeted drug delivery systems",
            "Smart nanomaterials for diagnostics",
            "Biodegradable implants with controlled release",
            "Nanoparticle-based imaging agents"
        ]
        return random.sample(applications, 2)

class ChemistryCognitiveModule:
    """
    Main chemistry module integrating all capabilities.
    Follows U-CogNet modular architecture principles.
    """

    def __init__(self):
        self.predictor = OrganicChemistryPredictor()
        self.retrosynthesis = RetrosynthesisPlanner()
        self.designer = MolecularDesigner()
        self.creative_thinker = CreativeChemistryThinker()

        # Cognitive metrics
        self.performance_metrics = {
            'predictions_made': 0,
            'accurate_predictions': 0,
            'designs_created': 0,
            'creative_ideas': 0
        }

    def predict_reaction(self, reactants: List[str], conditions: Dict = None) -> Dict[str, Any]:
        """Predict reaction product."""
        result = self.predictor.predict_reaction_product(reactants, conditions)
        self.performance_metrics['predictions_made'] += 1
        return result

    def plan_synthesis(self, target: str) -> Dict[str, Any]:
        """Plan retrosynthetic route."""
        return self.retrosynthesis.plan_retrosynthesis(target)

    def design_compound(self, properties: Dict[str, str]) -> Dict[str, Any]:
        """Design molecule with target properties."""
        result = self.designer.design_molecule(properties)
        self.performance_metrics['designs_created'] += 1
        return result

    def creative_hybrid(self, domains: List[str]) -> Dict[str, Any]:
        """Generate creative interdisciplinary idea."""
        result = self.creative_thinker.generate_creative_idea(domains)
        self.performance_metrics['creative_ideas'] += 1
        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()

    def evaluate_performance(self) -> Dict[str, Any]:
        """Evaluate overall module performance."""
        metrics = self.get_metrics()
        evaluation = {
            'prediction_accuracy': metrics.get('accurate_predictions', 0) / max(1, metrics.get('predictions_made', 1)),
            'design_productivity': metrics.get('designs_created', 0),
            'creative_output': metrics.get('creative_ideas', 0),
            'overall_score': (metrics.get('accurate_predictions', 0) + metrics.get('designs_created', 0) + metrics.get('creative_ideas', 0)) / 3
        }
        return evaluation

def demo_chemistry_module():
    """Demonstrate chemistry module capabilities."""
    print("🧪 U-CogNet Chemistry Module Demo")
    print("=" * 50)

    chemistry = ChemistryCognitiveModule()

    # 1. Reaction Prediction
    print("\n1. PREDICCIÓN DE REACCIONES")
    print("-" * 30)

    test_reactions = [
        (['ethene', 'HCl'], {'temperature': 25}),
        (['propanol'], {}),
        (['ethyl_bromide', 'NaOH'], {})
    ]

    for reactants, conditions in test_reactions:
        result = chemistry.predict_reaction(reactants, conditions)
        print(f"Reactants: {reactants}")
        print(f"Predicted: {result['predicted_products']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Explanation: {result['explanation']}")
        print()

    # 2. Retrosynthesis
    print("\n2. RETROSÍNTESIS")
    print("-" * 30)

    targets = ['alcohol', 'alkene']
    for target in targets:
        plan = chemistry.plan_synthesis(target)
        print(f"Target: {target}")
        print(f"Routes found: {len(plan['routes'])}")
        if plan['routes']:
            print(f"Sample route: {plan['routes'][0]['precursors']} -> {target}")
        print()

    # 3. Molecular Design
    print("\n3. DISEÑO MOLECULAR")
    print("-" * 30)

    design_specs = {'toxicity': 'low', 'solubility': 'high'}
    design = chemistry.design_compound(design_specs)
    print(f"Target properties: {design_specs}")
    print(f"Proposed structure: {design['proposed_structure']}")
    print(f"Estimated properties: {design['estimated_properties']}")
    print()

    # 4. Creative Thinking
    print("\n4. PENSAMIENTO CREATIVO")
    print("-" * 30)

    domains = ['pharmacology', 'nanotechnology']
    idea = chemistry.creative_hybrid(domains)
    print(f"Combined domains: {domains}")
    print(f"Creative concept: {idea['concept']}")
    print(f"Applications: {idea['potential_applications']}")
    print(f"Novelty: {idea['novelty_score']:.2f}")
    print()

    # Performance Evaluation
    print("\n5. EVALUACIÓN DEL MÓDULO")
    print("-" * 30)
    evaluation = chemistry.evaluate_performance()
    print(f"Predictions made: {chemistry.get_metrics()['predictions_made']}")
    print(f"Designs created: {chemistry.get_metrics()['designs_created']}")
    print(f"Creative ideas: {chemistry.get_metrics()['creative_ideas']}")
    print(".3f")

if __name__ == "__main__":
    demo_chemistry_module()