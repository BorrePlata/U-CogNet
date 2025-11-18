#!/usr/bin/env python3
"""
Interactive Chemistry Learning Experiment
Allows users to test and learn chemistry concepts through prediction, retrosynthesis, design, and creative thinking.
"""

import sys
import json
from pathlib import Path
from chemistry_cognitive_module import ChemistryCognitiveModule

class InteractiveChemistryExperiment:
    """Interactive experiment for chemistry learning."""

    def __init__(self):
        self.chemistry = ChemistryCognitiveModule()
        self.session_history = []

    def run_interactive_session(self):
        """Run interactive chemistry learning session."""
        print("🧪 U-CogNet Interactive Chemistry Learning Experiment")
        print("=" * 60)
        print("Aprende química orgánica a través de predicción, retrosíntesis, diseño y pensamiento creativo.")
        print()

        while True:
            self.show_menu()
            choice = input("Elige una opción (1-5): ").strip()

            if choice == '1':
                self.do_prediction_task()
            elif choice == '2':
                self.do_retrosynthesis_task()
            elif choice == '3':
                self.do_design_task()
            elif choice == '4':
                self.do_creative_task()
            elif choice == '5':
                self.show_progress()
            elif choice == '6':
                print("¡Gracias por aprender química con U-CogNet!")
                break
            else:
                print("Opción inválida. Intenta de nuevo.")

            print()

    def show_menu(self):
        """Show main menu."""
        print("MENÚ PRINCIPAL:")
        print("1. Predicción de Reacciones - ¿Qué producto se forma?")
        print("2. Retrosíntesis - ¿Cómo sintetizar un compuesto?")
        print("3. Diseño Molecular - Crea compuestos con propiedades específicas")
        print("4. Pensamiento Creativo - Ideas interdisciplinarias")
        print("5. Ver Progreso - Evalúa tu aprendizaje")
        print("6. Salir")

    def do_prediction_task(self):
        """Handle reaction prediction task."""
        print("\n🧪 PREDICCIÓN DE REACCIONES")
        print("-" * 30)
        print("Ingresa los reactantes separados por comas.")
        print("Ejemplos:")
        print("  - ethene, HCl")
        print("  - propanol")
        print("  - ethyl_bromide, NaOH")
        print()

        reactants_input = input("Reactantes: ").strip()
        if not reactants_input:
            print("Entrada vacía.")
            return

        reactants = [r.strip() for r in reactants_input.split(',')]

        # Optional conditions
        conditions_input = input("Condiciones (opcional, ej: temperature=25): ").strip()
        conditions = {}
        if conditions_input:
            try:
                key, value = conditions_input.split('=')
                conditions[key.strip()] = float(value.strip())
            except:
                print("Condiciones inválidas, usando condiciones estándar.")

        result = self.chemistry.predict_reaction(reactants, conditions)

        print(f"\nResultado de la predicción:")
        print(f"Reactantes: {result['reactants_used']}")
        print(f"Producto predicho: {result['predicted_products']}")
        print(f"Confianza: {result['confidence']:.2f}")
        print(f"Explicación: {result['explanation']}")

        # Store in session history
        self.session_history.append({
            'type': 'prediction',
            'input': reactants,
            'result': result
        })

    def do_retrosynthesis_task(self):
        """Handle retrosynthesis task."""
        print("\n🔄 RETROSÍNTESIS")
        print("-" * 30)
        print("Ingresa el compuesto objetivo para planear su síntesis.")
        print("Ejemplos: alcohol, alkene, alkyl_halide")
        print()

        target = input("Compuesto objetivo: ").strip()
        if not target:
            print("Entrada vacía.")
            return

        result = self.chemistry.plan_synthesis(target)

        print(f"\nPlan de retrosíntesis para: {target}")
        print(f"Rutas encontradas: {len(result['routes'])}")

        if result['routes']:
            for i, route in enumerate(result['routes'][:3], 1):
                print(f"Ruta {i}:")
                print(f"  Precursores: {route['precursors']}")
                print(f"  Rendimiento estimado: {route['estimated_yield']:.2f}")
                print(f"  Complejidad: {route['complexity']}")
                print()
        else:
            print("No se encontraron rutas de síntesis.")

        self.session_history.append({
            'type': 'retrosynthesis',
            'input': target,
            'result': result
        })

    def do_design_task(self):
        """Handle molecular design task."""
        print("\n🎯 DISEÑO MOLECULAR")
        print("-" * 30)
        print("Diseña un compuesto con propiedades específicas.")
        print("Propiedades disponibles: toxicity, solubility, bioavailability")
        print("Niveles: low, medium, high")
        print("Ejemplo: toxicity=low, solubility=high")
        print()

        properties_input = input("Propiedades objetivo: ").strip()
        if not properties_input:
            print("Entrada vacía.")
            return

        properties = {}
        try:
            for prop_pair in properties_input.split(','):
                key, value = prop_pair.split('=')
                properties[key.strip()] = value.strip()
        except:
            print("Formato inválido. Usa: propiedad=nivel, propiedad=nivel")
            return

        result = self.chemistry.design_compound(properties)

        print(f"\nDiseño molecular:")
        print(f"Propiedades objetivo: {properties}")
        print(f"Estructura propuesta: {result['proposed_structure']}")
        print(f"Propiedades estimadas: {result['estimated_properties']}")
        print(f"Complejidad de síntesis: {result['synthesis_complexity']}")
        print(f"Confianza: {result['confidence']:.2f}")

        self.session_history.append({
            'type': 'design',
            'input': properties,
            'result': result
        })

    def do_creative_task(self):
        """Handle creative thinking task."""
        print("\n💡 PENSAMIENTO CREATIVO")
        print("-" * 30)
        print("Genera ideas interdisciplinarias combinando campos.")
        print("Campos disponibles: pharmacology, nanotechnology, materials_science")
        print("Ejemplo: pharmacology, nanotechnology")
        print()

        domains_input = input("Campos a combinar: ").strip()
        if not domains_input:
            print("Entrada vacía.")
            return

        domains = [d.strip() for d in domains_input.split(',')]

        result = self.chemistry.creative_hybrid(domains)

        print(f"\nIdea creativa:")
        print(f"Campos combinados: {domains}")
        print(f"Concepto: {result['concept']}")
        print(f"Aplicaciones potenciales: {result['potential_applications']}")
        print(f"Novedad: {result['novelty_score']:.2f}")
        print(f"Factibilidad: {result['feasibility']}")

        self.session_history.append({
            'type': 'creative',
            'input': domains,
            'result': result
        })

    def show_progress(self):
        """Show learning progress."""
        print("\n📊 PROGRESO DE APRENDIZAJE")
        print("-" * 30)

        metrics = self.chemistry.get_metrics()
        evaluation = self.chemistry.evaluate_performance()

        print(f"Predicciones realizadas: {metrics['predictions_made']}")
        print(f"Diseños creados: {metrics['designs_created']}")
        print(f"Ideas creativas: {metrics['creative_ideas']}")
        print(".3f")
        print(".3f")
        print(".3f")

        print(f"\nSesión actual - Tareas completadas: {len(self.session_history)}")

        task_counts = {}
        for task in self.session_history:
            task_type = task['type']
            task_counts[task_type] = task_counts.get(task_type, 0) + 1

        print("Distribución de tareas:")
        for task_type, count in task_counts.items():
            print(f"  {task_type.capitalize()}: {count}")

        # Save session data
        session_data = {
            'session_history': self.session_history,
            'final_metrics': metrics,
            'evaluation': evaluation
        }

        with open('chemistry_learning_session.json', 'w') as f:
            json.dump(session_data, f, indent=2, default=str)

        print("\n💾 Sesión guardada en: chemistry_learning_session.json")

def main():
    experiment = InteractiveChemistryExperiment()
    experiment.run_interactive_session()

if __name__ == "__main__":
    main()