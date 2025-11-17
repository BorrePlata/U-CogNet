#!/usr/bin/env python3
"""
Evaluación de MycoNet: Métricas Antes vs Después de la Integración

Este script evalúa el impacto de integrar MycoNet en U-CogNet,
midiendo mejoras en rendimiento, seguridad, adaptabilidad y comportamientos emergentes.
"""

import numpy as np
import pandas as pd
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import json
import os

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MycoNetEvaluator:
    """
    Evaluador de rendimiento de MycoNet vs sistema base.

    Compara métricas de rendimiento, seguridad y adaptabilidad.
    """

    def __init__(self, security_architecture=None):
        self.security_architecture = security_architecture
        self.baseline_results = {}
        self.myco_results = {}
        self.comparison_metrics = {}

        # Configurar directorio de resultados
        self.results_dir = "reports/myco_evaluation"
        os.makedirs(self.results_dir, exist_ok=True)

    def run_baseline_evaluation(self, n_runs: int = 10) -> Dict[str, Any]:
        """
        Ejecutar evaluación del sistema base (sin MycoNet).
        """
        logger.info("Running baseline evaluation (without MycoNet)...")

        baseline_metrics = []

        for run in range(n_runs):
            logger.info(f"Baseline run {run + 1}/{n_runs}")

            # Simular procesamiento cognitivo básico
            start_time = time.time()

            # Simular pipeline: input -> vision -> cognitive -> output
            processing_time = self._simulate_baseline_processing()
            security_score = self._simulate_baseline_security()
            adaptation_score = self._simulate_baseline_adaptation()

            end_time = time.time()

            metrics = {
                'run_id': run,
                'processing_time': processing_time,
                'total_time': end_time - start_time,
                'security_score': security_score,
                'adaptation_score': adaptation_score,
                'efficiency': 1.0 / processing_time,  # operaciones por segundo
                'emergent_behaviors': 0,  # baseline no tiene comportamientos emergentes
                'resource_usage': np.random.uniform(0.6, 0.9),  # uso simulado
                'error_rate': np.random.uniform(0.05, 0.15)  # tasa de error simulado
            }

            baseline_metrics.append(metrics)

        self.baseline_results = {
            'metrics': baseline_metrics,
            'summary': self._calculate_summary_stats(baseline_metrics),
            'timestamp': datetime.now().isoformat()
        }

        return self.baseline_results

    def run_myco_evaluation(self, n_runs: int = 10) -> Dict[str, Any]:
        """
        Ejecutar evaluación con MycoNet integrado.
        """
        logger.info("Running MycoNet evaluation...")

        # Importar e inicializar MycoNet
        try:
            from src.ucognet.modules.mycelium.integration import MycoNetIntegration
            myco_integration = MycoNetIntegration(self.security_architecture)
            myco_integration.initialize_standard_modules()
        except ImportError as e:
            logger.error(f"Failed to import MycoNet: {e}")
            return {'error': str(e)}

        myco_metrics = []

        for run in range(n_runs):
            logger.info(f"MycoNet run {run + 1}/{n_runs}")

            start_time = time.time()

            # Procesar con MycoNet
            task_types = ['vision_detection', 'audio_processing', 'multimodal_fusion', 'learning_adaptation']

            run_results = []
            emergent_behaviors = 0

            for task in task_types:
                # Simular métricas de tarea
                task_metrics = {
                    'accuracy': np.random.uniform(0.7, 0.95),
                    'latency': np.random.uniform(0.1, 0.5),
                    'resource_usage': np.random.uniform(0.4, 0.8)
                }

                # Procesar con MycoNet
                result, confidence = myco_integration.process_request(
                    task_id=f"{task}_run_{run}",
                    metrics=task_metrics,
                    phase="execution"
                )

                if result:
                    # Simular ejecución y retroalimentación
                    actual_reward = confidence * np.random.uniform(0.8, 1.2)
                    myco_integration.reinforce_learning(result, actual_reward)

                    run_results.append({
                        'task': task,
                        'success': True,
                        'confidence': confidence,
                        'reward': actual_reward,
                        'path_length': len(result['path'])
                    })
                else:
                    run_results.append({
                        'task': task,
                        'success': False,
                        'confidence': 0.0,
                        'reward': 0.0,
                        'path_length': 0
                    })

            # Ciclo de mantenimiento
            myco_integration.maintenance_cycle()

            # Obtener métricas del sistema
            integration_status = myco_integration.get_integration_status()
            myco_net_metrics = integration_status['myco_net_metrics']

            # Detectar comportamientos emergentes
            emergent_behaviors = len(myco_net_metrics.get('emergent_behaviors', []))

            end_time = time.time()

            # Calcular métricas agregadas
            successful_tasks = sum(1 for r in run_results if r['success'])
            avg_confidence = np.mean([r['confidence'] for r in run_results])
            avg_reward = np.mean([r['reward'] for r in run_results])
            avg_path_length = np.mean([r['path_length'] for r in run_results if r['path_length'] > 0])

            metrics = {
                'run_id': run,
                'processing_time': end_time - start_time,
                'total_time': end_time - start_time,
                'security_score': myco_net_metrics.get('safety_compliance', 0.8),
                'adaptation_score': myco_net_metrics.get('adaptation_rate', 0.7),
                'efficiency': successful_tasks / (end_time - start_time) if (end_time - start_time) > 0 else 0,
                'emergent_behaviors': emergent_behaviors,
                'resource_usage': np.mean([r.get('resource_usage', 0.6) for r in run_results]),
                'error_rate': (len(run_results) - successful_tasks) / len(run_results),
                'task_success_rate': successful_tasks / len(run_results),
                'avg_confidence': avg_confidence,
                'avg_reward': avg_reward,
                'avg_path_length': avg_path_length,
                'path_efficiency': myco_net_metrics.get('path_efficiency', 0.0),
                'pheromone_entropy': myco_net_metrics.get('pheromone_entropy', 0.0)
            }

            myco_metrics.append(metrics)

        self.myco_results = {
            'metrics': myco_metrics,
            'summary': self._calculate_summary_stats(myco_metrics),
            'integration_status': integration_status,
            'timestamp': datetime.now().isoformat()
        }

        return self.myco_results

    def _simulate_baseline_processing(self) -> float:
        """Simular tiempo de procesamiento del sistema base"""
        # Simular pipeline lineal: input -> vision -> cognitive -> output
        base_time = 0.3  # tiempo base en segundos
        variation = np.random.normal(0, 0.05)  # variación aleatoria
        return max(0.1, base_time + variation)

    def _simulate_baseline_security(self) -> float:
        """Simular evaluación de seguridad del sistema base"""
        # Seguridad básica sin MycoNet
        return np.random.uniform(0.6, 0.85)

    def _simulate_baseline_adaptation(self) -> float:
        """Simular capacidad de adaptación del sistema base"""
        # Adaptación limitada sin red micelial
        return np.random.uniform(0.4, 0.7)

    def _calculate_summary_stats(self, metrics_list: List[Dict]) -> Dict[str, Any]:
        """Calcular estadísticas resumen de una lista de métricas"""
        if not metrics_list:
            return {}

        df = pd.DataFrame(metrics_list)

        summary = {}
        for column in df.columns:
            if column == 'run_id':
                continue
            if df[column].dtype in ['int64', 'float64']:
                summary[f"{column}_mean"] = df[column].mean()
                summary[f"{column}_std"] = df[column].std()
                summary[f"{column}_min"] = df[column].min()
                summary[f"{column}_max"] = df[column].max()

        return summary

    def compare_results(self) -> Dict[str, Any]:
        """
        Comparar resultados baseline vs MycoNet y calcular mejoras.
        """
        logger.info("Comparing baseline vs MycoNet results...")

        if not self.baseline_results or not self.myco_results:
            logger.error("Both baseline and MycoNet results required for comparison")
            return {'error': 'Missing results'}

        baseline_summary = self.baseline_results['summary']
        myco_summary = self.myco_results['summary']

        comparison = {}

        # Métricas a comparar
        metrics_to_compare = [
            'processing_time', 'security_score', 'adaptation_score', 'efficiency',
            'resource_usage', 'error_rate', 'emergent_behaviors'
        ]

        for metric in metrics_to_compare:
            baseline_mean = baseline_summary.get(f"{metric}_mean", 0)
            myco_mean = myco_summary.get(f"{metric}_mean", 0)

            if baseline_mean != 0:
                improvement = ((myco_mean - baseline_mean) / abs(baseline_mean)) * 100
            else:
                improvement = 0

            comparison[metric] = {
                'baseline_mean': baseline_mean,
                'myco_mean': myco_mean,
                'improvement_percent': improvement,
                'improvement_type': 'positive' if improvement > 0 else 'negative'
            }

        # Métricas específicas de MycoNet
        myco_specific = {
            'task_success_rate': myco_summary.get('task_success_rate_mean', 0),
            'avg_confidence': myco_summary.get('avg_confidence_mean', 0),
            'avg_reward': myco_summary.get('avg_reward_mean', 0),
            'path_efficiency': myco_summary.get('path_efficiency_mean', 0),
            'pheromone_entropy': myco_summary.get('pheromone_entropy_mean', 0)
        }

        self.comparison_metrics = {
            'comparisons': comparison,
            'myco_specific': myco_specific,
            'overall_assessment': self._assess_overall_impact(comparison),
            'timestamp': datetime.now().isoformat()
        }

        return self.comparison_metrics

    def _assess_overall_impact(self, comparisons: Dict) -> Dict[str, Any]:
        """Evaluar el impacto general de MycoNet"""
        # Definir pesos de importancia para cada métrica
        weights = {
            'processing_time': -0.3,  # negativo porque menor tiempo es mejor
            'security_score': 0.25,
            'adaptation_score': 0.25,
            'efficiency': 0.3,
            'resource_usage': -0.2,  # negativo porque menor uso es mejor
            'error_rate': -0.3,      # negativo porque menor error es mejor
            'emergent_behaviors': 0.15
        }

        overall_score = 0
        total_weight = 0

        for metric, weight in weights.items():
            if metric in comparisons:
                improvement = comparisons[metric]['improvement_percent']
                # Normalizar mejora (asumiendo que mejoras >50% son excelentes)
                normalized_improvement = min(improvement / 50.0, 2.0)  # cap at 200%
                overall_score += weight * normalized_improvement
                total_weight += abs(weight)

        if total_weight > 0:
            overall_score = overall_score / total_weight

        # Clasificar impacto
        if overall_score > 0.3:
            assessment = "EXCELLENT"
            description = "MycoNet proporciona mejoras significativas en rendimiento y capacidades"
        elif overall_score > 0.1:
            assessment = "GOOD"
            description = "MycoNet mejora el sistema de manera moderada"
        elif overall_score > -0.1:
            assessment = "NEUTRAL"
            description = "MycoNet mantiene rendimiento similar con capacidades adicionales"
        else:
            assessment = "CONCERNING"
            description = "MycoNet puede estar degradando el rendimiento - investigar"

        return {
            'overall_score': overall_score,
            'assessment': assessment,
            'description': description,
            'recommendations': self._generate_recommendations(comparisons, overall_score)
        }

    def _generate_recommendations(self, comparisons: Dict, overall_score: float) -> List[str]:
        """Generar recomendaciones basadas en los resultados"""
        recommendations = []

        # Analizar métricas individuales
        if comparisons.get('security_score', {}).get('improvement_percent', 0) < 0:
            recommendations.append("Investigar por qué la seguridad disminuyó con MycoNet")

        if comparisons.get('processing_time', {}).get('improvement_percent', 0) > 20:
            recommendations.append("MycoNet mejora significativamente la velocidad de procesamiento")

        if comparisons.get('emergent_behaviors', {}).get('improvement_percent', 0) > 0:
            recommendations.append("MycoNet está generando comportamientos emergentes - monitorear")

        if comparisons.get('error_rate', {}).get('improvement_percent', 0) < -10:
            recommendations.append("Aumentó la tasa de error - revisar integración de seguridad")

        # Recomendaciones generales
        if overall_score > 0.2:
            recommendations.append("MycoNet es beneficioso - considerar despliegue completo")
        elif overall_score < 0:
            recommendations.append("Revisar configuración de MycoNet - posible problema de integración")

        return recommendations

    def save_results(self, filename: str = None):
        """Guardar todos los resultados en archivos JSON"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"myco_evaluation_{timestamp}"

        results = {
            'baseline_results': self.baseline_results,
            'myco_results': self.myco_results,
            'comparison_metrics': self.comparison_metrics,
            'metadata': {
                'evaluation_timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'description': 'MycoNet integration evaluation results'
            }
        }

        filepath = os.path.join(self.results_dir, f"{filename}.json")
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {filepath}")

        # Generar reporte resumido
        self._generate_summary_report(filepath)

    def _generate_summary_report(self, results_filepath: str):
        """Generar reporte resumido en markdown"""
        if not self.comparison_metrics:
            return

        report_path = results_filepath.replace('.json', '_report.md')

        with open(report_path, 'w') as f:
            f.write("# MycoNet Integration Evaluation Report\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")

            # Resultados generales
            assessment = self.comparison_metrics.get('overall_assessment', {})
            f.write("## Overall Assessment\n\n")
            f.write(f"**Score:** {assessment.get('overall_score', 0):.3f}\n")
            f.write(f"**Rating:** {assessment.get('assessment', 'UNKNOWN')}\n")
            f.write(f"**Description:** {assessment.get('description', '')}\n\n")

            # Comparaciones detalladas
            f.write("## Detailed Comparisons\n\n")
            f.write("| Metric | Baseline | MycoNet | Improvement |\n")
            f.write("|--------|----------|---------|-------------|\n")

            comparisons = self.comparison_metrics.get('comparisons', {})
            for metric, data in comparisons.items():
                baseline = data.get('baseline_mean', 0)
                myco = data.get('myco_mean', 0)
                improvement = data.get('improvement_percent', 0)
                f.write(".3f")

            f.write("\n## MycoNet Specific Metrics\n\n")
            myco_specific = self.comparison_metrics.get('myco_specific', {})
            for metric, value in myco_specific.items():
                f.write(f"- **{metric}:** {value:.3f}\n")

            f.write("\n## Recommendations\n\n")
            recommendations = assessment.get('recommendations', [])
            for rec in recommendations:
                f.write(f"- {rec}\n")

        logger.info(f"Summary report saved to {report_path}")

def main():
    """Función principal para ejecutar la evaluación"""
    logger.info("Starting MycoNet evaluation...")

    # Inicializar evaluador
    evaluator = MycoNetEvaluator()

    try:
        # Ejecutar evaluación baseline
        logger.info("Phase 1: Baseline evaluation")
        baseline_results = evaluator.run_baseline_evaluation(n_runs=5)

        # Ejecutar evaluación con MycoNet
        logger.info("Phase 2: MycoNet evaluation")
        myco_results = evaluator.run_myco_evaluation(n_runs=5)

        # Comparar resultados
        logger.info("Phase 3: Results comparison")
        comparison = evaluator.compare_results()

        # Guardar resultados
        evaluator.save_results()

        # Mostrar resumen
        assessment = comparison.get('overall_assessment', {})
        logger.info(f"Evaluation complete. Overall assessment: {assessment.get('assessment', 'UNKNOWN')}")
        logger.info(f"Overall score: {assessment.get('overall_score', 0):.3f}")

        return comparison

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    results = main()
    print("\nEvaluation Results Summary:")
    print(json.dumps(results, indent=2, default=str))