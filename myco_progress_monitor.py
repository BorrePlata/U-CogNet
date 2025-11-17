#!/usr/bin/env python3
"""
MycoNet Progress Monitor - Monitoreo Continuo de Comportamientos Emergentes

Este script monitorea continuamente el progreso de MycoNet, documentando:
- Comportamientos emergentes
- Evolución de la topología
- Métricas de rendimiento
- Patrones de aprendizaje
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MycoNetProgressMonitor:
    """
    Monitor continuo del progreso de MycoNet y sus comportamientos emergentes.
    """

    def __init__(self, reports_dir: str = "reports/myco_progress"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.progress_history: List[Dict[str, Any]] = []
        self.emergent_behaviors_log: List[Dict[str, Any]] = []
        self.topology_evolution: List[Dict[str, Any]] = []

        # Estado del sistema
        self.baseline_metrics = {}
        self.current_metrics = {}
        self.improvement_tracking = {}

        logger.info("MycoNet Progress Monitor initialized")

    def load_baseline_evaluation(self, baseline_file: str = "reports/myco_evaluation/myco_evaluation_20251117_130733.json"):
        """Cargar evaluación baseline para comparación"""
        try:
            with open(baseline_file, 'r') as f:
                data = json.load(f)

            self.baseline_metrics = data['baseline_results']['summary']
            logger.info("Baseline evaluation loaded successfully")

        except FileNotFoundError:
            logger.warning(f"Baseline file not found: {baseline_file}")
        except Exception as e:
            logger.error(f"Error loading baseline: {e}")

    def record_progress_snapshot(self, myco_integration, context: Dict[str, Any] = None):
        """
        Registrar snapshot del progreso actual de MycoNet.
        """
        timestamp = datetime.now()

        try:
            # Obtener estado de integración
            status = myco_integration.get_integration_status()

            # Calcular métricas de progreso
            progress_snapshot = {
                'timestamp': timestamp.isoformat(),
                'context': context or {},
                'integration_metrics': status['integration_metrics'],
                'myco_net_metrics': status['myco_net_metrics'],
                'topology': {
                    'nodes': list(status['active_nodes']),
                    'connections': status['active_connections']
                },
                'emergent_behaviors': status['myco_net_metrics'].get('emergent_behaviors', []),
                'safety_status': status.get('security_status', {})
            }

            # Calcular mejoras vs baseline
            if self.baseline_metrics:
                progress_snapshot['improvements'] = self._calculate_improvements(
                    status['myco_net_metrics'],
                    status['integration_metrics']
                )

            # Registrar snapshot
            self.progress_history.append(progress_snapshot)

            # Procesar comportamientos emergentes
            self._process_emergent_behaviors(progress_snapshot)

            # Registrar evolución de topología
            self._record_topology_evolution(progress_snapshot)

            logger.info(f"Progress snapshot recorded at {timestamp}")

            return progress_snapshot

        except Exception as e:
            logger.error(f"Error recording progress snapshot: {e}")
            return None

    def _calculate_improvements(self, myco_metrics: Dict, integration_metrics: Dict) -> Dict[str, float]:
        """Calcular mejoras vs baseline"""
        improvements = {}

        # Mapear métricas de MycoNet a métricas baseline
        metric_mapping = {
            'safety_compliance': 'security_score',
            'adaptation_rate': 'adaptation_score',
            'path_efficiency': 'efficiency'
        }

        for myco_metric, baseline_key in metric_mapping.items():
            if myco_metric in myco_metrics and baseline_key in self.baseline_metrics:
                current = myco_metrics[myco_metric]
                baseline = self.baseline_metrics[f"{baseline_key}_mean"]

                if baseline != 0:
                    improvement = ((current - baseline) / abs(baseline)) * 100
                    improvements[myco_metric] = improvement

        # Métricas específicas de integración
        if 'routes_processed' in integration_metrics:
            improvements['routing_efficiency'] = integration_metrics['routes_processed']

        if 'performance_gain' in integration_metrics:
            improvements['performance_gain'] = integration_metrics['performance_gain']

        return improvements

    def _process_emergent_behaviors(self, snapshot: Dict):
        """Procesar y catalogar comportamientos emergentes"""
        behaviors = snapshot.get('emergent_behaviors', [])

        for behavior in behaviors:
            if behavior not in [b['behavior'] for b in self.emergent_behaviors_log]:
                emergent_record = {
                    'behavior': behavior,
                    'first_observed': snapshot['timestamp'],
                    'frequency': 1,
                    'last_observed': snapshot['timestamp'],
                    'context': snapshot.get('context', {}),
                    'associated_metrics': snapshot.get('myco_net_metrics', {})
                }
                self.emergent_behaviors_log.append(emergent_record)
                logger.info(f"New emergent behavior detected: {behavior}")
            else:
                # Actualizar frecuencia
                for record in self.emergent_behaviors_log:
                    if record['behavior'] == behavior:
                        record['frequency'] += 1
                        record['last_observed'] = snapshot['timestamp']
                        break

    def _record_topology_evolution(self, snapshot: Dict):
        """Registrar evolución de la topología"""
        topology_record = {
            'timestamp': snapshot['timestamp'],
            'nodes_count': len(snapshot['topology']['nodes']),
            'connections_count': snapshot['topology']['connections'],
            'nodes': snapshot['topology']['nodes'],
            'context': snapshot.get('context', {})
        }

        self.topology_evolution.append(topology_record)

    def generate_progress_report(self) -> str:
        """Generar reporte completo de progreso"""
        report_path = self.reports_dir / f"myco_progress_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        with open(report_path, 'w') as f:
            f.write("# MycoNet Progress Report\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")

            # Resumen ejecutivo
            f.write("## Executive Summary\n\n")
            if self.progress_history:
                latest = self.progress_history[-1]
                f.write(f"- **Total Snapshots:** {len(self.progress_history)}\n")
                f.write(f"- **Emergent Behaviors:** {len(self.emergent_behaviors_log)}\n")
                f.write(f"- **Topology Changes:** {len(self.topology_evolution)}\n")

                if 'improvements' in latest:
                    f.write("- **Key Improvements:**\n")
                    for metric, improvement in latest['improvements'].items():
                        f.write(f"  - {metric}: {improvement:+.1f}%\n")

            f.write("\n")

            # Comportamientos emergentes
            f.write("## Emergent Behaviors\n\n")
            if self.emergent_behaviors_log:
                f.write("| Behavior | First Observed | Frequency | Last Observed |\n")
                f.write("|----------|----------------|-----------|---------------|\n")

                for behavior in self.emergent_behaviors_log:
                    f.write(f"| {behavior['behavior']} | {behavior['first_observed'][:19]} | {behavior['frequency']} | {behavior['last_observed'][:19]} |\n")

            f.write("\n")

            # Evolución de topología
            f.write("## Topology Evolution\n\n")
            if self.topology_evolution:
                f.write("| Timestamp | Nodes | Connections | Context |\n")
                f.write("|-----------|-------|-------------|---------|\n")

                for topo in self.topology_evolution[-10:]:  # últimos 10
                    context = topo.get('context', {}).get('phase', 'N/A')
                    f.write(f"| {topo['timestamp'][:19]} | {topo['nodes_count']} | {topo['connections_count']} | {context} |\n")

            f.write("\n")

            # Tendencias de mejora
            f.write("## Improvement Trends\n\n")
            if len(self.progress_history) > 1:
                self._generate_improvement_trends(f)

        logger.info(f"Progress report generated: {report_path}")
        return str(report_path)

    def _generate_improvement_trends(self, file):
        """Generar análisis de tendencias de mejora"""
        # Extraer datos de mejoras a lo largo del tiempo
        timestamps = []
        improvements_over_time = {}

        for snapshot in self.progress_history:
            if 'improvements' in snapshot:
                timestamps.append(snapshot['timestamp'])
                for metric, value in snapshot['improvements'].items():
                    if metric not in improvements_over_time:
                        improvements_over_time[metric] = []
                    improvements_over_time[metric].append(value)

        if improvements_over_time:
            file.write("### Trends Over Time\n\n")
            for metric, values in improvements_over_time.items():
                if len(values) > 1:
                    trend = "↗️ Improving" if values[-1] > values[0] else "↘️ Declining" if values[-1] < values[0] else "➡️ Stable"
                    file.write(f"- **{metric}:** {trend} ({values[0]:+.1f}% → {values[-1]:+.1f}%)\n")

    def create_visualizations(self):
        """Crear visualizaciones del progreso"""
        if not self.progress_history:
            return

        # Crear directorio para visualizaciones
        viz_dir = self.reports_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Gráfico de mejoras a lo largo del tiempo
        self._create_improvements_plot(viz_dir)

        # Gráfico de evolución de topología
        self._create_topology_plot(viz_dir)

        # Gráfico de comportamientos emergentes
        self._create_emergent_behaviors_plot(viz_dir)

    def _create_improvements_plot(self, viz_dir):
        """Crear gráfico de mejoras"""
        if len(self.progress_history) < 2:
            return

        data = []
        for snapshot in self.progress_history:
            if 'improvements' in snapshot:
                row = {'timestamp': snapshot['timestamp']}
                row.update(snapshot['improvements'])
                data.append(row)

        if data:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            plt.figure(figsize=(12, 8))
            for col in df.columns:
                if col != 'timestamp' and df[col].notna().any():
                    plt.plot(df['timestamp'], df[col], marker='o', label=col)

            plt.title('MycoNet Improvements Over Time')
            plt.xlabel('Time')
            plt.ylabel('Improvement (%)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(viz_dir / 'improvements_over_time.png', dpi=300, bbox_inches='tight')
            plt.close()

    def _create_topology_plot(self, viz_dir):
        """Crear gráfico de evolución de topología"""
        if not self.topology_evolution:
            return

        df = pd.DataFrame(self.topology_evolution)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(df['timestamp'], df['nodes_count'], marker='o')
        plt.title('Number of Nodes Over Time')
        plt.xlabel('Time')
        plt.ylabel('Node Count')
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        plt.plot(df['timestamp'], df['connections_count'], marker='o', color='orange')
        plt.title('Number of Connections Over Time')
        plt.xlabel('Time')
        plt.ylabel('Connection Count')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(viz_dir / 'topology_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_emergent_behaviors_plot(self, viz_dir):
        """Crear gráfico de comportamientos emergentes"""
        if not self.emergent_behaviors_log:
            return

        df = pd.DataFrame(self.emergent_behaviors_log)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='behavior', y='frequency')
        plt.title('Emergent Behaviors Frequency')
        plt.xlabel('Behavior')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(viz_dir / 'emergent_behaviors.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_progress_data(self):
        """Guardar todos los datos de progreso"""
        progress_data = {
            'progress_history': self.progress_history,
            'emergent_behaviors': self.emergent_behaviors_log,
            'topology_evolution': self.topology_evolution,
            'baseline_metrics': self.baseline_metrics,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_snapshots': len(self.progress_history),
                'emergent_behaviors_count': len(self.emergent_behaviors_log),
                'topology_changes': len(self.topology_evolution)
            }
        }

        filepath = self.reports_dir / f"myco_progress_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w') as f:
            json.dump(progress_data, f, indent=2, default=str)

        logger.info(f"Progress data saved to {filepath}")
        return str(filepath)

def continuous_monitoring_demo(myco_integration, duration_minutes: int = 5):
    """
    Demo de monitoreo continuo durante un período de tiempo.
    """
    monitor = MycoNetProgressMonitor()
    monitor.load_baseline_evaluation()

    logger.info(f"Starting continuous monitoring for {duration_minutes} minutes...")

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

    snapshot_count = 0

    while time.time() < end_time:
        # Simular actividad del sistema
        context = {
            'phase': 'monitoring',
            'activity': f'snapshot_{snapshot_count}',
            'simulated_load': 0.5 + 0.3 * (snapshot_count % 3)  # variación periódica
        }

        # Registrar snapshot
        snapshot = monitor.record_progress_snapshot(myco_integration, context)

        if snapshot:
            snapshot_count += 1
            logger.info(f"Snapshot {snapshot_count} recorded")

        # Esperar entre snapshots
        time.sleep(10)  # 10 segundos entre snapshots

    # Generar reportes finales
    logger.info("Generating final reports...")
    monitor.generate_progress_report()
    monitor.create_visualizations()
    monitor.save_progress_data()

    logger.info(f"Continuous monitoring completed. {snapshot_count} snapshots recorded.")

    return monitor

if __name__ == "__main__":
    # Demo de monitoreo continuo
    try:
        from src.ucognet.modules.mycelium.integration import MycoNetIntegration

        # Crear instancia de MycoNet
        myco_integration = MycoNetIntegration()
        myco_integration.initialize_standard_modules()

        # Ejecutar monitoreo continuo
        monitor = continuous_monitoring_demo(myco_integration, duration_minutes=1)  # 1 minuto para demo

        print("Continuous monitoring demo completed successfully!")

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure MycoNet is properly installed and configured.")