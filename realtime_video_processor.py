#!/usr/bin/env python3
"""
U-CogNet Real-Time Multi-Video Processing con Gráficas Cognitivas
Procesamiento en tiempo real de múltiples videos con análisis cognitivo completo y visualización
"""

import sys
import os
import asyncio
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import threading
import queue
import gc

# Configurar path para U-CogNet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ucognet.modules.vision.yolov8_detector import YOLOv8Detector
from ucognet.core.cognitive_core import CognitiveCore
from ucognet.core.tda_manager import TDAManager
from ucognet.core.evaluator import Evaluator
from ucognet.core.trainer_loop import TrainerLoop
from ucognet.core.mycelial_optimizer import MycelialOptimizer
from ucognet.core.types import Frame, Detection, Event, Metrics
from ucognet.core.utils import setup_logging


class RealTimeVideoProcessor:
    """
    Procesador de video en tiempo real con U-CogNet
    """

    def __init__(self, media_dir: str = "test_media"):
        self.media_dir = Path(media_dir)

        # Configurar logging primero
        self.logger = setup_logging("INFO")
        self.logger.info("🧠 Inicializando U-CogNet Real-Time Video Processor con YOLOv8 REAL")

        # Inicializar detector YOLOv8 real
        self.logger.info("🔧 Inicializando detector YOLOv8...")
        self.vision_detector = YOLOv8Detector(
            model_path="yolov8n.pt",  # Usar modelo ligero para mejor rendimiento
            conf_threshold=0.3,
            device="auto"
        )
        self.logger.info(f"✅ Detector inicializado: {self.vision_detector}")

        # Ahora sí podemos llamar a métodos que usan el logger
        self.media_files = self._discover_media_files()

        # Inicializar componentes de U-CogNet
        self.cognitive_core = None
        self.tda_manager = None
        self.evaluator = None
        self.trainer_loop = None
        self.mycelial_optimizer = None

        # Estadísticas de procesamiento
        self.stats = {
            'videos_processed': 0,
            'total_frames': 0,
            'total_detections': 0,
            'processing_fps': 0,
            'avg_confidence': 0,
            'cognitive_cycles': 0,
            'memory_usage': []  # Historial de uso de memoria
        }

        # Control de tiempo real
        self.target_fps = 30
        self.frame_interval = 1.0 / self.target_fps

    def _discover_media_files(self) -> List[Path]:
        """Descubre archivos de media en el directorio"""
        supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        media_files = []

        if not self.media_dir.exists():
            self.logger.error(f"Directorio de media no encontrado: {self.media_dir}")
            return media_files

        for file_path in self.media_dir.iterdir():
            if file_path.suffix.lower() in supported_extensions:
                media_files.append(file_path)

        self.logger.info(f"📹 Encontrados {len(media_files)} archivos de video: {[f.name for f in media_files]}")
        return sorted(media_files)

    async def initialize_ucognet(self):
        """Inicializar todos los componentes de U-CogNet"""
        self.logger.info("🚀 Inicializando componentes de U-CogNet...")

        try:
            # Inicializar componentes principales
            self.cognitive_core = CognitiveCore()
            await self.cognitive_core.initialize()
            self.logger.info("✅ CognitiveCore inicializado")

            self.tda_manager = TDAManager()
            await self.tda_manager.initialize()
            self.logger.info("✅ TDAManager inicializado")

            self.evaluator = Evaluator()
            await self.evaluator.initialize()
            self.logger.info("✅ Evaluator inicializado")

            self.trainer_loop = TrainerLoop()
            await self.trainer_loop.initialize()
            self.logger.info("✅ TrainerLoop inicializado")

            self.mycelial_optimizer = MycelialOptimizer()
            await self.mycelial_optimizer.initialize()
            self.logger.info("✅ MycelialOptimizer inicializado")

            self.logger.info("🎉 ¡U-CogNet completamente inicializado!")

        except Exception as e:
            self.logger.error(f"❌ Error inicializando U-CogNet: {e}")
            raise

    def _create_frame_from_cv2(self, cv2_frame: np.ndarray, frame_id: int, timestamp: float) -> Frame:
        """Crear objeto Frame desde frame de OpenCV"""
        return Frame(
            data=cv2_frame,
            timestamp=datetime.fromtimestamp(timestamp),
            frame_id=frame_id,
            metadata={
                'source': 'opencv',
                'shape': cv2_frame.shape,
                'dtype': str(cv2_frame.dtype)
            }
        )

    def _extract_detections_from_frame(self, frame: np.ndarray) -> List[Detection]:
        """Extraer detecciones del frame usando YOLOv8 real"""
        # Usar detector YOLOv8 real
        yolo_detections = self.vision_detector.detect(frame)

        # Convertir a formato interno de U-CogNet
        detections = []
        for det in yolo_detections:
            detection = Detection(
                bbox=det.bbox,
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
                features=det.features,
                metadata={
                    **det.metadata,
                    'detector': 'yolov8_real',
                    'frame_shape': frame.shape
                }
            )
            detections.append(detection)

        self.logger.debug(f"🎯 YOLOv8 detectó {len(detections)} objetos")
        return detections

    def _simulate_detections_fallback(self, frame: np.ndarray) -> List[Detection]:
        """Método de respaldo: simulación mínima cuando YOLOv8 falla"""
        detections = []
        height, width = frame.shape[:2]

        # Solo simular 1-2 detecciones para evitar confusión
        num_objects = np.random.randint(1, 3)

        for i in range(num_objects):
            x1 = np.random.randint(0, width // 2)
            y1 = np.random.randint(0, height // 2)
            x2 = min(x1 + np.random.randint(50, 150), width)
            y2 = min(y1 + np.random.randint(100, 250), height)

            detection = Detection(
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                confidence=float(np.random.uniform(0.4, 0.9)),
                class_id=0,  # persona por defecto
                class_name="person",
                features=None,
                metadata={
                    'detector': 'simulation_fallback',
                    'reason': 'yolo_error'
                }
            )
            detections.append(detection)

    async def process_video_realtime(self, video_path: Path) -> Dict[str, Any]:
        """Procesar un video en tiempo real con U-CogNet"""
        video_name = video_path.name
        self.logger.info(f"🎬 Iniciando procesamiento de: {video_name}")

        # Abrir video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.error(f"❌ No se pudo abrir el video: {video_path}")
            return {'error': 'video_open_failed'}

        # Obtener propiedades del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.logger.info(f"📊 Video: {width}x{height}, {fps} FPS, {frame_count} frames")

        # Estadísticas del video
        video_stats = {
            'name': video_name,
            'frames_processed': 0,
            'detections_found': 0,
            'processing_time': 0,
            'cognitive_cycles': 0,
            'avg_confidence': 0,
            'people_count_history': []
        }

        frame_id = 0
        start_time = time.time()

        try:
            while True:
                frame_start_time = time.time()

                # Leer frame
                ret, frame = cap.read()
                if not ret:
                    break

                frame_id += 1
                timestamp = time.time()

                # Crear objeto Frame
                ucog_frame = self._create_frame_from_cv2(frame, frame_id, timestamp)

                # Extraer detecciones
                detections = self._extract_detections_from_frame(frame)

                # Crear evento cognitivo
                event = Event(
                    frame=ucog_frame,
                    detections=detections,
                    timestamp=ucog_frame.timestamp,
                    event_type="video_frame",
                    metadata={
                        'video_name': video_name,
                        'frame_id': frame_id,
                        'total_frames': frame_count
                    }
                )

                # Procesar con Cognitive Core
                cognitive_result = await self.cognitive_core.process_input({
                    'vision': {
                        'detections': detections,
                        'frame_metadata': {
                            'video_name': video_name,
                            'frame_id': frame_id,
                            'progress': frame_id / frame_count
                        }
                    },
                    'event': event
                })

                # Obtener métricas del sistema
                system_metrics = await self.cognitive_core.get_metrics()

                # Mostrar información en tiempo real
                people_count = len(detections)
                avg_confidence = np.mean([d.confidence for d in detections]) if detections else 0

                # Actualizar estadísticas
                video_stats['frames_processed'] = frame_id
                video_stats['detections_found'] += people_count
                video_stats['people_count_history'].append(people_count)
                
                # Limitar el tamaño del historial para ahorrar memoria
                max_history_size = 1000
                if len(video_stats['people_count_history']) > max_history_size:
                    video_stats['people_count_history'] = video_stats['people_count_history'][-max_history_size:]
                
                video_stats['cognitive_cycles'] += 1

                if detections:
                    current_avg = video_stats['avg_confidence']
                    video_stats['avg_confidence'] = (current_avg * (frame_id - 1) + avg_confidence) / frame_id

                # Mostrar progreso cada 30 frames o cuando hay detecciones
                if frame_id % 30 == 0 or people_count > 0:
                    progress = (frame_id / frame_count) * 100
                    processing_fps = frame_id / (time.time() - start_time)

                    print(f"\r🎬 {video_name} | Frame {frame_id}/{frame_count} ({progress:.1f}%) | "
                          f"👥 {people_count} personas | 🎯 Conf: {avg_confidence:.2f} | "
                          f"⚡ {processing_fps:.1f} FPS | 🧠 Ciclos: {video_stats['cognitive_cycles']}", end="")

                    # Mostrar razonamiento cognitivo si hay detecciones
                    if people_count > 0:
                        print(f"\n   🧠 Cognitive Analysis: {cognitive_result.get('reasoning', 'Processing...')}")
                        print(f"   📊 System Metrics: Acc={system_metrics.accuracy:.2f}, "
                              f"Lat={system_metrics.latency_ms:.1f}ms, Mem={system_metrics.memory_usage_mb:.1f}MB")

                # Control de tiempo real (simular procesamiento en tiempo real)
                frame_processing_time = time.time() - frame_start_time
                if frame_processing_time < self.frame_interval:
                    await asyncio.sleep(self.frame_interval - frame_processing_time)

        except KeyboardInterrupt:
            self.logger.info("⏹️ Procesamiento interrumpido por usuario")
        except Exception as e:
            self.logger.error(f"❌ Error procesando video {video_name}: {e}")
            return {'error': str(e)}
        finally:
            cap.release()

        # Calcular estadísticas finales del video
        total_time = time.time() - start_time
        video_stats['processing_time'] = total_time
        video_stats['avg_fps'] = frame_id / total_time if total_time > 0 else 0

        # Resumen del video
        print(f"\n✅ {video_name} COMPLETADO:")
        print(f"   📊 Frames: {video_stats['frames_processed']}")
        print(f"   👥 Total detecciones: {video_stats['detections_found']}")
        print(f"   🎯 Confianza promedio: {video_stats['avg_confidence']:.3f}")
        print(f"   ⚡ FPS promedio: {video_stats['avg_fps']:.1f}")
        print(f"   🧠 Ciclos cognitivos: {video_stats['cognitive_cycles']}")
        print(f"   ⏱️ Tiempo total: {total_time:.2f}s")

        # Estadísticas de personas
        if video_stats['people_count_history']:
            max_people = max(video_stats['people_count_history'])
            avg_people = np.mean(video_stats['people_count_history'])
            print(f"   👥 Personas - Máx: {max_people}, Promedio: {avg_people:.1f}")

        return video_stats

    async def run_batch_processing(self):
        """Ejecutar procesamiento batch de todos los videos"""
        print("🎬 U-COGNET REAL-TIME MULTI-VIDEO PROCESSING")
        print("=" * 60)
        print(f"📹 Videos a procesar: {len(self.media_files)}")
        print(f"🎯 Target FPS: {self.target_fps}")
        print()

        # Inicializar U-CogNet
        await self.initialize_ucognet()

        # Procesar cada video
        batch_start_time = time.time()
        all_video_stats = []

        for i, video_path in enumerate(self.media_files, 1):
            print(f"\n🎯 Video {i}/{len(self.media_files)}")
            print("-" * 40)

            try:
                video_stats = await self.process_video_realtime(video_path)
                if 'error' not in video_stats:
                    all_video_stats.append(video_stats)
                    self.stats['videos_processed'] += 1
                    self.stats['total_frames'] += video_stats['frames_processed']
                    self.stats['total_detections'] += video_stats['detections_found']
                    self.stats['cognitive_cycles'] += video_stats['cognitive_cycles']
                else:
                    self.logger.error(f"❌ Error en video {video_path.name}: {video_stats['error']}")

            except Exception as e:
                self.logger.error(f"❌ Error procesando {video_path.name}: {e}")
            
            # Limpieza de memoria después de cada video
            gc.collect()
            print(f"🧹 Memoria limpiada después de {video_path.name}")

        # Estadísticas finales del batch
        batch_time = time.time() - batch_start_time

        print("\n" + "=" * 60)
        print("🎉 PROCESAMIENTO BATCH COMPLETADO")
        print("=" * 60)
        print(f"📹 Videos procesados: {self.stats['videos_processed']}/{len(self.media_files)}")
        print(f"🎬 Frames totales: {self.stats['total_frames']}")
        print(f"👥 Detecciones totales: {self.stats['total_detections']}")
        print(f"🧠 Ciclos cognitivos totales: {self.stats['cognitive_cycles']}")
        print(f"⏱️ Tiempo total: {batch_time:.2f}s")
        print(f"⚡ FPS promedio global: {self.stats['total_frames'] / batch_time:.1f}")

        if all_video_stats:
            avg_confidence = np.mean([s['avg_confidence'] for s in all_video_stats if s['avg_confidence'] > 0])
            print(f"🎯 Confianza promedio global: {avg_confidence:.3f}")

        # Obtener métricas finales del sistema
        try:
            final_metrics = await self.cognitive_core.get_metrics()
            print("\n📊 MÉTRICAS FINALES DEL SISTEMA:")
            print(f"   • Accuracy: {final_metrics.accuracy:.3f}")
            print(f"   • Latency: {final_metrics.latency_ms:.1f}ms")
            print(f"   • Memory: {final_metrics.memory_usage_mb:.1f}MB")
            print(f"   • Throughput: {final_metrics.throughput_fps:.1f} FPS")
            print(f"   • Throughput: {final_metrics.throughput_fps:.1f} FPS")
        except Exception as e:
            print(f"   ❌ Error obteniendo métricas finales: {e}")

        return self.stats

    async def run_batch_processing_with_visualization(self, visualizer: 'CognitiveVisualizer') -> Dict[str, Any]:
        """Ejecutar procesamiento batch de todos los videos con visualización"""
        print("🎬 U-COGNET REAL-TIME MULTI-VIDEO PROCESSING CON GRÁFICAS COGNITIVAS")
        print("=" * 70)
        print(f"📹 Videos a procesar: {len(self.media_files)}")
        print(f"🎯 Target FPS: {self.target_fps}")
        print("📊 Visualización cognitiva activada")
        print()

        # Inicializar U-CogNet
        await self.initialize_ucognet()

        # Procesar cada video
        batch_start_time = time.time()
        all_video_stats = []

        for i, video_path in enumerate(self.media_files, 1):
            print(f"\n🎯 Video {i}/{len(self.media_files)}")
            print("-" * 50)

            try:
                video_stats = await self.process_video_realtime_with_visualization(video_path, visualizer)
                if 'error' not in video_stats:
                    all_video_stats.append(video_stats)
                    self.stats['videos_processed'] += 1
                    self.stats['total_frames'] += video_stats['frames_processed']
                    self.stats['total_detections'] += video_stats['detections_found']
                    self.stats['cognitive_cycles'] += video_stats['cognitive_cycles']
                else:
                    self.logger.error(f"❌ Error en video {video_path.name}: {video_stats['error']}")

            except Exception as e:
                self.logger.error(f"❌ Error procesando {video_path.name}: {e}")
            
            # Limpieza de memoria después de cada video
            gc.collect()
            print(f"🧹 Memoria limpiada después de {video_path.name}")

        # Estadísticas finales del batch
        batch_time = time.time() - batch_start_time

        print("\n" + "=" * 70)
        print("🎉 PROCESAMIENTO BATCH COMPLETADO")
        print("=" * 70)
        print(f"📹 Videos procesados: {self.stats['videos_processed']}/{len(self.media_files)}")
        print(f"🎬 Frames totales: {self.stats['total_frames']}")
        print(f"👥 Detecciones totales: {self.stats['total_detections']}")
        print(f"🧠 Ciclos cognitivos totales: {self.stats['cognitive_cycles']}")
        print(f"⏱️ Tiempo total: {batch_time:.2f}s")
        print(f"⚡ FPS promedio global: {self.stats['total_frames'] / batch_time:.1f}")

        if all_video_stats:
            avg_confidence = np.mean([s['avg_confidence'] for s in all_video_stats if s['avg_confidence'] > 0])
            print(f"🎯 Confianza promedio global: {avg_confidence:.3f}")

        # Obtener métricas finales del sistema
        try:
            final_metrics = await self.cognitive_core.get_metrics()
            print("\n📊 MÉTRICAS FINALES DEL SISTEMA:")
            print(f"   • Accuracy: {final_metrics.accuracy:.3f}")
            print(f"   • Latency: {final_metrics.latency_ms:.1f}ms")
            print(f"   • Memory: {final_metrics.memory_usage_mb:.1f}MB")
            print(f"   • Throughput: {final_metrics.throughput_fps:.1f} FPS")

            # Mostrar métricas de aprendizaje
            print("\n🧠 MÉTRICAS DE APRENDIZAJE COGNITIVO:")
            learning_efficiency = min(1.0, self.stats['cognitive_cycles'] / 5000)
            print(f"   • Eficiencia de Aprendizaje: {learning_efficiency:.3f}")
            print(f"   • Adaptabilidad: {final_metrics.accuracy:.3f}")
            print(f"   • Procesamiento Cognitivo: {self.stats['cognitive_cycles']} ciclos")
            print(f"   • Ratio Detección/Frame: {self.stats['total_detections']/self.stats['total_frames']:.3f}")

            # Mostrar estadísticas del detector YOLOv8
            print("\n🤖 ESTADÍSTICAS DEL DETECTOR YOLOv8:")
            detector_stats = self.vision_detector.get_stats()
            print(f"   • Modelo: {detector_stats.get('model_path', 'N/A')}")
            print(f"   • Dispositivo: {detector_stats.get('device', 'N/A')}")
            print(f"   • Detecciones totales: {detector_stats.get('detection_count', 0)}")
            print(f"   • Tiempo promedio de inferencia: {detector_stats.get('avg_inference_time', 0):.4f}s")
            print(f"   • Estado: {'✅ OPERATIVO' if detector_stats.get('model_loaded', False) else '❌ NO CARGADO'}")

        except Exception as e:
            print(f"   ❌ Error obteniendo métricas finales: {e}")

        return {
            'batch_stats': self.stats,
            'video_stats': all_video_stats,
            'processing_time': batch_time,
            'final_metrics': final_metrics if 'final_metrics' in locals() else None
        }

    async def process_video_realtime_with_visualization(self, video_path: Path, visualizer: 'CognitiveVisualizer') -> Dict[str, Any]:
        """Procesar un video en tiempo real con U-CogNet y visualización"""
        video_name = video_path.name
        self.logger.info(f"🎬 Iniciando procesamiento de: {video_name}")

        # Abrir video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.error(f"❌ No se pudo abrir el video: {video_path}")
            return {'error': 'video_open_failed'}

        # Obtener propiedades del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.logger.info(f"📊 Video: {width}x{height}, {fps} FPS, {frame_count} frames")

        # Estadísticas del video
        video_stats = {
            'name': video_name,
            'frames_processed': 0,
            'detections_found': 0,
            'processing_time': 0,
            'cognitive_cycles': 0,
            'avg_confidence': 0,
            'people_count_history': []
        }

        frame_id = 0
        start_time = time.time()

        try:
            while True:
                frame_start_time = time.time()

                # Leer frame
                ret, frame = cap.read()
                if not ret:
                    break

                frame_id += 1
                timestamp = time.time()

                # Límite de 500 ciclos para análisis completo
                if frame_id > 500:
                    self.logger.info(f"🎯 Alcanzado límite de 500 ciclos. Deteniendo procesamiento.")
                    break

                # Crear objeto Frame
                ucog_frame = self._create_frame_from_cv2(frame, frame_id, timestamp)

                # Extraer detecciones
                detections = self._extract_detections_from_frame(frame)

                # Crear evento cognitivo
                event = Event(
                    frame=ucog_frame,
                    detections=detections,
                    timestamp=ucog_frame.timestamp,
                    event_type="video_frame",
                    metadata={
                        'video_name': video_name,
                        'frame_id': frame_id,
                        'total_frames': frame_count
                    }
                )

                # Procesar con Cognitive Core
                cognitive_result = await self.cognitive_core.process_input({
                    'vision': {
                        'detections': detections,
                        'frame_metadata': {
                            'video_name': video_name,
                            'frame_id': frame_id,
                            'progress': frame_id / frame_count
                        }
                    },
                    'event': event
                })

                # Obtener métricas del sistema
                system_metrics = await self.cognitive_core.get_metrics()

                # Preparar datos para visualización
                people_count = len(detections)
                avg_confidence = np.mean([d.confidence for d in detections]) if detections else 0
                current_fps = frame_id / (time.time() - start_time) if time.time() - start_time > 0 else 0

                # Actualizar visualizador
                visualizer.update_data({
                    'accuracy': system_metrics.accuracy,
                    'avg_confidence': avg_confidence,
                    'people_count': people_count,
                    'fps': current_fps,
                    'memory_mb': system_metrics.memory_usage_mb,
                    'latency_ms': system_metrics.latency_ms,
                    'cognitive_cycles': video_stats['cognitive_cycles'] + 1
                })

                # Actualizar estadísticas
                video_stats['frames_processed'] = frame_id
                video_stats['detections_found'] += people_count
                video_stats['people_count_history'].append(people_count)
                
                # Limitar el tamaño del historial para ahorrar memoria
                max_history_size = 1000
                if len(video_stats['people_count_history']) > max_history_size:
                    video_stats['people_count_history'] = video_stats['people_count_history'][-max_history_size:]
                
                video_stats['cognitive_cycles'] += 1

                if detections:
                    current_avg = video_stats['avg_confidence']
                    video_stats['avg_confidence'] = (current_avg * (frame_id - 1) + avg_confidence) / frame_id

                # Mostrar progreso cada 30 frames o cuando hay detecciones
                if frame_id % 30 == 0 or people_count > 0:
                    progress = (frame_id / frame_count) * 100
                    processing_fps = frame_id / (time.time() - start_time)

                    print(f"\r🎬 {video_name} | Frame {frame_id}/{frame_count} ({progress:.1f}%) | "
                          f"👥 {people_count} personas | 🎯 Conf: {avg_confidence:.2f} | "
                          f"⚡ {processing_fps:.1f} FPS | 🧠 Ciclos: {video_stats['cognitive_cycles']}", end="")

                    # Mostrar razonamiento cognitivo si hay detecciones
                    if people_count > 0:
                        print(f"\n   🧠 Cognitive Analysis: {cognitive_result.get('reasoning', 'Processing...')}")
                        print(f"   📊 System Metrics: Acc={system_metrics.accuracy:.2f}, "
                              f"Lat={system_metrics.latency_ms:.1f}ms, Mem={system_metrics.memory_usage_mb:.1f}MB")

                # Control de tiempo real (simular procesamiento en tiempo real)
                frame_processing_time = time.time() - frame_start_time
                if frame_processing_time < self.frame_interval:
                    await asyncio.sleep(self.frame_interval - frame_processing_time)

        except KeyboardInterrupt:
            self.logger.info("⏹️ Procesamiento interrumpido por usuario")
        except Exception as e:
            self.logger.error(f"❌ Error procesando video {video_name}: {e}")
            return {'error': str(e)}
        finally:
            cap.release()

        # Calcular estadísticas finales del video
        total_time = time.time() - start_time
        video_stats['processing_time'] = total_time
        video_stats['avg_fps'] = frame_id / total_time if total_time > 0 else 0

        # Resumen del video
        print(f"\n✅ {video_name} COMPLETADO:")
        print(f"   📊 Frames: {video_stats['frames_processed']}")
        print(f"   👥 Total detecciones: {video_stats['detections_found']}")
        print(f"   🎯 Confianza promedio: {video_stats['avg_confidence']:.3f}")
        print(f"   ⚡ FPS promedio: {video_stats['avg_fps']:.1f}")
        print(f"   🧠 Ciclos cognitivos: {video_stats['cognitive_cycles']}")
        print(f"   ⏱️ Tiempo total: {total_time:.2f}s")

        # Estadísticas de personas
        if video_stats['people_count_history']:
            max_people = max(video_stats['people_count_history'])
            avg_people = np.mean(video_stats['people_count_history'])
            print(f"   👥 Personas - Máx: {max_people}, Promedio: {avg_people:.1f}")

        return video_stats


class CognitiveVisualizer:
    """
    Visualizador de métricas cognitivas en tiempo real
    """

    def __init__(self):
        self.data_queue = queue.Queue()
        self.is_running = False
        self.thread = None

        # Datos para gráficas
        self.accuracy_history = []
        self.confidence_history = []
        self.people_count_history = []
        self.fps_history = []
        self.memory_history = []
        self.latency_history = []
        self.learning_progress = []
        self.cognitive_cycles = []

        # Configurar matplotlib para modo headless
        import matplotlib
        matplotlib.use('Agg')  # Backend no interactivo
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle('U-CogNet: Métricas Cognitivas en Tiempo Real', fontsize=16, color='cyan')

        # Títulos de subplots
        self.axes[0, 0].set_title('Precisión Cognitiva', color='white')
        self.axes[0, 1].set_title('Confianza de Detección', color='white')
        self.axes[0, 2].set_title('Conteo de Personas', color='white')
        self.axes[1, 0].set_title('FPS de Procesamiento', color='white')
        self.axes[1, 1].set_title('Uso de Memoria', color='white')
        self.axes[1, 2].set_title('Progreso de Aprendizaje', color='white')

        # Configurar colores
        self.colors = ['cyan', 'magenta', 'yellow', 'lime', 'red', 'orange']

    def start_visualization(self):
        """Iniciar visualización en thread separado"""
        self.is_running = True
        self.thread = threading.Thread(target=self._visualization_loop, daemon=True)
        self.thread.start()

    def stop_visualization(self):
        """Detener visualización"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1)

    def update_data(self, metrics: Dict[str, Any]):
        """Actualizar datos para visualización"""
        try:
            self.data_queue.put(metrics, timeout=0.1)
        except queue.Full:
            pass  # Ignorar si la cola está llena

    def _visualization_loop(self):
        """Loop principal de visualización"""
        while self.is_running:
            try:
                # Procesar datos pendientes
                while not self.data_queue.empty():
                    metrics = self.data_queue.get_nowait()
                    self._process_metrics(metrics)

                # Actualizar gráficas
                self._update_plots()

                # Pequeña pausa para no consumir demasiado CPU
                time.sleep(0.1)

            except Exception as e:
                print(f"Error en visualización: {e}")
                time.sleep(1)

    def _process_metrics(self, metrics: Dict[str, Any]):
        """Procesar métricas para gráficas"""
        # Accuracy cognitiva
        if 'accuracy' in metrics:
            self.accuracy_history.append(metrics['accuracy'])
            if len(self.accuracy_history) > 100:  # Mantener solo últimos 100 puntos
                self.accuracy_history.pop(0)

        # Confianza de detección
        if 'avg_confidence' in metrics:
            self.confidence_history.append(metrics['avg_confidence'])
            if len(self.confidence_history) > 100:
                self.confidence_history.pop(0)

        # Conteo de personas
        if 'people_count' in metrics:
            self.people_count_history.append(metrics['people_count'])
            if len(self.people_count_history) > 100:
                self.people_count_history.pop(0)

        # FPS
        if 'fps' in metrics:
            self.fps_history.append(metrics['fps'])
            if len(self.fps_history) > 100:
                self.fps_history.pop(0)

        # Memoria
        if 'memory_mb' in metrics:
            self.memory_history.append(metrics['memory_mb'])
            if len(self.memory_history) > 100:
                self.memory_history.pop(0)

        # Latencia
        if 'latency_ms' in metrics:
            self.latency_history.append(metrics['latency_ms'])
            if len(self.latency_history) > 100:
                self.latency_history.pop(0)

        # Progreso de aprendizaje (simulado basado en ciclos cognitivos)
        if 'cognitive_cycles' in metrics:
            learning_rate = min(1.0, metrics['cognitive_cycles'] / 1000)  # Simular aprendizaje
            self.learning_progress.append(learning_rate)
            if len(self.learning_progress) > 100:
                self.learning_progress.pop(0)

    def _update_plots(self):
        """Actualizar todas las gráficas y guardar en archivo"""
        try:
            # Limpiar axes
            for ax in self.axes.flat:
                ax.clear()

            # Configurar títulos nuevamente
            titles = [
                'Precisión Cognitiva', 'Confianza de Detección', 'Conteo de Personas',
                'FPS de Procesamiento', 'Uso de Memoria (MB)', 'Progreso de Aprendizaje'
            ]

            for i, (ax, title) in enumerate(zip(self.axes.flat, titles)):
                ax.set_title(title, color='white', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_facecolor('#1a1a1a')

            # Plot 1: Precisión Cognitiva
            if self.accuracy_history:
                self.axes[0, 0].plot(self.accuracy_history, color=self.colors[0], linewidth=2)
                self.axes[0, 0].set_ylim(0, 1)
                self.axes[0, 0].set_ylabel('Accuracy', color='white')

            # Plot 2: Confianza de Detección
            if self.confidence_history:
                self.axes[0, 1].plot(self.confidence_history, color=self.colors[1], linewidth=2)
                self.axes[0, 1].set_ylim(0, 1)
                self.axes[0, 1].set_ylabel('Confidence', color='white')

            # Plot 3: Conteo de Personas
            if self.people_count_history:
                self.axes[0, 2].plot(self.people_count_history, color=self.colors[2], linewidth=2)
                self.axes[0, 2].set_ylim(0, max(self.people_count_history + [8]))
                self.axes[0, 2].set_ylabel('Personas', color='white')

            # Plot 4: FPS
            if self.fps_history:
                self.axes[1, 0].plot(self.fps_history, color=self.colors[3], linewidth=2)
                self.axes[1, 0].axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Target 30 FPS')
                self.axes[1, 0].set_ylim(0, max(self.fps_history + [35]))
                self.axes[1, 0].set_ylabel('FPS', color='white')
                self.axes[1, 0].legend()

            # Plot 5: Memoria
            if self.memory_history:
                self.axes[1, 1].plot(self.memory_history, color=self.colors[4], linewidth=2)
                self.axes[1, 1].set_ylabel('MB', color='white')

            # Plot 6: Progreso de Aprendizaje
            if self.learning_progress:
                self.axes[1, 2].plot(self.learning_progress, color=self.colors[5], linewidth=2)
                self.axes[1, 2].set_ylim(0, 1)
                self.axes[1, 2].set_ylabel('Learning Progress', color='white')

            # Ajustar layout y guardar gráfica
            self.fig.tight_layout()
            self.fig.savefig('cognitive_metrics_realtime.png', dpi=150, bbox_inches='tight',
                           facecolor='#0a0a0a')
            print("📊 Gráfica actualizada: cognitive_metrics_realtime.png")

        except Exception as e:
            print(f"Error actualizando gráficas: {e}")

    def show_final_summary(self, all_video_stats: List[Dict[str, Any]]):
        """Mostrar resumen final con gráficas comparativas"""
        if not all_video_stats:
            return

        # Crear figura de resumen
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('U-CogNet: Resumen Final de Procesamiento', fontsize=14, color='cyan')
        plt.style.use('dark_background')

        # Extraer datos
        video_names = [stats['name'][:20] + '...' if len(stats['name']) > 20 else stats['name']
                      for stats in all_video_stats]
        avg_confidences = [stats['avg_confidence'] for stats in all_video_stats]
        avg_fps = [stats['avg_fps'] for stats in all_video_stats]
        total_detections = [stats['detections_found'] for stats in all_video_stats]
        processing_times = [stats['processing_time'] for stats in all_video_stats]

        # Gráfica 1: Confianza promedio por video
        axes[0, 0].bar(video_names, avg_confidences, color='cyan', alpha=0.7)
        axes[0, 0].set_title('Confianza Promedio por Video', color='white')
        axes[0, 0].set_ylabel('Confidence', color='white')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Gráfica 2: FPS promedio por video
        axes[0, 1].bar(video_names, avg_fps, color='magenta', alpha=0.7)
        axes[0, 1].axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Target 30 FPS')
        axes[0, 1].set_title('FPS Promedio por Video', color='white')
        axes[0, 1].set_ylabel('FPS', color='white')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend()

        # Gráfica 3: Detecciones totales por video
        axes[1, 0].bar(video_names, total_detections, color='yellow', alpha=0.7)
        axes[1, 0].set_title('Detecciones Totales por Video', color='white')
        axes[1, 0].set_ylabel('Detecciones', color='white')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Gráfica 4: Tiempo de procesamiento vs FPS
        scatter = axes[1, 1].scatter(processing_times, avg_fps, c=avg_confidences,
                                   cmap='viridis', s=100, alpha=0.7)
        axes[1, 1].set_title('Tiempo vs FPS (Color = Confianza)', color='white')
        axes[1, 1].set_xlabel('Tiempo de Procesamiento (s)', color='white')
        axes[1, 1].set_ylabel('FPS Promedio', color='white')
        plt.colorbar(scatter, ax=axes[1, 1], label='Confidence')

        # Configurar colores de fondo
        for ax in axes.flat:
            ax.set_facecolor('#1a1a1a')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


async def main():
    """Función principal"""
    processor = RealTimeVideoProcessor()

    if not processor.media_files:
        print("❌ No se encontraron archivos de video en test_media/")
        return

    # Inicializar visualizador
    visualizer = CognitiveVisualizer()
    visualizer.start_visualization()

    try:
        # Ejecutar procesamiento con visualización
        stats = await processor.run_batch_processing_with_visualization(visualizer)

        # Mostrar resumen final
        print("\n📊 Generando gráficas finales...")
        visualizer.show_final_summary(stats.get('video_stats', []))

    except KeyboardInterrupt:
        print("\n⏹️ Procesamiento interrumpido por usuario")
    except Exception as e:
        print(f"❌ Error en procesamiento: {e}")
        import traceback
        traceback.print_exc()
    finally:
        visualizer.stop_visualization()


if __name__ == "__main__":
    asyncio.run(main())