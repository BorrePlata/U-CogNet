"""
Manejador de audio multimodal para U-CogNet.
Integra extracción, codificación y análisis de audio con el pipeline cognitivo.
"""

import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path
from .audio_extractor import AudioExtractor
from .audio_encoder import AudioEncoder
from ..common.audio_types import AudioData, AudioEvent, MultimodalEvent
from ..common.types import Event as VisualEvent, Context
from ..common.logging import logger

class AudioHandler:
    """
    Manejador multimodal de audio.
    Coordina extracción, codificación y análisis de audio con visión.
    """

    def __init__(self, embedding_dim: int = 512, sample_rate: int = 44100):
        self.extractor = AudioExtractor()
        self.encoder = AudioEncoder(embedding_dim=embedding_dim, sample_rate=sample_rate)
        self.audio_embeddings = []  # Historial de embeddings
        self.audio_events = []      # Historial de eventos de audio
        logger.info("AudioHandler multimodal inicializado")

    def process_video_audio(self, video_path: str) -> Optional[Tuple[np.ndarray, List[AudioEvent]]]:
        """
        Procesa audio de un video completo.

        Args:
            video_path: Ruta al video

        Returns:
            Tupla de (embedding_global, lista_eventos) o None si falla
        """
        try:
            # Extraer datos de audio
            audio_data = self.extractor.extract_audio_data(video_path)
            if audio_data is None:
                return None

            # Procesar con encoder
            embedding, events = self.encoder.process_audio(audio_data)

            # Almacenar en historial
            self.audio_embeddings.append(embedding)
            self.audio_events.extend(events)

            logger.info(f"Audio de video procesado: {len(events)} eventos detectados")
            return embedding, events

        except Exception as e:
            logger.error(f"Error procesando audio de video {video_path}: {str(e)}")
            return None

    def correlate_with_vision(self, visual_events: List[VisualEvent],
                            audio_events: List[AudioEvent],
                            timestamp: float) -> MultimodalEvent:
        """
        Correlaciona eventos visuales con eventos de audio.

        Args:
            visual_events: Eventos detectados por visión
            audio_events: Eventos detectados por audio
            timestamp: Timestamp del evento multimodal

        Returns:
            MultimodalEvent con correlación
        """
        try:
            # Calcular correlación simple (pueden mejorarse con ML)
            correlation_score = self._calculate_correlation(visual_events, audio_events)

            # Generar contexto conjunto
            joint_context = self._generate_joint_context(visual_events, audio_events)

            multimodal_event = MultimodalEvent(
                visual_events=visual_events,
                audio_events=audio_events,
                timestamp=timestamp,
                correlation_score=correlation_score,
                joint_context=joint_context
            )

            logger.debug(f"Evento multimodal creado: correlación {correlation_score:.3f}")
            return multimodal_event

        except Exception as e:
            logger.error(f"Error correlacionando multimodal: {str(e)}")
            # Retornar evento básico
            return MultimodalEvent(
                visual_events=visual_events,
                audio_events=audio_events,
                timestamp=timestamp,
                correlation_score=0.0,
                joint_context="Error en correlación"
            )

    def _calculate_correlation(self, visual_events: List[VisualEvent],
                             audio_events: List[AudioEvent]) -> float:
        """
        Calcula score de correlación entre eventos visuales y auditivos.
        Implementación simple - puede mejorarse con modelos de atención cruzada.
        """
        if not visual_events or not audio_events:
            return 0.0

        # Correlación basada en timing (eventos simultáneos)
        visual_times = [ev.timestamp for ev in visual_events]
        audio_times = [ev.start_time for ev in audio_events]

        # Contar eventos que ocurren en ventanas de tiempo similares
        correlation_count = 0
        time_window = 1.0  # 1 segundo de tolerancia

        for v_time in visual_times:
            for a_time in audio_times:
                if abs(v_time - a_time) < time_window:
                    correlation_count += 1

        # Normalizar
        max_possible = min(len(visual_events), len(audio_events))
        correlation = correlation_count / max_possible if max_possible > 0 else 0.0

        return min(correlation, 1.0)  # Clamp a [0,1]

    def _generate_joint_context(self, visual_events: List[VisualEvent],
                              audio_events: List[AudioEvent]) -> str:
        """
        Genera descripción semántica conjunta de eventos visuales y auditivos.
        """
        visual_desc = []
        for ev in visual_events[:3]:  # Limitar a primeros 3
            if hasattr(ev, 'detections'):
                detections = [d.class_name for d in ev.detections]
                visual_desc.append(f"veo {', '.join(detections)}")
            else:
                visual_desc.append("evento visual")

        audio_desc = []
        for ev in audio_events[:3]:  # Limitar a primeros 3
            audio_desc.append(f"escucho {ev.event_type}")

        if visual_desc and audio_desc:
            return f"{' y '.join(visual_desc)} mientras {' y '.join(audio_desc)}"
        elif visual_desc:
            return f"{' y '.join(visual_desc)}"
        elif audio_desc:
            return f"{' y '.join(audio_desc)}"
        else:
            return "escena sin eventos detectados"

    def get_audio_context(self, window_seconds: float = 10.0) -> List[AudioEvent]:
        """
        Obtiene contexto de audio reciente.

        Args:
            window_seconds: Ventana de tiempo en segundos

        Returns:
            Lista de eventos de audio recientes
        """
        if not self.audio_events:
            return []

        # Filtrar eventos recientes (simplificado - en producción usar timestamps)
        recent_events = self.audio_events[-10:]  # Últimos 10 eventos
        return recent_events

    def get_multimodal_embedding(self, visual_embedding: np.ndarray,
                               audio_embedding: np.ndarray) -> np.ndarray:
        """
        Fusiona embeddings visuales y auditivos en un espacio común.

        Args:
            visual_embedding: Embedding visual
            audio_embedding: Embedding auditivo

        Returns:
            Embedding multimodal fusionado
        """
        try:
            # Fusión simple: concatenar y normalizar
            combined = np.concatenate([visual_embedding, audio_embedding])
            normalized = combined / (np.linalg.norm(combined) + 1e-8)

            logger.debug(f"Embedding multimodal fusionado: {normalized.shape}")
            return normalized

        except Exception as e:
            logger.error(f"Error fusionando embeddings: {str(e)}")
            return visual_embedding  # Fallback a visual solo