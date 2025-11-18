"""
Extractor de audio para U-CogNet.
Extrae audio de videos para procesamiento multimodal.
"""

import os
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import moviepy as mp
from ..common.types import AudioData
from ..common.logging import logger

class AudioExtractor:
    """
    Extrae audio de videos para análisis multimodal.
    Parte del pipeline de procesamiento de audio.
    """

    def __init__(self, output_dir: str = "audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"AudioExtractor inicializado con directorio: {output_dir}")

    def extract_from_video(self, video_path: str, output_filename: Optional[str] = None) -> Optional[str]:
        """
        Extrae audio de un video y lo guarda como MP3.

        Args:
            video_path: Ruta al video fuente
            output_filename: Nombre del archivo de salida (opcional)

        Returns:
            Ruta al archivo de audio extraído, o None si falla
        """
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                logger.error(f"Video no encontrado: {video_path}")
                return None

            # Generar nombre de salida si no se proporciona
            if output_filename is None:
                output_filename = f"{video_path.stem}_audio.mp3"

            output_path = self.output_dir / output_filename

            logger.info(f"Procesando video: {video_path}")

            # Cargar el video
            video = mp.VideoFileClip(str(video_path))

            # Verificar si tiene audio
            if video.audio is None:
                logger.warning(f"El video no contiene audio: {video_path}")
                video.close()
                return None

            # Extraer el audio
            audio = video.audio

            # Guardar como MP3
            audio.write_audiofile(str(output_path), verbose=False, logger=None)

            # Liberar recursos
            video.close()
            audio.close()

            logger.info(f"Audio extraído exitosamente: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error extrayendo audio de {video_path}: {str(e)}")
            return None

    def extract_audio_data(self, video_path: str) -> Optional[AudioData]:
        """
        Extrae datos de audio crudos para procesamiento inmediato.

        Returns:
            AudioData con waveform y metadata, o None si falla
        """
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                logger.error(f"Video no encontrado: {video_path}")
                return None

            logger.info(f"Extrayendo datos de audio de: {video_path}")

            # Cargar video
            video = mp.VideoFileClip(str(video_path))

            if video.audio is None:
                logger.warning(f"El video no contiene audio: {video_path}")
                video.close()
                return None

            # Obtener datos de audio
            audio = video.audio
            waveform = audio.to_soundarray(fps=44100)  # Convertir a array numpy

            # Metadata
            duration = audio.duration
            sample_rate = 44100

            # Crear AudioData
            audio_data = AudioData(
                waveform=waveform,
                sample_rate=sample_rate,
                duration=duration,
                source=str(video_path),
                timestamp=float(video.start) if hasattr(video, 'start') else 0.0
            )

            # Liberar recursos
            video.close()
            audio.close()

            logger.info(f"Datos de audio extraídos: {duration:.2f}s, {waveform.shape}")
            return audio_data

        except Exception as e:
            logger.error(f"Error extrayendo datos de audio de {video_path}: {str(e)}")
            return None

    def batch_extract(self, video_paths: list, prefix: str = "batch") -> list:
        """
        Extrae audio de múltiples videos.

        Args:
            video_paths: Lista de rutas a videos
            prefix: Prefijo para nombres de archivos de salida

        Returns:
            Lista de rutas a archivos de audio extraídos
        """
        results = []
        for i, video_path in enumerate(video_paths):
            output_filename = f"{prefix}_{i:03d}_{Path(video_path).stem}.mp3"
            result = self.extract_from_video(video_path, output_filename)
            if result:
                results.append(result)

        logger.info(f"Extracción batch completada: {len(results)}/{len(video_paths)} exitosas")
        return results