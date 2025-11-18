"""
Encoder de audio para U-CogNet.
Procesa audio extraído y genera embeddings para análisis multimodal.
"""

import numpy as np
import librosa
from typing import Optional, List
from .audio_types import AudioData, AudioFeatures, AudioEvent
from .logging import logger

class AudioEncoder:
    """
    Codifica audio en embeddings para procesamiento multimodal.
    Extrae características acústicas y genera representaciones vectoriales.
    """

    def __init__(self, embedding_dim: int = 512, sample_rate: int = 44100):
        self.embedding_dim = embedding_dim
        self.sample_rate = sample_rate
        logger.info(f"AudioEncoder inicializado con dim={embedding_dim}, sr={sample_rate}")

    def extract_features(self, audio_data: AudioData) -> AudioFeatures:
        """
        Extrae características acústicas del audio.

        Args:
            audio_data: Datos de audio crudos

        Returns:
            AudioFeatures con todas las características extraídas
        """
        try:
            # Resamplear si es necesario
            if audio_data.sample_rate != self.sample_rate:
                waveform = librosa.resample(
                    audio_data.waveform.T,
                    orig_sr=audio_data.sample_rate,
                    target_sr=self.sample_rate
                ).T
            else:
                waveform = audio_data.waveform

            # Convertir a mono si es estéreo
            if waveform.ndim > 1:
                waveform = librosa.to_mono(waveform.T)

            # Extraer MFCC
            mfcc = librosa.feature.mfcc(
                y=waveform,
                sr=self.sample_rate,
                n_mfcc=13,
                n_fft=2048,
                hop_length=512
            )

            # Extraer chroma
            chroma = librosa.feature.chroma_stft(
                y=waveform,
                sr=self.sample_rate,
                n_fft=2048,
                hop_length=512
            )

            # Centroid espectral
            spectral_centroid = librosa.feature.spectral_centroid(
                y=waveform,
                sr=self.sample_rate,
                n_fft=2048,
                hop_length=512
            )

            # Ancho de banda espectral
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=waveform,
                sr=self.sample_rate,
                n_fft=2048,
                hop_length=512
            )

            # Tasa de cruce por cero
            zero_crossing_rate = librosa.feature.zero_crossing_rate(
                y=waveform,
                frame_length=2048,
                hop_length=512
            )

            # Energía RMS
            rms = librosa.feature.rms(
                y=waveform,
                frame_length=2048,
                hop_length=512
            )

            # Tempo y beats
            tempo, beat_positions = librosa.beat.beat_track(
                y=waveform,
                sr=self.sample_rate
            )

            features = AudioFeatures(
                mfcc=mfcc,
                chroma=chroma,
                spectral_centroid=spectral_centroid,
                spectral_bandwidth=spectral_bandwidth,
                zero_crossing_rate=zero_crossing_rate,
                rms=rms,
                tempo=tempo,
                beat_positions=beat_positions,
                sample_rate=self.sample_rate,
                duration=audio_data.duration
            )

            logger.debug(f"Características extraídas: MFCC shape {mfcc.shape}")
            return features

        except Exception as e:
            logger.error(f"Error extrayendo características de audio: {str(e)}")
            raise

    def encode_to_embedding(self, features: AudioFeatures) -> np.ndarray:
        """
        Convierte características de audio en embedding vectorial.

        Args:
            features: Características extraídas del audio

        Returns:
            Embedding vectorial de dimensión embedding_dim
        """
        try:
            # Concatenar todas las características
            feature_list = [
                features.mfcc.mean(axis=1),  # Promedio temporal de MFCC
                features.chroma.mean(axis=1),  # Promedio de chroma
                np.array([features.spectral_centroid.mean()]),
                np.array([features.spectral_bandwidth.mean()]),
                np.array([features.zero_crossing_rate.mean()]),
                np.array([features.rms.mean()]),
                np.array([features.tempo]),
            ]

            # Concatenar en un vector
            combined_features = np.concatenate(feature_list)

            # Reducir a embedding_dim usando PCA simple o truncamiento
            if combined_features.shape[0] > self.embedding_dim:
                # Truncar si es más grande
                embedding = combined_features[:self.embedding_dim]
            else:
                # Pad con ceros si es más pequeño
                embedding = np.pad(combined_features, (0, self.embedding_dim - combined_features.shape[0]))

            # Normalizar
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            logger.debug(f"Embedding generado: shape {embedding.shape}")
            return embedding

        except Exception as e:
            logger.error(f"Error generando embedding: {str(e)}")
            raise

    def detect_events(self, features: AudioFeatures) -> List[AudioEvent]:
        """
        Detecta eventos de audio basados en características.

        Args:
            features: Características del audio

        Returns:
            Lista de eventos detectados
        """
        events = []

        try:
            # Detección simple de silencio vs sonido
            rms_threshold = 0.01
            silence_frames = features.rms[0] < rms_threshold

            # Encontrar segmentos
            diff = np.diff(np.concatenate([[True], silence_frames, [True]]))
            starts = np.where(diff[:-1] == -1)[0]
            ends = np.where(diff[1:] == 1)[0]

            for start, end in zip(starts, ends):
                start_time = start * 512 / self.sample_rate  # hop_length / sample_rate
                end_time = end * 512 / self.sample_rate

                # Clasificar evento
                avg_rms = np.mean(features.rms[0, start:end])
                if avg_rms < rms_threshold:
                    event_type = "silence"
                    confidence = 0.9
                else:
                    # Clasificación simple basada en características
                    avg_centroid = np.mean(features.spectral_centroid[0, start:end])
                    if avg_centroid > 3000:  # Frecuencia alta
                        event_type = "noise"
                        confidence = 0.7
                    else:
                        event_type = "sound"
                        confidence = 0.6

                event = AudioEvent(
                    event_type=event_type,
                    confidence=confidence,
                    start_time=start_time,
                    end_time=end_time,
                    features=features
                )
                events.append(event)

            logger.debug(f"Eventos detectados: {len(events)}")
            return events

        except Exception as e:
            logger.error(f"Error detectando eventos: {str(e)}")
            return []

    def process_audio(self, audio_data: AudioData) -> Tuple[np.ndarray, List[AudioEvent]]:
        """
        Pipeline completo: extraer características, generar embedding y detectar eventos.

        Args:
            audio_data: Datos de audio crudos

        Returns:
            Tupla de (embedding, lista de eventos)
        """
        features = self.extract_features(audio_data)
        embedding = self.encode_to_embedding(features)
        events = self.detect_events(features)

        logger.info(f"Audio procesado: embedding shape {embedding.shape}, {len(events)} eventos")
        return embedding, events