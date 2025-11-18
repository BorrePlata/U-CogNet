"""
Procesador Cognitivo de Audio para U-CogNet
Integra extracción, razonamiento, interiorización e imaginación
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logging.warning("MoviePy not available. Audio extraction will be limited.")

from ..common.audio_types import AudioData, AudioFeatures, AudioEvent
from ..common.logging import logger
from ..cognitive_core import CognitiveCore
from ..semantic_feedback import SemanticFeedback

@dataclass
class AudioReasoning:
    """Resultado del razonamiento sobre audio."""
    event_type: str  # 'speech', 'music', 'environmental', 'silence'
    confidence: float
    semantic_description: str
    emotional_content: Dict[str, float]  # valence, arousal, etc.
    temporal_patterns: List[Dict[str, Any]]
    cognitive_insights: List[str]

@dataclass
class AudioImagination:
    """Representación imaginativa generada."""
    continuation_audio: Optional[np.ndarray]
    imagined_scenarios: List[Dict[str, Any]]
    creative_associations: List[str]
    novelty_score: float
    coherence_score: float

@dataclass
class AudioCognitiveMetrics:
    """Métricas de procesamiento cognitivo de audio."""
    extraction_quality: float  # SNR, distorsión
    reasoning_accuracy: float  # Precisión en clasificación
    interiorization_depth: float  # Profundidad de integración cognitiva
    imagination_creativity: float  # Novedad y coherencia
    processing_latency: float
    memory_utilization: float

class AudioExtractionStrategy(ABC):
    """Estrategia abstracta para extracción de audio."""

    @abstractmethod
    async def extract(self, video_path: str) -> Tuple[np.ndarray, int]:
        """Extrae audio del video. Retorna (audio_data, sample_rate)."""
        pass

class MoviePyAudioStrategy(AudioExtractionStrategy):
    """Extracción usando MoviePy - robusta y automática."""

    async def extract(self, video_path: str) -> Tuple[np.ndarray, int]:
        if not MOVIEPY_AVAILABLE:
            raise ImportError("MoviePy required for this strategy")

        def _extract_sync():
            clip = VideoFileClip(video_path)
            audio = clip.audio
            if audio is None:
                raise ValueError("Video has no audio track")

            # Extraer como numpy array
            audio_array = audio.to_soundarray(fps=22050)
            sample_rate = audio.fps

            clip.close()
            return audio_array, sample_rate

        # Ejecutar en thread pool para no bloquear
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _extract_sync)

class AudioCognitiveProcessor:
    """
    Procesador cognitivo completo para audio.
    Integra extracción → razonamiento → interiorización → imaginación
    """

    def __init__(self,
                 cognitive_core: CognitiveCore,
                 semantic_feedback: SemanticFeedback,
                 extraction_strategy: Optional[AudioExtractionStrategy] = None):

        self.cognitive_core = cognitive_core
        self.semantic_feedback = semantic_feedback
        self.extraction_strategy = extraction_strategy or MoviePyAudioStrategy()

        # Memoria interna para patrones de audio
        self.audio_memory: Dict[str, List[AudioFeatures]] = {}
        self.reasoning_patterns: Dict[str, Dict[str, Any]] = {}

        # Métricas acumuladas
        self.metrics_history: List[AudioCognitiveMetrics] = []

        logger.info("AudioCognitiveProcessor inicializado")

    async def process_video_audio(self, video_path: str) -> Dict[str, Any]:
        """
        Pipeline completo: extracción → razonamiento → interiorización → imaginación
        """
        start_time = datetime.now()

        try:
            # 1. EXTRACCIÓN
            logger.info(f"Iniciando procesamiento cognitivo de audio: {video_path}")
            audio_data, sample_rate = await self.extraction_strategy.extract(video_path)

            # Crear estructura de datos
            raw_audio = AudioData(
                waveform=audio_data,
                sample_rate=sample_rate,
                duration=len(audio_data) / sample_rate,
                source=video_path,
                timestamp=start_time.timestamp()
            )

            # 2. RAZONAMIENTO
            reasoning = await self._reason_about_audio(raw_audio)

            # 3. INTERIORIZACIÓN
            await self._interiorize_audio(raw_audio, reasoning)

            # 4. IMAGINACIÓN
            imagination = await self._generate_imagination(raw_audio, reasoning)

            # 5. MÉTRICAS
            metrics = self._calculate_metrics(raw_audio, reasoning, imagination,
                                            (datetime.now() - start_time).total_seconds())

            # Almacenar métricas
            self.metrics_history.append(metrics)

            result = {
                'audio_data': raw_audio,
                'reasoning': reasoning,
                'imagination': imagination,
                'metrics': metrics,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }

            logger.info(f"Procesamiento cognitivo completado en {result['processing_time']:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error en procesamiento cognitivo de audio: {e}")
            raise

    async def _reason_about_audio(self, audio_data: AudioData) -> AudioReasoning:
        """
        Razonamiento profundo sobre el contenido del audio.
        """
        # Análisis básico de características
        features = self._extract_basic_features(audio_data)

        # Clasificación de tipo de evento
        event_type, confidence = self._classify_audio_event(features)

        # Análisis semántico
        semantic_desc = await self._generate_semantic_description(features, event_type)

        # Análisis emocional
        emotional_content = self._analyze_emotional_content(features)

        # Patrones temporales
        temporal_patterns = self._extract_temporal_patterns(features)

        # Insights cognitivos
        cognitive_insights = self._generate_cognitive_insights(features, event_type)

        return AudioReasoning(
            event_type=event_type,
            confidence=confidence,
            semantic_description=semantic_desc,
            emotional_content=emotional_content,
            temporal_patterns=temporal_patterns,
            cognitive_insights=cognitive_insights
        )

    async def _interiorize_audio(self, audio_data: AudioData, reasoning: AudioReasoning):
        """
        Interioriza el audio en la memoria cognitiva del sistema.
        """
        # Crear evento de audio
        audio_event = AudioEvent(
            event_type=reasoning.event_type,
            confidence=reasoning.confidence,
            start_time=audio_data.timestamp,
            end_time=audio_data.timestamp + audio_data.duration,
            features=self._extract_basic_features(audio_data),
            transcription=None  # Podría añadirse ASR más tarde
        )

        # Almacenar en memoria de audio
        if reasoning.event_type not in self.audio_memory:
            self.audio_memory[reasoning.event_type] = []
        self.audio_memory[reasoning.event_type].append(audio_event.features)

        # Limitar memoria
        if len(self.audio_memory[reasoning.event_type]) > 50:
            self.audio_memory[reasoning.event_type] = self.audio_memory[reasoning.event_type][-50:]

        # Actualizar patrones de razonamiento
        self._update_reasoning_patterns(reasoning)

        # Integrar con núcleo cognitivo general
        # Nota: Adaptar cuando el núcleo cognitivo soporte audio
        logger.debug(f"Audio interiorizado: {reasoning.event_type} ({reasoning.confidence:.2f})")

    async def _generate_imagination(self, audio_data: AudioData, reasoning: AudioReasoning) -> AudioImagination:
        """
        Genera imaginación creativa basada en el audio procesado.
        """
        # Generar continuación imaginativa
        continuation = self._generate_audio_continuation(audio_data, reasoning)

        # Escenarios imaginados
        scenarios = self._imagine_scenarios(reasoning)

        # Asociaciones creativas
        associations = self._generate_creative_associations(reasoning)

        # Evaluar novedad y coherencia
        novelty = self._calculate_novelty(continuation, reasoning)
        coherence = self._calculate_coherence(continuation, audio_data)

        return AudioImagination(
            continuation_audio=continuation,
            imagined_scenarios=scenarios,
            creative_associations=associations,
            novelty_score=novelty,
            coherence_score=coherence
        )

    def _extract_basic_features(self, audio_data: AudioData) -> AudioFeatures:
        """Extrae características básicas usando análisis simple."""
        # Implementación simplificada - en producción usar Librosa
        waveform = audio_data.waveform
        sample_rate = audio_data.sample_rate

        # Cálculos básicos
        rms = np.sqrt(np.mean(waveform**2))
        zero_crossings = np.sum(np.abs(np.diff(np.sign(waveform)))) / (2 * len(waveform))

        # Simular MFCC y chroma (en producción usar librosa)
        n_frames = max(1, int(len(waveform) / 512))
        mfcc = np.random.randn(13, n_frames) * 0.1  # Placeholder
        chroma = np.random.randn(12, n_frames) * 0.1  # Placeholder

        return AudioFeatures(
            mfcc=mfcc,
            chroma=chroma,
            spectral_centroid=np.array([2000.0]),  # Placeholder
            spectral_bandwidth=np.array([1000.0]),  # Placeholder
            zero_crossing_rate=np.array([zero_crossings]),
            rms=np.array([rms]),
            tempo=120.0,  # Placeholder
            beat_positions=np.array([]),  # Placeholder
            sample_rate=sample_rate,
            duration=audio_data.duration
        )

    def _classify_audio_event(self, features: AudioFeatures) -> Tuple[str, float]:
        """Clasifica el tipo de evento de audio."""
        # Lógica simplificada basada en características
        rms = np.mean(features.rms)
        zcr = np.mean(features.zero_crossing_rate)

        if rms < 0.01:
            return "silence", 0.9
        elif zcr > 0.1:
            return "speech", 0.7
        elif np.mean(features.chroma) > 0.05:
            return "music", 0.8
        else:
            return "environmental", 0.6

    async def _generate_semantic_description(self, features: AudioFeatures, event_type: str) -> str:
        """Genera descripción semántica."""
        descriptions = {
            "speech": "Contenido vocal con patrones lingüísticos",
            "music": "Secuencia musical con estructura rítmica",
            "environmental": "Sonidos ambientales con características naturales",
            "silence": "Ausencia de señal audible"
        }
        return descriptions.get(event_type, "Contenido de audio no clasificado")

    def _analyze_emotional_content(self, features: AudioFeatures) -> Dict[str, float]:
        """Analiza contenido emocional."""
        # Análisis simplificado
        rms = np.mean(features.rms)
        tempo = features.tempo

        valence = 0.5 + 0.3 * np.sin(rms * 10)  # Placeholder
        arousal = min(1.0, rms * 5)
        dominance = 0.5 + 0.2 * (tempo - 120) / 120

        return {
            'valence': float(valence),
            'arousal': float(arousal),
            'dominance': float(dominance)
        }

    def _extract_temporal_patterns(self, features: AudioFeatures) -> List[Dict[str, Any]]:
        """Extrae patrones temporales."""
        # Análisis simplificado
        return [{
            'type': 'rhythm',
            'period': 60.0 / features.tempo,
            'strength': 0.8
        }]

    def _generate_cognitive_insights(self, features: AudioFeatures, event_type: str) -> List[str]:
        """Genera insights cognitivos."""
        insights = []
        if event_type == "speech":
            insights.append("Posible comunicación intencional detectada")
        elif event_type == "music":
            insights.append("Patrón estético con estructura temporal")
        elif event_type == "environmental":
            insights.append("Información contextual del entorno")
        return insights

    def _update_reasoning_patterns(self, reasoning: AudioReasoning):
        """Actualiza patrones de razonamiento aprendidos."""
        if reasoning.event_type not in self.reasoning_patterns:
            self.reasoning_patterns[reasoning.event_type] = {
                'count': 0,
                'avg_confidence': 0.0,
                'common_insights': []
            }

        pattern = self.reasoning_patterns[reasoning.event_type]
        pattern['count'] += 1
        pattern['avg_confidence'] = (pattern['avg_confidence'] * (pattern['count'] - 1) +
                                   reasoning.confidence) / pattern['count']
        pattern['common_insights'].extend(reasoning.cognitive_insights)

    def _generate_audio_continuation(self, audio_data: AudioData, reasoning: AudioReasoning) -> Optional[np.ndarray]:
        """Genera continuación imaginativa del audio."""
        # Implementación simplificada: ruido blanco correlacionado
        if reasoning.event_type == "silence":
            return None

        continuation_length = min(int(audio_data.sample_rate * 5), len(audio_data.waveform))
        continuation = np.random.randn(continuation_length) * 0.1

        # Hacerlo similar al audio original
        if len(audio_data.waveform) > 1000:
            # Usar características del final del audio original
            recent_segment = audio_data.waveform[-1000:]
            continuation *= np.std(recent_segment) / np.std(continuation)
            continuation += np.mean(recent_segment)

        return continuation

    def _imagine_scenarios(self, reasoning: AudioReasoning) -> List[Dict[str, Any]]:
        """Genera escenarios imaginados."""
        scenarios = []
        if reasoning.event_type == "speech":
            scenarios.append({
                'scenario': 'conversación_continua',
                'probability': 0.7,
                'description': 'Continuación de diálogo iniciado'
            })
        elif reasoning.event_type == "music":
            scenarios.append({
                'scenario': 'desarrollo_musical',
                'probability': 0.8,
                'description': 'Evolución de la composición musical'
            })
        return scenarios

    def _generate_creative_associations(self, reasoning: AudioReasoning) -> List[str]:
        """Genera asociaciones creativas."""
        associations = []
        if reasoning.event_type == "environmental":
            associations.extend([
                "Recuerdos de naturaleza",
                "Sensaciones táctiles asociadas",
                "Conexiones emocionales con el entorno"
            ])
        return associations

    def _calculate_novelty(self, continuation: Optional[np.ndarray], reasoning: AudioReasoning) -> float:
        """Calcula novedad de la imaginación."""
        if continuation is None:
            return 0.0
        # Novedad basada en desviación de patrones conocidos
        base_novelty = np.std(continuation) / (np.mean(np.abs(continuation)) + 1e-6)
        return min(1.0, base_novelty)

    def _calculate_coherence(self, continuation: Optional[np.ndarray], original: AudioData) -> float:
        """Calcula coherencia con el audio original."""
        if continuation is None:
            return 0.0
        # Coherencia basada en similitud estadística
        orig_std = np.std(original.waveform)
        cont_std = np.std(continuation)
        coherence = 1.0 - abs(orig_std - cont_std) / (orig_std + 1e-6)
        return max(0.0, coherence)

    def _calculate_metrics(self, audio_data: AudioData, reasoning: AudioReasoning,
                          imagination: AudioImagination, processing_time: float) -> AudioCognitiveMetrics:
        """Calcula métricas completas del procesamiento."""

        # Calidad de extracción (simplificada)
        extraction_quality = 0.9  # SNR estimado

        # Precisión de razonamiento
        reasoning_accuracy = reasoning.confidence

        # Profundidad de interiorización (basada en memoria)
        memory_depth = len(self.audio_memory.get(reasoning.event_type, []))
        interiorization_depth = min(1.0, memory_depth / 10.0)

        # Creatividad de imaginación
        imagination_creativity = (imagination.novelty_score + imagination.coherence_score) / 2.0

        return AudioCognitiveMetrics(
            extraction_quality=extraction_quality,
            reasoning_accuracy=reasoning_accuracy,
            interiorization_depth=interiorization_depth,
            imagination_creativity=imagination_creativity,
            processing_latency=processing_time,
            memory_utilization=len(self.audio_memory) / 10.0  # Normalizado
        )

    def get_cognitive_status(self) -> Dict[str, Any]:
        """Obtiene estado cognitivo del procesador de audio."""
        return {
            'audio_memory_size': sum(len(mem) for mem in self.audio_memory.values()),
            'reasoning_patterns': len(self.reasoning_patterns),
            'metrics_history_length': len(self.metrics_history),
            'average_creativity': np.mean([m.imagination_creativity for m in self.metrics_history]) if self.metrics_history else 0.0,
            'average_reasoning_accuracy': np.mean([m.reasoning_accuracy for m in self.metrics_history]) if self.metrics_history else 0.0
        }