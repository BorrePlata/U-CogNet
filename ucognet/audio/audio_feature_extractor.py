# U-CogNet Audio Feature Extractor
# Advanced Audio Feature Extraction using Librosa

import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import logging

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("Librosa not available. Using fallback feature extraction.")

from .audio_types import AudioFrame, AudioFeatures
from .audio_protocols import AudioFeatureExtractionProtocol

class LibrosaFeatureExtractor(AudioFeatureExtractionProtocol):
    """Advanced audio feature extraction using Librosa."""

    def __init__(self):
        if not LIBROSA_AVAILABLE:
            raise ImportError("Librosa is required for LibrosaFeatureExtractor")

        # Default parameters
        self._n_mfcc = 13
        self._n_chroma = 12
        self._n_fft = 2048
        self._hop_length = 512
        self._sr = 22050

    async def extract_features(self, audio_frame: AudioFrame) -> AudioFeatures:
        """Extract comprehensive audio features using Librosa."""
        audio_data = audio_frame.data
        sample_rate = audio_frame.sample_rate

        # Ensure audio is mono
        if audio_data.ndim > 1:
            audio_data = librosa.to_mono(audio_data.T)

        # Normalize audio
        audio_data = librosa.util.normalize(audio_data)

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio_data,
            sr=sample_rate,
            n_mfcc=self._n_mfcc,
            n_fft=self._n_fft,
            hop_length=self._hop_length
        )

        # Extract chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio_data,
            sr=sample_rate,
            n_fft=self._n_fft,
            hop_length=self._hop_length
        )

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_data, sr=sample_rate, n_fft=self._n_fft, hop_length=self._hop_length
        )
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio_data, sr=sample_rate, n_fft=self._n_fft, hop_length=self._hop_length
        )
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_data, sr=sample_rate, n_fft=self._n_fft, hop_length=self._hop_length
        )

        # Zero crossing rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            y=audio_data, hop_length=self._hop_length
        )

        # RMS energy
        rms_energy = librosa.feature.rms(y=audio_data, hop_length=self._hop_length)

        # Onset strength
        onset_env = librosa.onset.onset_strength(
            y=audio_data, sr=sample_rate, hop_length=self._hop_length
        )

        # Tempo estimation
        tempo, _ = librosa.beat.tempo(
            onset_envelope=onset_env, sr=sample_rate, hop_length=self._hop_length
        )

        # Beat positions
        _, beat_positions = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sample_rate, hop_length=self._hop_length
        )

        # Harmonic-percussive separation
        harmonic, percussive = librosa.effects.hpss(y=audio_data)

        # Harmonic ratio (approximate)
        harmonic_energy = np.mean(harmonic ** 2)
        percussive_energy = np.mean(percussive ** 2)
        total_energy = harmonic_energy + percussive_energy
        harmonic_ratio = harmonic_energy / total_energy if total_energy > 0 else 0.5
        percussive_ratio = percussive_energy / total_energy if total_energy > 0 else 0.5

        # Chroma features for tonnetz
        tonnetz = librosa.feature.tonnetz(
            y=harmonic, sr=sample_rate
        )

        # Spectrograms
        spectrogram = librosa.stft(y=audio_data, n_fft=self._n_fft, hop_length=self._hop_length)
        spectrogram = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)

        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_data, sr=sample_rate, n_fft=self._n_fft, hop_length=self._hop_length
        )
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Aggregate features (take means for scalar values)
        features = AudioFeatures(
            timestamp=audio_frame.timestamp,
            mfcc=mfcc,
            chroma=chroma,
            spectral_centroid=float(np.mean(spectral_centroid)),
            spectral_bandwidth=float(np.mean(spectral_bandwidth)),
            spectral_rolloff=float(np.mean(spectral_rolloff)),
            zero_crossing_rate=float(np.mean(zero_crossing_rate)),
            rms_energy=float(np.mean(rms_energy)),
            onset_strength=float(np.mean(onset_env)),
            tempo=float(tempo),
            beat_positions=beat_positions.tolist(),
            harmonic_ratio=harmonic_ratio,
            percussive_ratio=percussive_ratio,
            tonnetz=tonnetz,
            spectrogram=spectrogram,
            mel_spectrogram=mel_spectrogram,
            metadata={
                'extraction_method': 'librosa',
                'n_mfcc': self._n_mfcc,
                'n_fft': self._n_fft,
                'hop_length': self._hop_length
            }
        )

        return features

    async def analyze_spectrogram(self, audio_data: np.ndarray,
                                sample_rate: int) -> Dict[str, np.ndarray]:
        """Generate detailed spectrogram analysis."""
        # STFT spectrogram
        stft = librosa.stft(y=audio_data, n_fft=self._n_fft, hop_length=self._hop_length)
        spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, sr=sample_rate, n_fft=self._n_fft, hop_length=self._hop_length
        )
        mel_spectrogram = librosa.power_to_db(mel_spec, ref=np.max)

        # Chromagram
        chromagram = librosa.feature.chroma_stft(
            y=audio_data, sr=sample_rate, n_fft=self._n_fft, hop_length=self._hop_length
        )

        # Constant-Q transform (for better frequency resolution)
        cqt = librosa.cqt(y=audio_data, sr=sample_rate, hop_length=self._hop_length)
        cqt_spectrogram = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

        return {
            'stft_spectrogram': spectrogram,
            'mel_spectrogram': mel_spectrogram,
            'chromagram': chromagram,
            'cqt_spectrogram': cqt_spectrogram
        }

    async def detect_onsets(self, audio_data: np.ndarray, sample_rate: int) -> List[float]:
        """Detect sound onsets with timestamps."""
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sample_rate, hop_length=self._hop_length
        )

        # Convert frame indices to time
        onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=self._hop_length)

        return onset_times.tolist()

    async def estimate_tempo(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Estimate tempo with confidence."""
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
        tempo, _ = librosa.beat.tempo(
            onset_envelope=onset_env, sr=sample_rate, hop_length=self._hop_length
        )
        return float(tempo)

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with custom parameters."""
        self._n_mfcc = config.get('n_mfcc', 13)
        self._n_chroma = config.get('n_chroma', 12)
        self._n_fft = config.get('n_fft', 2048)
        self._hop_length = config.get('hop_length', 512)
        self._sr = config.get('sample_rate', 22050)

        logging.info("Initialized LibrosaFeatureExtractor with custom parameters")

    async def cleanup(self) -> None:
        """Clean up resources (no-op for librosa)."""
        pass

class FallbackFeatureExtractor(AudioFeatureExtractionProtocol):
    """Fallback feature extractor when librosa is not available."""

    def __init__(self):
        self._sample_rate = 22050

    async def extract_features(self, audio_frame: AudioFrame) -> AudioFeatures:
        """Extract basic features without librosa."""
        audio_data = audio_frame.data
        sample_rate = audio_frame.sample_rate

        # Ensure mono
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=0 if audio_data.shape[0] > audio_data.shape[1] else 1)

        # Basic spectral features (simplified)
        # Zero crossing rate
        zero_crossing_rate = np.mean(np.abs(np.diff(np.sign(audio_data))))

        # RMS energy
        rms_energy = np.sqrt(np.mean(audio_data ** 2))

        # Spectral centroid approximation
        # (Very simplified - would need proper FFT in production)
        spectral_centroid = 1000.0  # Placeholder
        spectral_bandwidth = 500.0  # Placeholder
        spectral_rolloff = 2000.0   # Placeholder

        # Simplified features
        mfcc = np.random.randn(13, 10)  # Placeholder MFCC
        chroma = np.random.randn(12, 10)  # Placeholder chroma
        tonnetz = np.random.randn(6, 10)  # Placeholder tonnetz

        # Spectrogram placeholder
        spectrogram = np.random.randn(1024, 10)
        mel_spectrogram = np.random.randn(128, 10)

        features = AudioFeatures(
            timestamp=audio_frame.timestamp,
            mfcc=mfcc,
            chroma=chroma,
            spectral_centroid=spectral_centroid,
            spectral_bandwidth=spectral_bandwidth,
            spectral_rolloff=spectral_rolloff,
            zero_crossing_rate=zero_crossing_rate,
            rms_energy=rms_energy,
            onset_strength=0.5,  # Placeholder
            tempo=120.0,  # Placeholder
            beat_positions=[],  # Placeholder
            harmonic_ratio=0.6,  # Placeholder
            percussive_ratio=0.4,  # Placeholder
            tonnetz=tonnetz,
            spectrogram=spectrogram,
            mel_spectrogram=mel_spectrogram,
            metadata={'extraction_method': 'fallback', 'warning': 'librosa_not_available'}
        )

        return features

    async def analyze_spectrogram(self, audio_data: np.ndarray,
                                sample_rate: int) -> Dict[str, np.ndarray]:
        """Basic spectrogram analysis fallback."""
        # Very simplified spectrogram
        spectrogram = np.random.randn(256, 10)
        return {'basic_spectrogram': spectrogram}

    async def detect_onsets(self, audio_data: np.ndarray, sample_rate: int) -> List[float]:
        """Basic onset detection fallback."""
        # Simple energy-based onset detection
        energy = audio_data ** 2
        threshold = np.mean(energy) + np.std(energy)
        onsets = np.where(energy > threshold)[0]
        onset_times = onsets / sample_rate
        return onset_times.tolist()[:10]  # Limit to 10 onsets

    async def estimate_tempo(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Basic tempo estimation fallback."""
        return 120.0  # Default tempo

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize fallback extractor."""
        logging.warning("Using fallback feature extractor - install librosa for full functionality")

    async def cleanup(self) -> None:
        """Clean up (no-op)."""
        pass

# Factory function to get appropriate extractor
def create_feature_extractor() -> AudioFeatureExtractionProtocol:
    """Create the best available feature extractor."""
    if LIBROSA_AVAILABLE:
        return LibrosaFeatureExtractor()
    else:
        logging.warning("Using fallback feature extractor due to missing librosa")
        return FallbackFeatureExtractor()
<parameter name="filePath">/mnt/c/Users/desar/Documents/Science/UCogNet/src/ucognet/modules/audio/feature_extractor.py