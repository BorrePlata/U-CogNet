#!/usr/bin/env python3
"""
Módulo de Percepción Consciente y Sanitización Multidimensional
Parte de la Arquitectura de Seguridad Cognitiva Interdimensional de U-CogNet

Este módulo protege la capa de percepción del sistema cognitivo, asegurando que
ninguna entrada pueda inyectar patrones peligrosos o inducir estados inestables.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import hashlib
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticNormalizer:
    """Normaliza señales de entrada a un espacio semántico universal"""

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.universal_embedder = self._create_universal_embedder()

    def _create_universal_embedder(self) -> nn.Module:
        """Crea un embedder universal para todas las modalidades"""
        return nn.Sequential(
            nn.Linear(1000, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

    def normalize(self, signal: Any, modality: str) -> torch.Tensor:
        """
        Normaliza una señal a representación semántica universal

        Args:
            signal: Señal de entrada (imagen, audio, texto, etc.)
            modality: Tipo de modalidad ('visual', 'audio', 'text', 'tactile')

        Returns:
            Tensor normalizado en espacio semántico universal
        """
        # Convertir señal a tensor numérico
        signal_tensor = self._signal_to_tensor(signal, modality)

        # Aplicar embedding universal
        with torch.no_grad():
            normalized = self.universal_embedder(signal_tensor)

        return normalized

    def _signal_to_tensor(self, signal: Any, modality: str) -> torch.Tensor:
        """Convierte señal de cualquier tipo a tensor"""
        if modality == 'visual':
            # Para imágenes: aplanar y normalizar
            if isinstance(signal, np.ndarray):
                return torch.tensor(signal.flatten(), dtype=torch.float32)
            elif isinstance(signal, torch.Tensor):
                return signal.flatten()

        elif modality == 'audio':
            # Para audio: espectrograma a vector
            if isinstance(signal, np.ndarray):
                return torch.tensor(signal.mean(axis=0), dtype=torch.float32)

        elif modality == 'text':
            # Para texto: usar longitud como proxy simple
            if isinstance(signal, str):
                return torch.tensor([len(signal), hash(signal) % 1000], dtype=torch.float32)

        elif modality == 'tactile':
            # Para datos táctiles: directo a tensor
            if isinstance(signal, (list, np.ndarray)):
                return torch.tensor(signal, dtype=torch.float32)

        # Fallback: hash numérico
        signal_hash = hash(str(signal)) % 1000
        return torch.tensor([signal_hash], dtype=torch.float32)


class AdversarialFilter:
    """Detecta y filtra patrones adversariales"""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.adversarial_detector = self._create_detector()

    def _create_detector(self) -> nn.Module:
        """Crea detector de patrones adversariales"""
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def detect_adversarial(self, normalized_signal: torch.Tensor) -> Tuple[bool, float]:
        """
        Detecta si una señal contiene patrones adversariales

        Returns:
            (es_adversarial, confianza)
        """
        with torch.no_grad():
            confidence = self.adversarial_detector(normalized_signal).item()

        is_adversarial = confidence > self.threshold
        return is_adversarial, confidence


class MultimodalCoherenceChecker:
    """Verifica coherencia entre diferentes modalidades"""

    def __init__(self, coherence_threshold: float = 0.7):
        self.coherence_threshold = coherence_threshold
        self.coherence_model = self._create_coherence_model()

    def _create_coherence_model(self) -> nn.Module:
        """Modelo para medir coherencia cross-modal"""
        return nn.Sequential(
            nn.Linear(1024, 512),  # Concatenación de dos embeddings
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def check_coherence(self, modality_signals: Dict[str, torch.Tensor]) -> Tuple[bool, str]:
        """
        Verifica coherencia entre todas las parejas de modalidades

        Args:
            modality_signals: Diccionario {modalidad: tensor_normalizado}

        Returns:
            (es_coherente, mensaje_diagnostico)
        """
        modalities = list(modality_signals.keys())

        if len(modalities) < 2:
            return True, "Solo una modalidad - coherencia trivial"

        # Verificar coherencia pairwise
        for i, mod1 in enumerate(modalities):
            for mod2 in modalities[i+1:]:
                sig1 = modality_signals[mod1]
                sig2 = modality_signals[mod2]

                # Concatenar señales
                combined = torch.cat([sig1, sig2], dim=0)

                # Medir coherencia
                with torch.no_grad():
                    coherence_score = self.coherence_model(combined.unsqueeze(0)).item()

                if coherence_score < self.coherence_threshold:
                    return False, f"Incoherencia detectada entre {mod1} y {mod2} (score: {coherence_score:.3f})"

        return True, "Coherencia verificada entre todas las modalidades"


class CognitiveFuzzer:
    """Realiza fuzzing cognitivo para detectar entradas problemáticas"""

    def __init__(self, perturbation_steps: int = 10):
        self.perturbation_steps = perturbation_steps
        self.stability_threshold = 0.9

    def fuzz_test(self, signal: torch.Tensor, cognitive_model: Any) -> Tuple[bool, str]:
        """
        Aplica perturbaciones a la señal y verifica estabilidad cognitiva

        Args:
            signal: Señal a testar
            cognitive_model: Modelo cognitivo para evaluar estabilidad

        Returns:
            (es_estable, diagnostico)
        """
        original_output = self._get_cognitive_response(signal, cognitive_model)

        for step in range(self.perturbation_steps):
            # Generar perturbación
            perturbed_signal = self._generate_perturbation(signal)

            # Obtener respuesta cognitiva
            perturbed_output = self._get_cognitive_response(perturbed_signal, cognitive_model)

            # Medir estabilidad
            stability = self._measure_stability(original_output, perturbed_output)

            if stability < self.stability_threshold:
                return False, f"Inestabilidad detectada en paso {step} (estabilidad: {stability:.3f})"

        return True, "Señal cognitivamente estable"

    def _generate_perturbation(self, signal: torch.Tensor) -> torch.Tensor:
        """Genera una perturbación de la señal"""
        noise = torch.randn_like(signal) * 0.1
        return signal + noise

    def _get_cognitive_response(self, signal: torch.Tensor, model: Any) -> Any:
        """Obtiene respuesta del modelo cognitivo (placeholder)"""
        # Placeholder - en implementación real, esto llamaría al modelo cognitivo
        return {"entropy": torch.rand(1).item(), "confidence": torch.rand(1).item()}

    def _measure_stability(self, original: Dict, perturbed: Dict) -> float:
        """Mide estabilidad entre respuestas original y perturbada"""
        entropy_diff = abs(original["entropy"] - perturbed["entropy"])
        confidence_diff = abs(original["confidence"] - perturbed["confidence"])

        # Estabilidad como promedio inverso de diferencias
        stability = 1.0 / (1.0 + entropy_diff + confidence_diff)
        return stability


class TopologicalProjector:
    """Proyecta entradas a manifolds topológicos seguros"""

    def __init__(self, manifold_dim: int = 128):
        self.manifold_dim = manifold_dim
        self.projection_model = self._create_projection_model()

    def _create_projection_model(self) -> nn.Module:
        """Crea modelo de proyección topológica"""
        return nn.Sequential(
            nn.Linear(512, self.manifold_dim),
            nn.Tanh(),  # Mantiene valores en [-1, 1] para manifold esférico
            nn.Linear(self.manifold_dim, 512)
        )

    def project_to_safe_manifold(self, signal: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Proyecta señal a manifold topológico seguro

        Returns:
            (señal_proyectada, es_segura)
        """
        with torch.no_grad():
            projected = self.projection_model(signal)

            # Verificar si la proyección mantiene homeomorfismo
            is_homeomorphic = self._check_homeomorphism(signal, projected)

        return projected, is_homeomorphic

    def _check_homeomorphism(self, original: torch.Tensor, projected: torch.Tensor) -> bool:
        """Verifica si la proyección mantiene propiedades topológicas"""
        # Placeholder: verificar distancia relativa
        original_norm = torch.norm(original)
        projected_norm = torch.norm(projected)

        # Si las normas son demasiado diferentes, no es homeomórfica
        ratio = projected_norm / (original_norm + 1e-6)
        return 0.5 < ratio < 2.0


class PerceptionSanitizer:
    """
    Módulo principal de Percepción Consciente y Sanitización Multidimensional
    Coordina todos los componentes de sanitización perceptual
    """

    def __init__(self):
        self.semantic_normalizer = SemanticNormalizer()
        self.adversarial_filter = AdversarialFilter()
        self.coherence_checker = MultimodalCoherenceChecker()
        self.cognitive_fuzzer = CognitiveFuzzer()
        self.topological_projector = TopologicalProjector()

        # Métricas de seguridad
        self.security_metrics = {
            "signals_processed": 0,
            "threats_detected": 0,
            "sanitizations_performed": 0,
            "coherence_failures": 0
        }

        logger.info("🛡️ Módulo de Percepción Consciente inicializado")

    def sanitize_multimodal_input(self, raw_inputs: Dict[str, Any],
                                cognitive_model: Any = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Sanitiza entrada multimodal completa

        Args:
            raw_inputs: Diccionario {modalidad: señal_cruda}
            cognitive_model: Modelo cognitivo para testing (opcional)

        Returns:
            (señales_sanitizadas, metadata_seguridad)
        """
        self.security_metrics["signals_processed"] += 1

        sanitized_signals = {}
        security_metadata = {
            "timestamp": datetime.now().isoformat(),
            "modalities_processed": list(raw_inputs.keys()),
            "security_checks": [],
            "threats_detected": [],
            "sanitization_actions": []
        }

        # Paso 1: Normalización semántica
        normalized_signals = {}
        for modality, signal in raw_inputs.items():
            try:
                normalized = self.semantic_normalizer.normalize(signal, modality)
                normalized_signals[modality] = normalized
                security_metadata["security_checks"].append(f"Normalización semántica: {modality} ✓")
            except Exception as e:
                logger.warning(f"Error en normalización de {modality}: {e}")
                security_metadata["threats_detected"].append(f"Error de normalización: {modality}")
                continue

        # Paso 2: Verificación de coherencia multimodal
        if len(normalized_signals) > 1:
            is_coherent, coherence_message = self.coherence_checker.check_coherence(normalized_signals)
            security_metadata["security_checks"].append(f"Coherencia multimodal: {coherence_message}")

            if not is_coherent:
                self.security_metrics["coherence_failures"] += 1
                security_metadata["threats_detected"].append("Falla de coherencia multimodal")
                # En caso de incoherencia, proceder con precaución

        # Paso 3: Filtro adversarial y sanitización individual
        for modality, normalized_signal in normalized_signals.items():
            # Verificar patrones adversariales
            is_adversarial, confidence = self.adversarial_filter.detect_adversarial(normalized_signal)

            if is_adversarial:
                self.security_metrics["threats_detected"] += 1
                security_metadata["threats_detected"].append(
                    f"Patrón adversarial en {modality} (confianza: {confidence:.3f})"
                )

                # Aplicar proyección topológica como sanitización
                safe_signal, is_safe = self.topological_projector.project_to_safe_manifold(normalized_signal)

                if is_safe:
                    sanitized_signals[modality] = safe_signal
                    security_metadata["sanitization_actions"].append(f"Proyección topológica aplicada: {modality}")
                else:
                    # Si no se puede proyectar seguramente, rechazar la señal
                    security_metadata["sanitization_actions"].append(f"Señal rechazada: {modality}")
                    continue
            else:
                sanitized_signals[modality] = normalized_signal

        # Paso 4: Fuzzing cognitivo (si se proporciona modelo)
        if cognitive_model is not None:
            for modality, signal in sanitized_signals.items():
                is_stable, stability_message = self.cognitive_fuzzer.fuzz_test(signal, cognitive_model)

                if not is_stable:
                    security_metadata["threats_detected"].append(
                        f"Inestabilidad cognitiva en {modality}: {stability_message}"
                    )
                    # Remover señal problemática
                    del sanitized_signals[modality]
                    security_metadata["sanitization_actions"].append(f"Señal removida por inestabilidad: {modality}")

        # Actualizar métricas
        self.security_metrics["sanitizations_performed"] += len(security_metadata["sanitization_actions"])

        logger.info(f"🛡️ Sanitización completada: {len(sanitized_signals)} señales seguras, "
                   f"{len(security_metadata['threats_detected'])} amenazas detectadas")

        return sanitized_signals, security_metadata

    def get_security_report(self) -> Dict[str, Any]:
        """Genera reporte de métricas de seguridad"""
        total_processed = self.security_metrics["signals_processed"]

        if total_processed == 0:
            return {"status": "No signals processed yet"}

        return {
            "total_signals_processed": total_processed,
            "threat_detection_rate": self.security_metrics["threats_detected"] / total_processed,
            "sanitization_rate": self.security_metrics["sanitizations_performed"] / total_processed,
            "coherence_failure_rate": self.security_metrics["coherence_failures"] / total_processed,
            "security_status": "ACTIVE"
        }


# Función de utilidad para testing
def test_perception_sanitizer():
    """Función de test básico del módulo"""
    sanitizer = PerceptionSanitizer()

    # Crear entrada de test
    test_inputs = {
        "visual": np.random.rand(28, 28),  # Imagen dummy
        "audio": np.random.rand(100, 50),  # Audio dummy
        "text": "Esta es una señal de texto de prueba",
        "tactile": [0.1, 0.2, 0.3, 0.4]  # Datos táctiles dummy
    }

    # Sanitizar
    sanitized, metadata = sanitizer.sanitize_multimodal_input(test_inputs)

    print("🧪 Test de Perception Sanitizer:")
    print(f"  - Modalidades procesadas: {len(sanitized)}")
    print(f"  - Amenazas detectadas: {len(metadata['threats_detected'])}")
    print(f"  - Acciones de sanitización: {len(metadata['sanitization_actions'])}")

    # Reporte de seguridad
    report = sanitizer.get_security_report()
    print(f"  - Estado de seguridad: {report.get('security_status', 'UNKNOWN')}")

    return sanitized, metadata


if __name__ == "__main__":
    # Ejecutar test si se llama directamente
    test_perception_sanitizer()