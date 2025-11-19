"""
Núcleo Cognitivo de U-CogNet
Implementa el procesamiento central de información y toma de decisiones
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .interfaces import CognitiveModule
from .tracing import get_event_bus, EventType
from .types import Metrics


class CognitiveState(Enum):
    """Estados cognitivos del sistema"""
    IDLE = "idle"
    PROCESSING = "processing"
    DECIDING = "deciding"
    LEARNING = "learning"
    ADAPTING = "adapting"


@dataclass
class CognitiveContext:
    """Contexto cognitivo actual"""
    state: CognitiveState
    confidence: float
    attention_focus: str
    working_memory: Dict[str, Any]
    episodic_buffer: List[Dict[str, Any]]
    semantic_knowledge: Dict[str, Any]
    last_decision: Optional[Dict[str, Any]] = None
    decision_timestamp: Optional[float] = None


class CognitiveCore(CognitiveModule):
    """
    Núcleo cognitivo principal de U-CogNet
    Coordina percepción, memoria, razonamiento y acción
    """

    def __init__(self):
        self.event_bus = get_event_bus()
        self.context = CognitiveContext(
            state=CognitiveState.IDLE,
            confidence=0.0,
            attention_focus="general",
            working_memory={},
            episodic_buffer=[],
            semantic_knowledge={}
        )

        # Componentes cognitivos
        self.perception_system = None
        self.memory_system = None
        self.reasoning_engine = None
        self.decision_maker = None

        # Estado interno
        self.episode_counter = 0
        self.step_counter = 0

        print("🧠 Cognitive Core inicializado")

    async def initialize(self) -> bool:
        """Inicializa el núcleo cognitivo"""
        try:
            # Emitir evento de inicialización
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "CognitiveCore",
                outputs={"initialized": True},
                explanation="Inicialización del Cognitive Core"
            )

            print("✅ Cognitive Core inicializado correctamente")
            return True

        except Exception as e:
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "CognitiveCore",
                outputs={"initialized": False, "error": str(e)},
                log_level=2  # ERROR
            )
            print(f"❌ Error inicializando Cognitive Core: {e}")
            return False

    async def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa entrada multimodal y actualiza estado cognitivo

        Args:
            input_data: Datos de entrada con diferentes modalidades

        Returns:
            Resultado del procesamiento cognitivo
        """
        self.context.state = CognitiveState.PROCESSING
        episode_id = f"cog_episode_{self.episode_counter}"

        # Emitir evento de procesamiento
        trace_id = self.event_bus.emit(
            EventType.MODULE_INTERACTION,
            "CognitiveCore",
            inputs=input_data,
            episode_id=episode_id,
            step_id=self.step_counter,
            explanation="Procesamiento de entrada multimodal"
        )

        try:
            # Procesar diferentes modalidades
            processed_data = await self._process_multimodal_input(input_data)

            # Actualizar memoria de trabajo
            await self.update_memory(processed_data)

            # Evaluar estado cognitivo
            evaluation = await self.evaluate_state()

            # Preparar resultado
            result = {
                "processed_data": processed_data,
                "cognitive_state": self.context.state.value,
                "confidence": evaluation.get("confidence", 0.0),
                "attention_focus": self.context.attention_focus,
                "recommendations": evaluation.get("recommendations", [])
            }

            # Emitir resultado
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "CognitiveCore",
                outputs=result,
                episode_id=episode_id,
                step_id=self.step_counter,
                trace_id=trace_id,
                explanation="Resultado del procesamiento cognitivo"
            )

            self.step_counter += 1
            return result

        except Exception as e:
            # Emitir error
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "CognitiveCore",
                outputs={"success": False, "error": str(e)},
                episode_id=episode_id,
                step_id=self.step_counter,
                trace_id=trace_id,
                log_level=2
            )
            raise

    async def _process_multimodal_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa entrada de múltiples modalidades"""
        processed = {}

        # Procesar visión si existe
        if "vision" in input_data:
            processed["vision"] = await self._process_vision(input_data["vision"])

        # Procesar audio si existe
        if "audio" in input_data:
            processed["audio"] = await self._process_audio(input_data["audio"])

        # Procesar texto si existe
        if "text" in input_data:
            processed["text"] = await self._process_text(input_data["text"])

        # Procesar series temporales si existen
        if "timeseries" in input_data:
            processed["timeseries"] = await self._process_timeseries(input_data["timeseries"])

        # Integrar modalidades en representación unificada
        integrated = await self._integrate_modalities(processed)

        return {
            "modalities": processed,
            "integrated": integrated,
            "dominant_modality": self._determine_dominant_modality(processed)
        }

    async def _process_vision(self, vision_data: Any) -> Dict[str, Any]:
        """Procesa datos visuales"""
        if isinstance(vision_data, dict) and "detections" in vision_data:
            detections = vision_data["detections"]
            num_people = len(detections)

            # Calcular confianza basada en número de detecciones y consistencia
            base_confidence = min(0.95, 0.5 + (num_people * 0.1))  # Más personas = más confianza

            # Ajustar por calidad de detecciones
            if detections:
                # Manejar tanto objetos con atributo confidence como diccionarios
                confidences = []
                for d in detections:
                    if hasattr(d, 'confidence'):
                        confidences.append(d.confidence)
                    elif isinstance(d, dict) and 'confidence' in d:
                        confidences.append(d['confidence'])
                    else:
                        confidences.append(0.5)  # Default confidence
                
                avg_conf = sum(confidences) / len(confidences)
                confidence = (base_confidence + avg_conf) / 2
            else:
                confidence = base_confidence * 0.5  # Menos confianza si no hay detecciones

            return {
                "type": "vision",
                "features": vision_data,
                "confidence": confidence,
                "objects_detected": [f"person_{i}" for i in range(num_people)],
                "num_people": num_people
            }
        else:
            # Fallback para datos de visión genéricos
            return {
                "type": "vision",
                "features": vision_data,
                "confidence": 0.8,
                "objects_detected": []
            }

    async def _process_audio(self, audio_data: Any) -> Dict[str, Any]:
        """Procesa datos de audio"""
        # Placeholder - se integrará con audio processor
        return {
            "type": "audio",
            "features": audio_data,
            "confidence": 0.7,
            "events_detected": []
        }

    async def _process_text(self, text_data: Any) -> Dict[str, Any]:
        """Procesa datos de texto"""
        # Placeholder - se integrará con NLP processor
        return {
            "type": "text",
            "features": text_data,
            "confidence": 0.9,
            "entities": []
        }

    async def _process_timeseries(self, ts_data: Any) -> Dict[str, Any]:
        """Procesa series temporales"""
        # Placeholder - se integrará con time series processor
        return {
            "type": "timeseries",
            "features": ts_data,
            "confidence": 0.6,
            "patterns": []
        }

    async def _integrate_modalities(self, processed_modalities: Dict[str, Any]) -> Dict[str, Any]:
        """Integra información de múltiples modalidades"""
        # Crear embedding unificado
        integrated_features = []
        weights = []

        for modality, data in processed_modalities.items():
            if "features" in data:
                # Placeholder para integración real
                integrated_features.append(data["features"])
                weights.append(data.get("confidence", 0.5))

        # Promedio ponderado simple (placeholder)
        if integrated_features:
            avg_confidence = sum(weights) / len(weights) if weights else 0.5
        else:
            avg_confidence = 0.0

        return {
            "unified_embedding": integrated_features,
            "integration_confidence": avg_confidence,
            "active_modalities": list(processed_modalities.keys())
        }

    def _determine_dominant_modality(self, processed_modalities: Dict[str, Any]) -> str:
        """Determina la modalidad dominante"""
        if not processed_modalities:
            return "none"

        # Elegir basado en confianza
        best_modality = max(
            processed_modalities.items(),
            key=lambda x: x[1].get("confidence", 0)
        )[0]

        return best_modality

    async def update_memory(self, processed_data: Dict[str, Any]) -> None:
        """
        Actualiza los sistemas de memoria con nueva información

        Args:
            processed_data: Datos procesados para almacenar
        """
        # Actualizar memoria de trabajo
        self.context.working_memory.update({
            "last_processed": processed_data,
            "timestamp": time.time(),
            "step": self.step_counter
        })

        # Agregar a buffer episódico
        episode_entry = {
            "step": self.step_counter,
            "data": processed_data,
            "timestamp": time.time(),
            "context": {
                "state": self.context.state.value,
                "attention": self.context.attention_focus
            }
        }

        self.context.episodic_buffer.append(episode_entry)

        # Mantener buffer episódico limitado
        if len(self.context.episodic_buffer) > 100:
            self.context.episodic_buffer = self.context.episodic_buffer[-100:]

        # Actualizar conocimiento semántico (placeholder)
        await self._update_semantic_knowledge(processed_data)

    async def _update_semantic_knowledge(self, processed_data: Dict[str, Any]) -> None:
        """Actualiza conocimiento semántico a largo plazo"""
        # Placeholder - implementación real requeriría análisis semántico
        key_concepts = processed_data.get("integrated", {}).get("active_modalities", [])

        for concept in key_concepts:
            if concept not in self.context.semantic_knowledge:
                self.context.semantic_knowledge[concept] = {
                    "frequency": 0,
                    "last_seen": time.time(),
                    "confidence": 0.5
                }

            self.context.semantic_knowledge[concept]["frequency"] += 1
            self.context.semantic_knowledge[concept]["last_seen"] = time.time()

    async def make_decision(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Toma una decisión basada en el estado cognitivo actual

        Args:
            context: Contexto adicional para la decisión

        Returns:
            Decisión tomada con justificación
        """
        self.context.state = CognitiveState.DECIDING
        episode_id = f"cog_decision_{self.episode_counter}"

        # Emitir evento de decisión
        trace_id = self.event_bus.emit(
            EventType.DECISION,
            "CognitiveCore",
            inputs={"context": context or {}, "cognitive_state": self.context.state.value},
            episode_id=episode_id,
            step_id=self.step_counter,
            explanation="Inicio del proceso de toma de decisión"
        )

        try:
            # Evaluar opciones disponibles
            options = await self._evaluate_decision_options(context)

            # Seleccionar mejor opción
            decision = await self._select_best_decision(options)

            # Registrar decisión
            self.context.last_decision = decision
            self.context.decision_timestamp = time.time()

            # Emitir decisión tomada
            self.event_bus.emit(
                EventType.DECISION,
                "CognitiveCore",
                outputs=decision,
                context={"options_evaluated": len(options)},
                episode_id=episode_id,
                step_id=self.step_counter,
                trace_id=trace_id,
                explanation=f"Decisión tomada: {decision.get('action', 'unknown')}"
            )

            self.step_counter += 1
            return decision

        except Exception as e:
            self.event_bus.emit(
                EventType.MODULE_INTERACTION,
                "CognitiveCore",
                outputs={"decision_success": False, "error": str(e)},
                episode_id=episode_id,
                step_id=self.step_counter,
                trace_id=trace_id,
                log_level=2
            )
            raise

    async def _evaluate_decision_options(self, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Evalúa opciones de decisión disponibles"""
        # Placeholder - implementación real requeriría lógica de decisión compleja
        options = [
            {
                "action": "process_further",
                "reasoning": "Más análisis requerido",
                "confidence": 0.7,
                "expected_outcome": "Mejor comprensión"
            },
            {
                "action": "make_conclusion",
                "reasoning": "Información suficiente disponible",
                "confidence": 0.8,
                "expected_outcome": "Decisión informada"
            },
            {
                "action": "request_clarification",
                "reasoning": "Información ambigua o incompleta",
                "confidence": 0.6,
                "expected_outcome": "Datos adicionales"
            }
        ]

        return options

    async def _select_best_decision(self, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Selecciona la mejor decisión de las opciones disponibles"""
        if not options:
            return {
                "action": "no_action",
                "reasoning": "No options available",
                "confidence": 0.0
            }

        # Seleccionar basado en confianza (placeholder)
        best_option = max(options, key=lambda x: x.get("confidence", 0))

        return {
            "action": best_option["action"],
            "reasoning": best_option["reasoning"],
            "confidence": best_option["confidence"],
            "expected_outcome": best_option.get("expected_outcome", "Unknown"),
            "decision_basis": "confidence_maximization"
        }

    async def evaluate_state(self) -> Dict[str, Any]:
        """
        Evalúa el estado cognitivo actual

        Returns:
            Evaluación del estado con métricas y recomendaciones
        """
        # Calcular métricas de estado
        confidence = self._calculate_overall_confidence()
        coherence = self._calculate_cognitive_coherence()
        attention_stability = self._calculate_attention_stability()

        # Generar recomendaciones
        recommendations = []

        if confidence < 0.5:
            recommendations.append("Aumentar procesamiento de datos para mejorar confianza")

        if coherence < 0.6:
            recommendations.append("Revisar integración multimodal para mejorar coherencia")

        if attention_stability < 0.7:
            recommendations.append("Estabilizar foco atencional")

        evaluation = {
            "confidence": confidence,
            "coherence": coherence,
            "attention_stability": attention_stability,
            "cognitive_load": len(self.context.working_memory),
            "episodic_memory_size": len(self.context.episodic_buffer),
            "recommendations": recommendations
        }

        return evaluation

    def _calculate_overall_confidence(self) -> float:
        """Calcula confianza general del sistema"""
        confidences = []

        # Confianza de modalidades procesadas
        if self.context.working_memory.get("last_processed"):
            processed = self.context.working_memory["last_processed"]
            for modality_data in processed.get("modalities", {}).values():
                if "confidence" in modality_data:
                    confidences.append(modality_data["confidence"])

        # Confianza de integración
        if self.context.working_memory.get("last_processed"):
            integrated = self.context.working_memory["last_processed"].get("integrated", {})
            if "integration_confidence" in integrated:
                confidences.append(integrated["integration_confidence"])

        return sum(confidences) / len(confidences) if confidences else 0.5

    def _calculate_cognitive_coherence(self) -> float:
        """Calcula coherencia cognitiva"""
        # Placeholder - implementación real requeriría análisis de consistencia
        if len(self.context.episodic_buffer) < 2:
            return 0.5

        # Calcular coherencia basada en consistencia de decisiones
        recent_decisions = [entry for entry in self.context.episodic_buffer[-10:]
                           if entry.get("context", {}).get("state") == "deciding"]

        if len(recent_decisions) < 2:
            return 0.5

        # Placeholder: coherencia basada en estabilidad
        return min(1.0, len(recent_decisions) / 10.0)

    def _calculate_attention_stability(self) -> float:
        """Calcula estabilidad atencional"""
        # Placeholder - implementación real requeriría tracking de atención
        return 0.8  # Valor fijo por ahora

    async def adapt_cognitively(self, feedback: Dict[str, Any]) -> None:
        """
        Adapta el comportamiento cognitivo basado en feedback

        Args:
            feedback: Feedback sobre rendimiento cognitivo
        """
        self.context.state = CognitiveState.ADAPTING

        # Emitir evento de adaptación
        self.event_bus.emit(
            EventType.LEARNING_STEP,
            "CognitiveCore",
            inputs={"feedback": feedback},
            episode_id=f"adaptation_{self.episode_counter}",
            explanation="Inicio de adaptación cognitiva"
        )

        # Adaptar foco atencional
        if "attention_target" in feedback:
            self.context.attention_focus = feedback["attention_target"]

        # Adaptar umbrales de confianza
        if "confidence_adjustment" in feedback:
            # Placeholder para ajuste de umbrales
            pass

        # Limpiar memoria si es necesario
        if feedback.get("memory_reset", False):
            self.context.working_memory.clear()
            self.context.episodic_buffer.clear()

        self.episode_counter += 1

    def get_status(self) -> Dict[str, Any]:
        """Obtiene estado actual del cognitive core"""
        return {
            "state": self.context.state.value,
            "confidence": self.context.confidence,
            "attention_focus": self.context.attention_focus,
            "working_memory_size": len(self.context.working_memory),
            "episodic_buffer_size": len(self.context.episodic_buffer),
            "semantic_concepts": len(self.context.semantic_knowledge),
            "episodes_processed": self.episode_counter,
            "steps_processed": self.step_counter
        }

    async def get_metrics(self) -> Metrics:
        """Obtiene métricas de rendimiento del cognitive core"""
        # Obtener evaluación del estado
        state_eval = await self.evaluate_state()
        
        # Calcular métricas adicionales
        import psutil
        import os
        
        # Memoria usada por el proceso
        process = psutil.Process(os.getpid())
        memory_usage_mb = process.memory_info().rss / 1024 / 1024
        
        # Latencia promedio (placeholder - necesitaríamos tracking real)
        avg_latency = 50.0  # ms, placeholder
        
        # Throughput (procesos por segundo)
        throughput = self.step_counter / max(1, time.time() - self.start_time) if hasattr(self, 'start_time') else 0.0
        
        # Accuracy basado en confianza
        accuracy = state_eval.get("confidence", 0.5)
        
        # Precision y recall (placeholders)
        precision = accuracy * 0.9
        recall = accuracy * 0.95
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        custom_metrics = {
            "cognitive_coherence": state_eval.get("coherence", 0.0),
            "attention_stability": state_eval.get("attention_stability", 0.0),
            "cognitive_load": state_eval.get("cognitive_load", 0),
            "episodic_memory_size": state_eval.get("episodic_memory_size", 0)
        }
        
        return Metrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            latency_ms=avg_latency,
            throughput_fps=throughput,
            memory_usage_mb=memory_usage_mb,
            custom_metrics=custom_metrics
        )

    def get_config(self) -> Dict[str, Any]:
        """Obtiene configuración del cognitive core"""
        return {
            "max_working_memory": getattr(self, 'max_working_memory', 100),
            "max_episodic_buffer": getattr(self, 'max_episodic_buffer', 1000),
            "attention_threshold": getattr(self, 'attention_threshold', 0.5),
            "learning_rate": getattr(self, 'learning_rate', 0.01),
            "adaptation_rate": getattr(self, 'adaptation_rate', 0.1),
            "modules": list(self.modules.keys()) if hasattr(self, 'modules') else []
        }