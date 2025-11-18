
"""
Protocolos e interfaces core para U-CogNet
Según arquitectura detallada
"""

from typing import Protocol, List, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np

# Protocolos de módulos principales
class InputHandlerProtocol(Protocol):
    """Protocolo para manejadores de entrada"""
    def get_frame(self) -> np.ndarray:
        ...
    
    def get_frame_with_metadata(self) -> tuple:
        ...

class VisionDetectorProtocol(Protocol):
    """Protocolo para detectores de visión"""
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        ...
    
    def detect_military_targets(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        ...

class CognitiveCoreProtocol(Protocol):
    """Protocolo para núcleo cognitivo"""
    def store(self, event) -> None:
        ...
    
    def get_context(self) -> Dict[str, Any]:
        ...

class AudioProcessorProtocol(Protocol):
    """Protocolo para procesamiento de audio"""
    def extract_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        ...
    
    def extract_from_video(self, video_path: str) -> np.ndarray:
        ...

class SecurityProtocol(Protocol):
    """Protocolo para módulos de seguridad"""
    def sanitize(self, data: Any) -> Any:
        ...
    
    def validate_ethics(self, action: Dict[str, Any]) -> bool:
        ...

# Clases abstractas base
class BaseModule(ABC):
    """Clase base para todos los módulos"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Inicializar el módulo"""
        pass
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Procesar datos de entrada"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Obtener estado del módulo"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Limpiar recursos"""
        pass

class AudioModule(BaseModule):
    """Módulo base para procesamiento de audio"""
    
    @abstractmethod
    def extract_audio_from_video(self, video_path: str, output_path: str) -> bool:
        """Extraer audio de video"""
        pass
    
    @abstractmethod
    def analyze_audio_content(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Analizar contenido del audio"""
        pass
