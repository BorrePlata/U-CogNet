# /mnt/c/Users/desar/Documents/Science/UCogNet/src/ucognet/modules/mycelium/__init__.py
"""
MycoNet - Sistema Nervioso Micelial de U-CogNet

MycoNet es la capa de optimización, atención y comunicación distribuida de U-CogNet,
inspirada en el comportamiento de las redes miceliales de los hongos. Actúa como un
sistema nervioso subterráneo que conecta módulos cognitivos, gestiona recursos,
y facilita la autoevolución controlada del sistema.

Componentes principales:
- MycoNode: Nodos que representan módulos cognitivos
- MycoEdge: Conexiones con feromonas entre nodos
- MycoNet: Grafo vivo que rutea atención y recursos
- Integración completa con seguridad cognitiva
"""

from .types import MycoContext, MycoSignal, MycoPath, MycoMetrics
from .core import MycoNode, MycoEdge, MycoNet
from .optimizer import MetaMushMind

__all__ = [
    'MycoContext',
    'MycoSignal',
    'MycoPath',
    'MycoMetrics',
    'MycoNode',
    'MycoEdge',
    'MycoNet',
    'MetaMushMind'
]