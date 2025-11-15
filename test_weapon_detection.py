#!/usr/bin/env python3
"""
Script de prueba para validar las mejoras en la detección de armas de U-CogNet.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ucognet.modules.vision.yolov8_detector import YOLOv8Detector
from ucognet.core.types import Frame
import cv2
import numpy as np

def test_weapon_detection():
    """Probar la detección de armas con diferentes escenarios."""

    print("🧪 Probando detección de armas mejorada...")

    # Inicializar detector
    detector = YOLOv8Detector(conf_threshold=0.5)

    # Crear frames de prueba
    test_frames = []

    # Frame 1: Persona con cuchillo cerca (debe detectar persona armada)
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame1[:] = (100, 100, 100)

    # Dibujar persona (bounding box realista)
    cv2.rectangle(frame1, (200, 100), (300, 400), (0, 255, 0), 2)

    # Dibujar cuchillo en zona de mano (debe detectar como armado)
    cv2.rectangle(frame1, (170, 250), (200, 280), (0, 0, 255), 2)

    test_frames.append(("Persona con cuchillo en mano", frame1))

    # Frame 2: Persona con tijeras lejos (no debe detectar como armado)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2[:] = (100, 100, 100)

    cv2.rectangle(frame2, (200, 100), (300, 400), (0, 255, 0), 2)
    cv2.rectangle(frame2, (500, 200), (530, 230), (0, 0, 255), 2)

    test_frames.append(("Persona con tijeras lejos", frame2))

    # Frame 3: Solo armas sin personas
    frame3 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame3[:] = (100, 100, 100)

    cv2.rectangle(frame3, (150, 200), (180, 230), (0, 0, 255), 2)  # Cuchillo
    cv2.rectangle(frame3, (400, 200), (430, 230), (0, 0, 255), 2)  # Bate

    test_frames.append(("Solo armas sin personas", frame3))

    # Probar cada frame
    for test_name, frame_data in test_frames:
        print(f"\n📋 Probando: {test_name}")

        # Crear objeto Frame
        frame = Frame(data=frame_data, timestamp=0.0, metadata={})

        # Detectar
        detections = detector.detect(frame)

        # Analizar resultados
        persons = [d for d in detections if d.class_name == 'person']
        weapons = [d for d in detections if hasattr(d, 'is_weapon') and d.is_weapon]
        armed_persons = [d for d in detections if d.class_name == 'person' and hasattr(d, 'is_armed') and d.is_armed]

        print(f"   👥 Personas detectadas: {len(persons)}")
        print(f"   ⚔️  Armas detectadas: {len(weapons)}")
        print(f"   🚨 Personas armadas: {len(armed_persons)}")

        for weapon in weapons:
            print(f"      - {weapon.class_name} (conf: {weapon.confidence:.2f})")

        for armed in armed_persons:
            weapon_name = getattr(armed, 'nearby_weapon', 'desconocido')
            distance = getattr(armed, 'weapon_distance', 0)
            print(f"      - Persona armada con {weapon_name} (dist: {distance:.1f}px)")

    print("\n✅ Prueba completada")

if __name__ == "__main__":
    test_weapon_detection()