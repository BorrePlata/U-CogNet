#!/usr/bin/env python3
"""
Crear un video de prueba con armas para validar la detección mejorada.
"""

import cv2
import numpy as np
import os

def create_weapon_test_video():
    """Crear un video de prueba con escenas de armas."""

    # Configuración del video
    width, height = 640, 480
    fps = 10
    duration = 5  # segundos
    total_frames = fps * duration

    # Crear directorio si no existe
    os.makedirs('test_videos', exist_ok=True)
    output_path = 'test_videos/weapon_test.avi'

    # Crear video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"🎬 Creando video de prueba: {output_path}")

    for frame_num in range(total_frames):
        # Crear frame base
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (240, 240, 240)  # Fondo gris claro

        # Escena 1: Frames 0-15 (1.5s) - Persona sola
        if frame_num < 15:
            # Dibujar persona
            cv2.rectangle(frame, (250, 150), (350, 350), (0, 255, 0), 3)
            cv2.putText(frame, 'PERSON', (260, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Escena 2: Frames 15-30 (1.5s) - Persona con cuchillo cerca
        elif frame_num < 30:
            # Persona
            cv2.rectangle(frame, (250, 150), (350, 350), (0, 255, 0), 3)
            cv2.putText(frame, 'PERSON', (260, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Cuchillo cerca (en zona de mano)
            cv2.rectangle(frame, (220, 250), (250, 280), (0, 0, 255), 3)
            cv2.putText(frame, 'KNIFE', (225, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Escena 3: Frames 30-45 (1.5s) - Persona con bate lejos
        else:
            # Persona
            cv2.rectangle(frame, (250, 150), (350, 350), (0, 255, 0), 3)
            cv2.putText(frame, 'PERSON', (260, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Bate lejos
            cv2.rectangle(frame, (500, 200), (580, 230), (0, 0, 255), 3)
            cv2.putText(frame, 'BAT', (510, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Agregar timestamp
        cv2.putText(frame, f'Frame: {frame_num}/{total_frames-1}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Escribir frame al video
        out.write(frame)

    # Liberar recursos
    out.release()
    print(f"✅ Video creado exitosamente: {output_path}")
    print(f"   Duración: {duration}s, Frames: {total_frames}, FPS: {fps}")
    print("   Escenas:")
    print("   - 0-1.5s: Persona sola")
    print("   - 1.5-3s: Persona con cuchillo cerca (debe detectar ARMADO)")
    print("   - 3-5s: Persona con bate lejos (no debe detectar ARMADO)")

if __name__ == "__main__":
    create_weapon_test_video()