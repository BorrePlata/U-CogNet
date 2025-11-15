import cv2
import numpy as np

def create_test_video_with_objects():
    """Crear un video de prueba con objetos detectables por YOLO."""
    width, height = 640, 480
    fps = 30
    duration = 5  # 5 segundos

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_objects.mp4', fourcc, fps, (width, height))

    # Crear frames con objetos simples (rectángulos que simulen personas/autos)
    for frame_num in range(fps * duration):
        # Frame base negro
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Agregar "persona" (rectángulo blanco)
        cv2.rectangle(frame, (100 + frame_num*2, 200), (150 + frame_num*2, 300), (255, 255, 255), -1)

        # Agregar "auto" (rectángulo azul)
        cv2.rectangle(frame, (400 - frame_num, 250), (500 - frame_num, 320), (255, 0, 0), -1)

        # Agregar texto para simular detección
        cv2.putText(frame, f"Frame {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)

    out.release()
    print("Video de prueba creado: test_objects.mp4")

if __name__ == "__main__":
    create_test_video_with_objects()