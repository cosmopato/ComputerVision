"""
================================================================================
DETECCIÓN DE LÍNEAS DE CARRIL
Dataset: Udacity Self-Driving Car Nanodegree
================================================================================
Proyecto Individual - Fundamentos de la Visión por Computador
Universidad de Deusto
================================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
import time

class LaneDetector:
    """
    Detector de líneas de carril usando técnicas clásicas de visión por computador.
    
    Pipeline:
    1. Conversión a escala de grises
    2. Suavizado Gaussiano
    3. Detección de bordes (Canny)
    4. Región de Interés (ROI)
    5. Transformada de Hough
    6. Separación izquierda/derecha por pendiente
    7. Regresión lineal
    8. Suavizado temporal
    """
    
    def __init__(self, buffer_size=10):
        self.left_lines_buffer = deque(maxlen=buffer_size)
        self.right_lines_buffer = deque(maxlen=buffer_size)
        
        # Parámetros Canny
        self.canny_low = 50
        self.canny_high = 150
        
        # Parámetros Hough
        self.hough_rho = 2
        self.hough_theta = np.pi/180
        self.hough_threshold = 50
        self.hough_min_line_length = 40
        self.hough_max_line_gap = 100
        
    def process_frame(self, frame):
        """Procesa un frame completo del video."""
        height, width = frame.shape[:2]
        
        # 1. Escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Suavizado Gaussiano
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Detección de bordes Canny
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # 4. Región de interés (trapecio)
        mask = np.zeros_like(edges)
        vertices = np.array([[
            (int(width * 0.05), height),
            (int(width * 0.45), int(height * 0.6)),
            (int(width * 0.55), int(height * 0.6)),
            (int(width * 0.95), height)
        ]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, 255)
        roi_edges = cv2.bitwise_and(edges, mask)
        
        # 5. Transformada de Hough
        lines = cv2.HoughLinesP(roi_edges, self.hough_rho, self.hough_theta, 
                                self.hough_threshold, minLineLength=self.hough_min_line_length, 
                                maxLineGap=self.hough_max_line_gap)
        
        # 6. Separar líneas izquierda/derecha
        left_lines, right_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.5:
                    continue
                if slope < 0 and (x1 + x2) / 2 < width / 2:
                    left_lines.append(line[0])
                elif slope > 0 and (x1 + x2) / 2 > width / 2:
                    right_lines.append(line[0])
        
        # 7. Ajustar líneas por regresión
        left_line = self._fit_line(left_lines, height)
        right_line = self._fit_line(right_lines, height)
        
        # 8. Suavizado temporal
        left_line = self._average_line(left_line, self.left_lines_buffer)
        right_line = self._average_line(right_line, self.right_lines_buffer)
        
        # Dibujar resultado
        result = self._draw_lanes(frame.copy(), left_line, right_line)
        
        return result, edges, roi_edges
    
    def _fit_line(self, lines, img_height):
        if not lines:
            return None
        points_x, points_y = [], []
        for x1, y1, x2, y2 in lines:
            points_x.extend([x1, x2])
            points_y.extend([y1, y2])
        if len(points_x) < 2:
            return None
        try:
            m, b = np.polyfit(points_x, points_y, 1)
            if m == 0:
                return None
            y_bottom, y_top = img_height, int(img_height * 0.6)
            x_bottom = int((y_bottom - b) / m)
            x_top = int((y_top - b) / m)
            return (x_bottom, y_bottom, x_top, y_top)
        except:
            return None
    
    def _average_line(self, current_line, buffer):
        if current_line is not None:
            buffer.append(current_line)
        if not buffer:
            return None
        return tuple(np.mean(buffer, axis=0).astype(int))
    
    def _draw_lanes(self, img, left_line, right_line):
        if left_line:
            cv2.line(img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 0), 10)
        if right_line:
            cv2.line(img, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 10)
        if left_line and right_line:
            pts = np.array([[left_line[0], left_line[1]], [left_line[2], left_line[3]],
                           [right_line[2], right_line[3]], [right_line[0], right_line[1]]], np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
        return img
    
    def reset(self):
        self.left_lines_buffer.clear()
        self.right_lines_buffer.clear()


def visualize_pipeline(frame, save_path='pipeline_visualization.png'):
    """Genera visualización de todas las etapas del pipeline."""
    detector = LaneDetector()
    height, width = frame.shape[:2]
    
    # Ejecutar cada paso
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    mask = np.zeros_like(edges)
    vertices = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.45), int(height * 0.6)),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.95), height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    roi_edges = cv2.bitwise_and(edges, mask)
    
    # Hough lines visualization
    lines = cv2.HoughLinesP(roi_edges, 2, np.pi/180, 50, minLineLength=40, maxLineGap=100)
    hough_img = np.zeros((height, width, 3), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    result, _, _ = detector.process_frame(frame)
    
    # Crear figura
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Pipeline de Detección de Líneas de Carril - Dataset Udacity', fontsize=14, fontweight='bold')
    
    axes[0,0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title('1. Imagen Original')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(gray, cmap='gray')
    axes[0,1].set_title('2. Escala de Grises')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(blurred, cmap='gray')
    axes[0,2].set_title('3. Filtro Gaussiano')
    axes[0,2].axis('off')
    
    axes[0,3].imshow(edges, cmap='gray')
    axes[0,3].set_title('4. Bordes (Canny)')
    axes[0,3].axis('off')
    
    roi_vis = frame.copy()
    cv2.polylines(roi_vis, vertices, True, (0, 0, 255), 3)
    axes[1,0].imshow(cv2.cvtColor(roi_vis, cv2.COLOR_BGR2RGB))
    axes[1,0].set_title('5. Región de Interés (ROI)')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(roi_edges, cmap='gray')
    axes[1,1].set_title('6. Bordes en ROI')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(cv2.cvtColor(hough_img, cv2.COLOR_BGR2RGB))
    axes[1,2].set_title('7. Líneas Hough')
    axes[1,2].axis('off')
    
    axes[1,3].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1,3].set_title('8. Resultado Final')
    axes[1,3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualización guardada: {save_path}")
    return fig


def analyze_performance(detector, video_path, max_frames=100):
    """Analiza métricas de rendimiento del detector."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    detector.reset()
    times = []
    detections_left = 0
    detections_right = 0
    total = 0
    
    while total < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.time()
        detector.process_frame(frame)
        times.append(time.time() - start)
        
        if len(detector.left_lines_buffer) > 0:
            detections_left += 1
        if len(detector.right_lines_buffer) > 0:
            detections_right += 1
        total += 1
    
    cap.release()
    
    return {
        'frames': total,
        'avg_time_ms': np.mean(times) * 1000,
        'std_time_ms': np.std(times) * 1000,
        'fps': 1 / np.mean(times),
        'detection_left': (detections_left / total) * 100,
        'detection_right': (detections_right / total) * 100
    }


def process_video(input_path, output_path='output_lanes.avi'):
    """Procesa un video completo y guarda el resultado."""
    detector = LaneDetector()
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"ERROR: No se puede abrir '{input_path}'")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total} frames")
    
    # Codec XVID (más compatible)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print("AVISO: Guardando frames como imágenes...")
        os.makedirs('frames_output', exist_ok=True)
        save_as_images = True
    else:
        save_as_images = False
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result, _, _ = detector.process_frame(frame)
        
        if save_as_images:
            cv2.imwrite(f'frames_output/frame_{count:04d}.png', result)
        else:
            writer.write(result)
        
        count += 1
        if count % 50 == 0:
            print(f"Procesando: {count}/{total} frames ({100*count/total:.1f}%)")
    
    cap.release()
    if not save_as_images:
        writer.release()
    
    print(f"\n✓ Completado: {count} frames procesados")
    print(f"✓ Guardado en: {output_path}")
    return True


def main():
    """Función principal."""
    print("=" * 70)
    print("DETECTOR DE LÍNEAS DE CARRIL")
    print("Dataset: Udacity Self-Driving Car Nanodegree")
    print("=" * 70)
    
    # Buscar videos de Udacity
    udacity_videos = ['solidWhiteRight.mp4', 'solidYellowLeft.mp4', 'challenge.mp4']
    video_path = None
    
    for v in udacity_videos:
        if os.path.exists(v):
            video_path = v
            print(f"\n✓ Encontrado video de Udacity: {v}")
            break
    
    # Si no hay videos de Udacity, buscar cualquier video
    if not video_path:
        for f in os.listdir('.'):
            if f.endswith(('.mp4', '.avi', '.mov')):
                video_path = f
                break
    
    if not video_path:
        print("\n" + "=" * 70)
        print("ERROR: No se encontró ningún video")
        print("=" * 70)
        print("\nPor favor, descarga los videos de Udacity:")
        print("\n1. Ve a: https://github.com/udacity/CarND-LaneLines-P1")
        print("2. Navega a la carpeta 'test_videos'")
        print("3. Descarga: solidWhiteRight.mp4 y/o solidYellowLeft.mp4")
        print("4. Colócalos en la misma carpeta que este script")
        print("5. Ejecuta de nuevo: python deteccion_carriles.py")
        print("\nO descarga directamente:")
        print("  https://github.com/udacity/CarND-LaneLines-P1/raw/master/test_videos/solidWhiteRight.mp4")
        print("  https://github.com/udacity/CarND-LaneLines-P1/raw/master/test_videos/solidYellowLeft.mp4")
        return
    
    print(f"\nUsando video: {video_path}")
    
    # Cargar un frame para visualización
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        print("\n[1/3] Generando visualización del pipeline...")
        visualize_pipeline(frame)
        
        print("\n[2/3] Analizando rendimiento...")
        detector = LaneDetector()
        metrics = analyze_performance(detector, video_path)
        
        if metrics:
            print("\n" + "=" * 50)
            print("MÉTRICAS DE RENDIMIENTO")
            print("=" * 50)
            print(f"  Frames analizados:        {metrics['frames']}")
            print(f"  Tiempo por frame:         {metrics['avg_time_ms']:.2f} ± {metrics['std_time_ms']:.2f} ms")
            print(f"  FPS procesamiento:        {metrics['fps']:.1f}")
            print(f"  Detección línea izq:      {metrics['detection_left']:.1f}%")
            print(f"  Detección línea der:      {metrics['detection_right']:.1f}%")
            print("=" * 50)
        
        print("\n[3/3] Procesando video completo...")
        output_name = 'output_' + os.path.basename(video_path).replace('.mp4', '.avi')
        process_video(video_path, output_name)
    
    print("\n" + "=" * 70)
    print("¡PROCESAMIENTO COMPLETADO!")
    print("=" * 70)


if __name__ == "__main__":
    main()