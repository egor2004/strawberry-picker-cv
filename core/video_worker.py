import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

# Импортируем наш детектор
from core.inference import StrawberryDetector

class VideoWorker(QThread):
    # Определяем сигналы для связи с главным потоком (UI)
    frame_ready = pyqtSignal(QImage)  # Сигнал передает готовую картинку
    data_ready = pyqtSignal(dict)     # Сигнал передает JSON с координатами
    error_occurred = pyqtSignal(str)  # Сигнал для передачи текста ошибок

    def __init__(self, source, detector: StrawberryDetector):
        super().__init__()
        self.source = source          # 0 (веб-камера), путь к mp4/jpg, или RTSP ссылка
        self.detector = detector      # Экземпляр нейросети
        self.is_running = True        # Флаг работы цикла
        self.confidence = 0.5         # Порог уверенности по умолчанию

    def run(self):
        """
        Этот метод автоматически запускается в отдельном потоке при вызове worker.start()
        """
        # OpenCV универсален: он одинаково читает и веб-камеры, и видео, и RTSP
        cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            self.error_occurred.emit(f"Не удалось открыть источник: {self.source}")
            return

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break  # Конец видео или обрыв потока

            # 1. Отдаем кадр нейросети на обработку
            annotated_frame, json_data = self.detector.process_frame(
                frame, conf_threshold=self.confidence
            )

            # 2. Конвертируем цвета из BGR (формат OpenCV) в RGB (формат PyQt)
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            
            # Создаем объект QImage для передачи в интерфейс
            qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

            # 3. Отправляем сигналы в главное окно
            self.frame_ready.emit(qt_img)
            self.data_ready.emit(json_data)

            # Небольшая пауза, чтобы не забивать процессор на 100%, 
            # если локальное видео читается слишком быстро (быстрее реального времени)
            self.msleep(10) 

        # Очищаем ресурсы при остановке
        cap.release()

    def stop(self):
        """Метод для безопасной остановки потока"""
        self.is_running = False
        self.wait()  # Блокируем вызывающий поток, пока этот полностью не завершится

    def set_confidence(self, conf: float):
        """Метод для изменения порога уверенности 'на лету' с ползунка интерфейса"""
        self.confidence = conf