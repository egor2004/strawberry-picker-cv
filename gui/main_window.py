import sys
import cv2
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QLineEdit, 
                             QGroupBox, QFormLayout, QSlider, QTextEdit,
                             QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap

# Импортируем наши модули
from core.inference import StrawberryDetector
from core.video_worker import VideoWorker
from network.transmitter import RobotTransmitter

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ПП ОСК v1.0 — Система детекции клубники")
        self.resize(1200, 800)

        # 1. Инициализируем детектор
        self.detector = StrawberryDetector(model_path="models/best.pt")
        self.worker = None # Сюда будем сохранять поток видео
        # Сетевой передатчик (отправляет на локальный порт 5005 для тестов)
        self.transmitter = RobotTransmitter(host="127.0.0.1", port=5005)

        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Левая часть
        self.left_layout = QVBoxLayout()
        self.video_display = QLabel("Ожидание источника данных...")
        self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display.setStyleSheet("background-color: black; color: gray; border: 2px solid #333;")
        self.video_display.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.left_layout.addWidget(self.video_display, stretch=4)

        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas; font-size: 10pt;")
        self.left_layout.addWidget(self.console_output, stretch=1)

        # Правая часть
        self.right_panel = QVBoxLayout()
        
        # Группы управления
        self.create_source_group()
        self.create_params_group()
        self.create_stats_group()

        self.btn_stop = QPushButton("ОСТАНОВИТЬ")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background-color: #444; color: white; padding: 10px;")

        self.right_panel.addWidget(self.source_group)
        self.right_panel.addWidget(self.params_group)
        self.right_panel.addWidget(self.stats_group)
        self.right_panel.addStretch()
        self.right_panel.addWidget(self.btn_stop)

        self.main_layout.addLayout(self.left_layout, stretch=3)
        self.main_layout.addLayout(self.right_panel, stretch=1)

    def create_source_group(self):
        self.source_group = QGroupBox("Источник данных")
        layout = QVBoxLayout()
        self.btn_photo = QPushButton("Открыть фото")
        self.btn_video = QPushButton("Открыть видео")
        self.btn_webcam = QPushButton("Веб-камера")
        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("rtsp://...")
        self.btn_rtsp = QPushButton("Подключить RTSP")
        
        for w in [self.btn_photo, self.btn_video, self.btn_webcam, self.rtsp_input, self.btn_rtsp]:
            layout.addWidget(w)
        self.source_group.setLayout(layout)

    def create_params_group(self):
        self.params_group = QGroupBox("Параметры")
        layout = QFormLayout()
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(50)
        self.conf_label = QLabel("0.50")
        layout.addRow("Confidence:", self.conf_slider)
        layout.addRow("Значение:", self.conf_label)
        self.params_group.setLayout(layout)

    def create_stats_group(self):
        self.stats_group = QGroupBox("Статистика")
        layout = QVBoxLayout()
        self.label_ripe = QLabel("Спелых: 0")
        self.label_unripe = QLabel("Неспелых: 0")
        self.label_fps = QLabel("Задержка (ms): 0")
        layout.addWidget(self.label_ripe)
        layout.addWidget(self.label_unripe)
        layout.addWidget(self.label_fps)
        self.stats_group.setLayout(layout)

    def connect_signals(self):
        self.btn_photo.clicked.connect(self.process_static_image)
        self.btn_video.clicked.connect(lambda: self.start_worker("video"))
        self.btn_webcam.clicked.connect(lambda: self.start_worker("webcam"))
        self.btn_rtsp.clicked.connect(lambda: self.start_worker("rtsp"))
        self.btn_stop.clicked.connect(self.stop_worker)
        self.conf_slider.valueChanged.connect(self.update_conf)


    def update_conf(self):
        val = self.conf_slider.value() / 100.0
        self.conf_label.setText(f"{val:.2f}")
        if self.worker:
            self.worker.set_confidence(val)

    def process_static_image(self):
        """Обработка одного фото без создания потока"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Открыть фото", "", "Images (*.jpg *.png *.jpeg)")
        if not file_path: return

        frame = cv2.imread(file_path)
        if frame is not None:
            # Запускаем инференс напрямую
            annotated, data = self.detector.process_frame(frame, self.conf_slider.value()/100.0)
            
            # Конвертируем для UI
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
            self.update_image(qt_img)
            self.update_data(data)
            self.log(f"Обработано фото: {file_path}")

    def start_worker(self, source_type):
        """Запуск асинхронного потока для видео/камеры"""
        if self.worker and self.worker.isRunning():
            self.stop_worker()

        source = 0 # По умолчанию веб-камера
        if source_type == "video":
            source, _ = QFileDialog.getOpenFileName(self, "Открыть видео", "", "Video (*.mp4 *.avi)")
            if not source: return
        elif source_type == "rtsp":
            source = self.rtsp_input.text()
            if not source: return

        # Создаем и запускаем воркер
        self.worker = VideoWorker(source, self.detector)
        self.worker.set_confidence(self.conf_slider.value() / 100.0)
        
        # Соединяем сигналы воркера со слотами окна
        self.worker.frame_ready.connect(self.update_image)
        self.worker.data_ready.connect(self.update_data)
        self.worker.error_occurred.connect(self.log)
        
        self.worker.start()
        
        self.btn_stop.setEnabled(True)
        self.btn_stop.setStyleSheet("background-color: #aa0000; color: white;")
        self.log(f"Запущен поток: {source_type}")

    def stop_worker(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background-color: #444; color: white;")
        self.log("Поток остановлен.")

    @pyqtSlot(QImage)
    def update_image(self, qt_img):
        self.video_display.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    @pyqtSlot(dict)
    def update_data(self, data):
        """Обновление статистики и вывод JSON в консоль"""
        stats = data.get("stats", {})
        self.label_ripe.setText(f"Спелых: {stats.get('ripe', 0)}")
        self.label_unripe.setText(f"Неспелых: {stats.get('unripe', 0)}")
        self.label_fps.setText(f"Задержка: {data.get('latency_ms', 0)} ms")

        # Выводим в консоль и ОТПРАВЛЯЕМ ПО СЕТИ, если есть детекции
        if data.get("detections"):
            import json
            # 1. Реальная отправка пакета роботу
            self.transmitter.send(data)
            
            # 2. Логирование для интерфейса (берем первую попавшуюся ягоду для примера)
            sample = data["detections"][0]
            if sample["actionable"]:
                point = sample['keypoints']['cut_point']
                self.log(f"UDP SEND -> Срез ягоды {sample['id']}: X={point['x']:.1f}, Y={point['y']:.1f}")

    def log(self, message):
        self.console_output.append(message)
        # Автопрокрутка вниз
        self.console_output.verticalScrollBar().setValue(self.console_output.verticalScrollBar().maximum())