import cv2
import numpy as np
import time
from ultralytics import YOLO

class StrawberryDetector:
    def __init__(self, model_path="models/yolo_pose_best.pt"):
        """
        Инициализация модели YOLO Pose.
        При первом запуске TensorRT/ONNX может потребоваться время на "прогрев" (warmup).
        """
        try:
            self.model = YOLO(model_path)
            print(f"Модель {model_path} успешно загружена.")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            self.model = None

        # Словарь классов. В твоем датасете индексы могут отличаться, поменяй при необходимости.
        self.class_names = {0: "ripe", 1: "unripe"}

    def process_frame(self, frame: np.ndarray, conf_threshold: float = 0.5) -> tuple:
        """
        Основной метод инференса. 
        Принимает кадр (OpenCV BGR), возвращает размеченный кадр и список словарей с данными.
        """
        if self.model is None:
            return frame, {"error": "Модель не загружена", "detections": []}

        # Фиксируем время для расчета задержки (latency)
        start_time = time.time()

        # Запуск инференса
        # verbose=False отключает спам в консоль при каждом кадре
        results = self.model.predict(source=frame, conf=conf_threshold, verbose=False)
        result = results[0] # Берем первый результат (так как передаем один кадр)

        detections_data = []
        annotated_frame = frame.copy()
        
        # Счетчик для статистики
        stats = {"ripe": 0, "unripe": 0}

        # Если ничего не найдено, возвращаем пустой результат
        if not result.boxes or not result.keypoints:
            return annotated_frame, self._build_json([], stats, start_time)

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        # Получаем ключевые точки: формат (N, 3, 3) -> N объектов, 3 точки, (x, y, conf)
        keypoints = result.keypoints.data.cpu().numpy()

        for i in range(len(boxes)):
            x_min, y_min, x_max, y_max = map(int, boxes[i])
            conf = float(confs[i])
            cls_id = int(classes[i])
            ripeness = self.class_names.get(cls_id, "unknown")
            
            stats[ripeness] = stats.get(ripeness, 0) + 1

            # Извлекаем 3 точки: 0 - центр, 1 - шляпка, 2 - срез (согласно твоей разметке)
            # Внимание: убедись, что при разметке датасета порядок точек был именно таким
            pts = keypoints[i]
            kpt_center = {"x": float(pts[0][0]), "y": float(pts[0][1]), "conf": float(pts[0][2])}
            kpt_calyx  = {"x": float(pts[1][0]), "y": float(pts[1][1]), "conf": float(pts[1][2])}
            kpt_cut    = {"x": float(pts[2][0]), "y": float(pts[2][1]), "conf": float(pts[2][2])}

            # Флаг actionable: робот может резать только спелую клубнику с четко видимым срезом
            actionable = bool(ripeness == "ripe" and kpt_cut["conf"] > 0.6)

            det_dict = {
                "id": i + 1,
                "ripeness": ripeness,
                "confidence": round(conf, 2),
                "bounding_box": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
                "keypoints": {
                    "center": kpt_center,
                    "calyx": kpt_calyx,
                    "cut_point": kpt_cut
                },
                "actionable": actionable
            }
            detections_data.append(det_dict)

            # --- КАСТОМНАЯ ОТРИСОВКА ---
            color = (0, 255, 0) if ripeness == "ripe" else (0, 0, 255) # Зеленый для спелой, красный для неспелой
            
            # 1. Рамка и текст
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(annotated_frame, f"{ripeness} {conf:.2f}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 2. Ключевые точки и вектор среза
            # Отрисовываем точки, если нейросеть в них уверена
            valid_pts = []
            for name, kp in det_dict["keypoints"].items():
                if kp["conf"] > 0.3: # Порог видимости точки
                    px, py = int(kp["x"]), int(kp["y"])
                    valid_pts.append((px, py))
                    # Разные цвета для разных точек
                    pt_color = (255, 0, 0) if name == "center" else (0, 255, 255) if name == "calyx" else (255, 0, 255)
                    cv2.circle(annotated_frame, (px, py), 5, pt_color, -1)

            # Рисуем линию (вектор) от центра к шляпке и к точке среза
            if len(valid_pts) == 3:
                cv2.line(annotated_frame, valid_pts[0], valid_pts[1], (255, 255, 255), 2) # Центр -> Шляпка
                cv2.line(annotated_frame, valid_pts[1], valid_pts[2], (0, 255, 255), 2)   # Шляпка -> Срез

        # Формируем итоговый пакет данных
        final_payload = self._build_json(detections_data, stats, start_time)
        return annotated_frame, final_payload

    def _build_json(self, detections: list, stats: dict, start_time: float) -> dict:
        """Формирует итоговый словарь для отправки в UI и по сети"""
        latency_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "timestamp": time.time(),
            "latency_ms": latency_ms,
            "stats": stats,
            "detections": detections,
            "system_status": "active"
        }