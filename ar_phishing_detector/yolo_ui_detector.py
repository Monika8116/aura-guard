from ultralytics import YOLO
import cv2
import numpy as np


class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize YOLOv8 model for UI anomaly detection.
        """
        self.model = YOLO(model_path)
        self.suspicious_classes = [
            'keyboard', 'screen', 'cell phone', 'button', 'input_field',
            'login_form', 'password_field', 'qr_code', 'popup'
        ]

    def preprocess_image(self, image_path):
        """
        Preprocess image: load, enhance contrast/brightness.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Enhance contrast and brightness
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        return img

    def detect_ui_elements(self, image_path):
        """
        Detect suspicious UI elements from image using YOLO.
        """
        try:
            img = self.preprocess_image(image_path)
            results = self.model(img)
            detections = results[0].boxes
            suspicious_items = []

            for box in detections:
                cls_id = int(box.cls)
                cls_name = results[0].names[cls_id]

                if cls_name in self.suspicious_classes:
                    conf = float(box.conf)
                    if conf > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        area = (x2 - x1) * (y2 - y1)
                        suspicious_items.append({
                            'label': cls_name,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2],
                            'area': float(area)
                        })

            # Sort and return top 5 by confidence
            suspicious_items.sort(key=lambda x: x['confidence'], reverse=True)
            return suspicious_items[:5]

        except Exception as e:
            print(f"Error in YOLO detection: {str(e)}")
            return []
