# ar_phishing_detector/ui_analyzer.py

import cv2
import os
import shutil
from .yolo_ui_detector import YOLODetector
from .ocr_analysis import OCRAnalyzer


def extract_frames(video_path, frame_rate=15, max_frames=100):
    """
    Extract frames from video at specified rate with error handling and cleanup.
    """
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        frames = []
        index = 0
        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)

        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if index % frame_rate == 0:
                frame_path = os.path.join(temp_dir, f"frame_{index}.jpg")
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)
            index += 1

        cap.release()
        return frames

    except Exception as e:
        print(f"Error in frame extraction: {str(e)}")
        return []

    finally:
        if os.path.exists(temp_dir) and not frames:
            shutil.rmtree(temp_dir, ignore_errors=True)


def analyze_ui_anomalies(image_path):
    """
    Analyze a single image for UI-based phishing anomalies using YOLO and OCR.
    """
    try:
        # Initialize detectors
        yolo = YOLODetector()
        ocr = OCRAnalyzer()

        # Detect UI elements (YOLO)
        ui_elements = yolo.detect_ui_elements(image_path)

        # Extract text and OCR results
        text, ocr_results = ocr.extract_text(image_path)

        # Detect suspicious patterns
        suspicious_keywords = ocr.detect_suspicious_keywords(text)
        suspicious_urls = ocr.detect_suspicious_urls(text)

        # Compute phishing confidence
        confidence = 0.0
        if ui_elements:
            confidence += sum(item['confidence'] for item in ui_elements) * 0.4
        if suspicious_keywords:
            confidence += len(suspicious_keywords) * 0.3
        if suspicious_urls:
            confidence += len(suspicious_urls) * 0.2

        return {
            'ui_elements': ui_elements,
            'suspicious_keywords': suspicious_keywords,
            'suspicious_urls': suspicious_urls,
            'ocr_results': ocr_results,
            'confidence': min(confidence, 1.0),
            'is_phishing': confidence > 0.7
        }

    except Exception as e:
        print(f"Error analyzing image {image_path}: {str(e)}")
        return {'error': str(e), 'confidence': 0.0, 'is_phishing': False}


def analyze_video_ui(video_path):
    """
    Analyze video for phishing using multiple frame-based UI/OCR scans.
    """
    try:
        frames = extract_frames(video_path)
        if not frames:
            return {'error': 'No frames extracted', 'results': []}

        results = []
        total_confidence = 0.0
        phishing_frames = 0

        for path in frames:
            result = analyze_ui_anomalies(path)
            results.append((path, result))
            total_confidence += result['confidence']
            if result['is_phishing']:
                phishing_frames += 1

        shutil.rmtree("temp_frames", ignore_errors=True)

        avg_confidence = total_confidence / len(frames) if frames else 0.0
        is_phishing = avg_confidence > 0.7 or phishing_frames / len(frames) > 0.3

        return {
            'frame_results': results,
            'average_confidence': avg_confidence,
            'phishing_frames_ratio': phishing_frames / len(frames) if frames else 0.0,
            'is_phishing': is_phishing
        }

    except Exception as e:
        print(f"Error in video analysis: {str(e)}")
        return {'error': str(e), 'results': [], 'is_phishing': False}


def some_function():
    return None