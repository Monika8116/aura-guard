# main.py
from deepfake_detector_core.inference import load_model, preprocess_image, classify_image
from ar_phishing_detector.ocr_analysis import extract_text, detect_suspicious_keywords, detect_suspicious_urls
from ar_phishing_detector.yolo_ui_detector import detect_ui_elements
from voice_phishing_detector.transcriber import transcribe_audio
from voice_phishing_detector.phishing_nlp import detect_phishing_nlp

if __name__ == "__main__":
    model = load_model()
    image_path = "data/sample_images/sample1.jpg"  # your image
    input_tensor = preprocess_image(image_path)
    label, confidence = classify_image(model, input_tensor)

    print(f"Prediction: {label} ({confidence}%)")


image_path = "data/ar_screenshots/ar_sample1.jpg"

# OCR Analysis
text, raw_ocr = extract_text(image_path)
keywords = detect_suspicious_keywords(text)
urls = detect_suspicious_urls(text)

# YOLOv8 UI Detection
ui_detections = detect_ui_elements(image_path)

print("\n[OCR TEXT]:", text)
print("[Suspicious Keywords]:", keywords)
print("[Suspicious URLs]:", urls)
print("[YOLO UI Alerts]:", ui_detections)

audio_path = "data/voice_clips/phish_sample1.mp3"

# Step 1: Transcribe audio
transcript = transcribe_audio(audio_path)
print("\n[Transcript]:", transcript)

# Step 2: Phishing Detection
label, suspicious_phrases, raw_nlp = detect_phishing_nlp(transcript)
print(f"[Prediction]: {label}")
print(f"[Suspicious Phrases]: {suspicious_phrases}")
