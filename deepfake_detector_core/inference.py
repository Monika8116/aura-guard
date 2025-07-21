# ar_phishing_detector/inference.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from ar_phishing_detector.ui_analyzer import analyze_ui_anomalies
from .model import load_xception_model


class PhishingClassifier:
    def __init__(self):
        self.model = load_xception_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(p=0.3),  # Reduced augmentation for stability
            transforms.RandomRotation(5),  # Tighter rotation for UI consistency
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path):
        """Preprocess image with robust error handling and optimized augmentation."""
        try:
            img = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)
            return input_tensor
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            return None

    def classify_image(self, image_path):
        """Classify image using ensemble of deep learning and UI analysis."""
        try:
            input_tensor = self.preprocess_image(image_path)
            if input_tensor is None:
                return {
                    'label': 'Error',
                    'confidence': 0.0,
                    'is_phishing': False,
                    'error': 'Image preprocessing failed'
                }

            # Deep learning classification
            with torch.no_grad():
                output = self.model(input_tensor)
                dl_score = torch.sigmoid(output[0][0]).item()

            # UI analysis
            ui_results = analyze_ui_anomalies(image_path)
            if 'error' in ui_results:
                ui_confidence = 0.0
            else:
                ui_confidence = ui_results.get('confidence', 0.0)

            # Ensemble scoring with adjusted weights
            final_confidence = (
                    dl_score * 0.6 +  # Increased weight for deep learning
                    ui_confidence * 0.4  # UI analysis contribution
            )
            label = "Phishing" if final_confidence > 0.65 else "Legitimate"  # Lowered threshold
            is_phishing = final_confidence > 0.65

            return {
                'label': label,
                'confidence': round(final_confidence * 100, 2),
                'is_phishing': is_phishing,
                'deep_learning_score': round(dl_score * 100, 2),
                'ui_anomalies': ui_results
            }

        except Exception as e:
            print(f"Error classifying image {image_path}: {str(e)}")
            return {
                'label': 'Error',
                'confidence': 0.0,
                'is_phishing': False,
                'error': str(e)
            }

    def classify_video(self, video_path):
        """Classify video by aggregating frame-level results."""
        from .ui_analyzer import analyze_video_ui
        try:
            video_results = analyze_video_ui(video_path)
            if 'error' in video_results:
                return {
                    'label': 'Error',
                    'confidence': 0.0,
                    'is_phishing': False,
                    'error': video_results['error']
                }

            # Aggregate frame-level classifications
            frame_results = video_results['frame_results']
            total_confidence = 0.0
            phishing_count = 0

            for _, frame_result in frame_results:
                # Run deep learning on select frames for efficiency
                frame_confidence = frame_result.get('confidence', 0.0)
                if frame_result.get('is_phishing', False):
                    input_tensor = self.preprocess_image(frame_result[0])
                    if input_tensor is not None:
                        with torch.no_grad():
                            output = self.model(input_tensor)
                            dl_score = torch.sigmoid(output[0][0]).item()
                        frame_confidence = (frame_confidence * 0.4 + dl_score * 0.6)

                total_confidence += frame_confidence
                if frame_confidence > 0.65:
                    phishing_count += 1

            avg_confidence = total_confidence / len(frame_results) if frame_results else 0.0
            is_phishing = avg_confidence > 0.65 or phishing_count / len(frame_results) > 0.25

            return {
                'label': 'Phishing' if is_phishing else 'Legitimate',
                'confidence': round(avg_confidence * 100, 2),
                'phishing_frames_ratio': phishing_count / len(frame_results) if frame_results else 0.0,
                'is_phishing': is_phishing,
                'frame_results': video_results['frame_results']
            }

        except Exception as e:
            print(f"Error classifying video {video_path}: {str(e)}")
            return {
                'label': 'Error',
                'confidence': 0.0,
                'is_phishing': False,
                'error': str(e)
            }