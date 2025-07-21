# ar_phishing_detector/model.py
import torch
import timm


def load_xception_model():
    """
    Load Xception model optimized for phishing detection with fallback.
    """
    try:
        # Load Xception model using timm
        model = timm.create_model('xception', pretrained=True, num_classes=1)

        # Placeholder for fine-tuned weights (uncomment when available)
        # try:
        #     model.load_state_dict(torch.load('path_to_phishing_weights.pth'))
        # except FileNotFoundError:
        #     print("Fine-tuned weights not found, using pretrained model")

        model.eval()
        return model
    except Exception as e:
        print(f"Error loading Xception model: {str(e)}")
        # Fallback to EfficientNet-B0
        try:
            model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, num_classes=1)
            model.eval()
            return model
        except Exception as e2:
            print(f"Error loading fallback model: {str(e2)}")
            raise RuntimeError("Failed to load any model")