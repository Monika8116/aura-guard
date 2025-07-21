# voice_phishing_detector/phishing_nlp.py
from transformers import pipeline
import re
import transformers
from transformers import pipeline

assert hasattr(transformers, "pipeline"), "Transformers version too old!"


class PhishingNLPDetector:
    def __init__(self):
        # Use a DistilBERT model fine-tuned for phishing detection (placeholder for custom model)
        try:
            self.classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased",  # Replace with fine-tuned model when available
                tokenizer="distilbert-base-uncased",
                top_k=None
            )
        except Exception as e:
            print(f"Error loading NLP model: {str(e)}")
            # Fallback to a simpler model
            self.classifier = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)

        self.phishing_keywords = [
            "verify", "urgent", "account", "transfer", "otp", "security alert",
            "support team", "click here", "update your details", "password",
            "login", "bank", "payment", "confirm", "access"
        ]
        self.context_phrases = [
            "immediately", "now", "required", "secure", "critical", "action needed"
        ]

    def detect_phishing_nlp(self, text):
        """Detect phishing in text using NLP model and keyword analysis."""
        try:
            if not text or not isinstance(text, str):
                return {
                    'label': 'Error',
                    'confidence': 0.0,
                    'is_phishing': False,
                    'matches': [],
                    'error': 'Invalid or empty text input'
                }

            # Limit text to 512 tokens for transformer models
            text = text[:512].lower()

            # NLP model prediction
            try:
                result = self.classifier(text)[0]
                nlp_score = result['score'] if result['label'].lower().startswith('positive') else 1.0 - result['score']
            except Exception as e:
                print(f"Error in NLP classification: {str(e)}")
                nlp_score = 0.0

            # Keyword analysis with context
            matches = []
            keyword_score = 0.0
            for kw in self.phishing_keywords:
                if kw in text:
                    # Check for contextual phrases to increase confidence
                    context_found = any(f"{kw} {phrase}" in text or f"{phrase} {kw}" in text
                                        for phrase in self.context_phrases)
                    confidence = 0.9 if context_found else 0.6
                    matches.append((kw, confidence))
                    keyword_score += confidence

            # Normalize keyword score
            keyword_score = min(keyword_score / len(self.phishing_keywords), 1.0) if matches else 0.0

            # Ensemble scoring
            final_confidence = (nlp_score * 0.6 + keyword_score * 0.4)
            label = "Phishing" if final_confidence > 0.65 else "Safe"
            is_phishing = final_confidence > 0.65

            return {
                'label': label,
                'confidence': round(final_confidence * 100, 2),
                'is_phishing': is_phishing,
                'nlp_score': round(nlp_score * 100, 2),
                'keyword_matches': matches
            }

        except Exception as e:
            print(f"Error in phishing NLP detection: {str(e)}")
            return {
                'label': 'Error',
                'confidence': 0.0,
                'is_phishing': False,
                'matches': [],
                'error': str(e)
            }