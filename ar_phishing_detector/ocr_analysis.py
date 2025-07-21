# ar_phishing_detector/ocr_analysis.py
import easyocr
import re
from urllib.parse import urlparse


class OCRAnalyzer:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)  # GPU off for broader compatibility
        self.suspicious_keywords = [
            'login', 'verify', 'update', 'account', 'password', 'bank',
            'urgent', 'security', 'authentication', 'credentials',
            'payment', 'confirm', 'access', 'secure'
        ]
        self.suspicious_domains = [
            'bit.ly', 'tinyurl', 'phish', 'fake', 'login', 'secure',
            'verify', 'account', 'bank', 'update'
        ]

    def extract_text(self, image_path):
        """Extract text from image with enhanced OCR settings."""
        try:
            results = self.reader.readtext(image_path, detail=1, paragraph=True)
            text_data = " ".join([res[1] for res in results])
            return text_data, results
        except Exception as e:
            print(f"Error in OCR extraction: {str(e)}")
            return "", []

    def detect_suspicious_keywords(self, text):
        """Detect suspicious keywords with context analysis."""
        if not text:
            return []

        text_lower = text.lower()
        matches = []

        for keyword in self.suspicious_keywords:
            # Look for keyword in context
            if keyword in text_lower:
                # Check for phrases that increase suspicion
                context_phrases = [
                    f"{keyword} now", f"immediate {keyword}", f"{keyword} required",
                    f"urgent {keyword}", f"secure {keyword}"
                ]
                if any(phrase in text_lower for phrase in context_phrases):
                    matches.append((keyword, 0.9))  # Higher confidence for contextual matches
                else:
                    matches.append((keyword, 0.6))

        return matches

    def detect_suspicious_urls(self, text):
        """Detect suspicious URLs with domain analysis."""
        if not text:
            return []

        urls = re.findall(r'https?://[\S]+', text)
        phishing_urls = []

        for url in urls:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                # Check for suspicious domains and URL patterns
                if any(susp_domain in domain for susp_domain in self.suspicious_domains) or \
                        len(domain) > 50 or \
                        sum(c.isdigit() for c in domain) > 5:  # Suspiciously long or number-heavy domains
                    phishing_urls.append({
                        'url': url,
                        'domain': domain,
                        'confidence': 0.8 if any(
                            susp_domain in domain for susp_domain in self.suspicious_domains) else 0.6
                    })
            except:
                continue

        return phishing_urls