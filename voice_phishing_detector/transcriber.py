# voice_phishing_detector/transcriber.py
import os
import whisper
import subprocess
import tempfile

class AudioTranscriber:
    def __init__(self, model_name="base"):
        # Ensure ffmpeg is available
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("FFmpeg not found. Please install FFmpeg and ensure it's in PATH.")

        # Load Whisper model
        try:
            self.model = whisper.load_model(model_name)
        except Exception as e:
            print(f"Error loading Whisper model {model_name}: {str(e)}")
            # Fallback to smallest model
            self.model = whisper.load_model("tiny")

    def preprocess_audio(self, audio_path):
        """Preprocess audio by converting to WAV format if needed."""
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Check if audio is already in WAV format
            if not audio_path.lower().endswith('.wav'):
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    subprocess.run([
                        "ffmpeg", "-i", audio_path, "-ar", "16000", "-ac", "1",
                        "-c:a", "pcm_s16le", temp_path
                    ], check=True, capture_output=True)
                return temp_path
            return audio_path

        except Exception as e:
            print(f"Error preprocessing audio {audio_path}: {str(e)}")
            return None

    def transcribe_audio(self, audio_path):
        """Transcribe audio with error handling and cleanup."""
        temp_file = None
        try:
            processed_path = self.preprocess_audio(audio_path)
            if processed_path is None:
                return {
                    'text': '',
                    'error': 'Audio preprocessing failed'
                }

            # Store temp file path for cleanup
            if processed_path != audio_path:
                temp_file = processed_path

            result = self.model.transcribe(processed_path, fp16=False)
            return {
                'text': result["text"].strip(),
                'error': None
            }

        except Exception as e:
            print(f"Error transcribing audio {audio_path}: {str(e)}")
            return {
                'text': '',
                'error': str(e)
            }
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"Error cleaning up temporary file {temp_file}: {str(e)}")