import sys
import os
import streamlit as st
import shutil
import requests
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
from PIL import Image
import time
import uuid
import subprocess
import torch

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

# Imports from modules
from deepfake_detector_core.inference import PhishingClassifier
from ar_phishing_detector.ocr_analysis import OCRAnalyzer
from ar_phishing_detector.yolo_ui_detector import YOLODetector
from voice_phishing_detector.transcriber import AudioTranscriber
from voice_phishing_detector.phishing_nlp import PhishingNLPDetector


# Load Lottie JSON from URL
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Enhanced Plotly gauge chart
def draw_gauge(title, value, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24, 'color': '#00ff88'}},
        delta={'reference': 50, 'increasing': {'color': "#00ff88"}, 'decreasing': {'color': "#ff3366"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "white"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0.7)",
            'borderwidth': 2,
            'bordercolor': "#00ff88",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 51, 102, 0.3)'},
                {'range': [50, 80], 'color': 'rgba(255, 204, 102, 0.3)'},
                {'range': [80, 100], 'color': 'rgba(0, 255, 136, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#ff3366", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Roboto"}
    )
    return fig


# Check FFmpeg availability
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


# Streamlit configuration
st.set_page_config(
    page_title="AURA-GUARD",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛡️"
)

# CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    .main {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 100%);
        color: #e0e1dd;
        font-family: 'Roboto', sans-serif;
        padding: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(27, 38, 59, 0.8);
        border-radius: 12px;
        padding: 10px;
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab"] {
        color: #e0e1dd;
        font-weight: 700;
        border-radius: 8px;
        margin: 5px;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(45deg, #4158d0, #c850c0);
        transform: translateY(-2px);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #4158d0, #c850c0);
        color: #00ff88;
        box-shadow: 0 4px 15px rgba(65, 88, 208, 0.4);
    }
    .card {
        background: rgba(27, 38, 59, 0.8);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid rgba(0, 255, 136, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
    }
    .phishing { border-left: 5px solid #ff3366; }
    .safe { border-left: 5px solid #00ff88; }
    .error { border-left: 5px solid #ffcc00; }
    .stButton > button {
        background: linear-gradient(45deg, #4158d0, #c850c0);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(65, 88, 208, 0.4);
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #c850c0, #4158d0);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(65, 88, 208, 0.6);
    }
    h1, h2, h3 {
        color: #00ff88;
        text-shadow: 0 2px 4px rgba(0, 255, 136, 0.3);
    }
    .stFileUploader > div > div {
        background: rgba(27, 38, 59, 0.8);
        border-radius: 12px;
        border: 2px dashed rgba(0, 255, 136, 0.3);
    }
    .stProgress > div > div {
        background: linear-gradient(45deg, #4158d0, #c850c0);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150?text=AURA-GUARD", caption="AURA-GUARD Logo")
    st.markdown("<h2 style='text-align: center;'>🛡️ AURA-GUARD</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #e0e1dd;'>AI-powered cybersecurity platform</p>",
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Features**")
    st.markdown("""
    - 🎭 Deepfake Detection
    - 🧠 AR Phishing Analysis
    - 🎤 Voice Phishing Detection
    """)
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #7780a1;'>© 2025 AURA-GUARD</p>", unsafe_allow_html=True)

# Main title
st.markdown("""
<div class='card'>
    <h1 style='text-align: center;'>🛡️ AURA-GUARD</h1>
    <p style='text-align: center; color: #e0e1dd;'>Next-generation AI cybersecurity dashboard for deepfake, AR, and voice phishing detection</p>
</div>
""", unsafe_allow_html=True)


# Initialize models
@st.cache_resource
def initialize_models():
    return {
        'deepfake': PhishingClassifier(),
        'ocr': OCRAnalyzer(),
        'yolo': YOLODetector(),
        'transcriber': AudioTranscriber(),
        'nlp': PhishingNLPDetector()
    }


models = initialize_models()
lottie_spinner = load_lottie_url("https://assets6.lottiefiles.com/packages/lf20_kkflmtur.json")

# Check GPU availability
if not torch.cuda.is_available():
    st.markdown("""
    <div class='card error'>
        <h3>Warning</h3>
        <p>Running on CPU. For better performance, consider using a GPU with CUDA support.</p>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tabs = st.tabs(["🎭 Deepfake Detection", "🧠 AR Phishing Detection", "🎤 Voice Phishing Detection"])

# ---------------- Deepfake Tab ----------------
with tabs[0]:
    st.markdown("<div class='card'><h2>Deepfake Image Detection</h2></div>", unsafe_allow_html=True)
    uploaded_img = st.file_uploader("Upload Image for Analysis", type=["jpg", "png", "jpeg"],
                                    key=f"deepfake_{uuid.uuid4()}")

    if uploaded_img:
        with st.spinner("🔍 Analyzing image..."):
            st_lottie(lottie_spinner, width=200, height=200)
            progress = st.progress(0)
            temp_dir = "temp_files"
            os.makedirs(temp_dir, exist_ok=True)
            img_path = os.path.join(temp_dir, "temp_image.jpg")

            try:
                with open(img_path, "wb") as f:
                    f.write(uploaded_img.read())
                progress.progress(33)

                result = models['deepfake'].classify_image(img_path)
                progress.progress(66)

                # Convert image to base64
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    import base64
                    from io import BytesIO

                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()

                st.markdown(
                    f"<div class='card'><img src='data:image/jpeg;base64,{img_str}' style='width:100%; max-width:400px; border-radius:8px;'/></div>",
                    unsafe_allow_html=True)

                result_class = "phishing" if result['is_phishing'] else "safe"
                st.markdown(f"""
                <div class='card {result_class}'>
                    <h3>Prediction: {result['label']}</h3>
                    <p><strong>Confidence:</strong> {result['confidence']:.1f}%</p>
                    <p><strong>Deep Learning Score:</strong> {result['deep_learning_score']:.1f}%</p>
                    <p><strong>UI Anomalies:</strong></p>
                    <ul style='color: #e0e1dd;'>
                        <li>Keywords: {', '.join(result['ui_anomalies']['suspicious_keywords'])}</li>
                        <li>URLs: {', '.join(url['url'] for url in result['ui_anomalies']['suspicious_urls'])}</li>
                        <li>UI Elements: {', '.join(f"{item['label']} ({item['confidence'] * 100:.1f}%)" for item in result['ui_anomalies']['ui_elements'])}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(draw_gauge("Confidence", result['confidence'], "#00ff88"), use_container_width=True)
                with col2:
                    st.plotly_chart(draw_gauge("DL Score", result['deep_learning_score'], "#ff3366"),
                                    use_container_width=True)

                progress.progress(100)

            except Exception as e:
                st.markdown(f"""
                <div class='card error'>
                    <h3>Error</h3>
                    <p>{str(e)}</p>
                </div>
                """, unsafe_allow_html=True)

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

# ---------------- AR Tab ----------------
with tabs[1]:
    st.markdown("<div class='card'><h2>AR Phishing Detection</h2></div>", unsafe_allow_html=True)
    uploaded_ar = st.file_uploader("Upload Screenshot or Video", type=["jpg", "png", "jpeg", "mp4"],
                                   key=f"ar_{uuid.uuid4()}")

    if uploaded_ar:
        with st.spinner("🔍 Analyzing AR content..."):
            st_lottie(lottie_spinner, width=200, height=200)
            progress = st.progress(0)
            temp_dir = "temp_files"
            os.makedirs(temp_dir, exist_ok=True)
            file_ext = uploaded_ar.name.split('.')[-1].lower()
            file_path = os.path.join(temp_dir, f"temp_ar.{file_ext}")

            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_ar.read())
                progress.progress(33)

                if file_ext in ['jpg', 'jpeg', 'png']:
                    result = models['deepfake'].classify_image(file_path)
                    with Image.open(file_path) as img:
                        img = img.convert('RGB')
                        import base64
                        from io import BytesIO

                        buffered = BytesIO()
                        img.save(buffered, format=file_ext.upper())
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                    st.markdown(
                        f"<div class='card'><img src='data:image/{file_ext};base64,{img_str}' style='width:100%; max-width:400px; border-radius:8px;'/></div>",
                        unsafe_allow_html=True)
                else:
                    result = models['deepfake'].classify_video(file_path)
                    st.video(file_path)

                result_class = "phishing" if result['is_phishing'] else "safe"
                st.markdown(f"""
                <div class='card {result_class}'>
                    <h3>Prediction: {result['label']}</h3>
                    <p><strong>Confidence:</strong> {result['confidence']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

                if file_ext in ['mp4']:
                    st.plotly_chart(
                        draw_gauge("Phishing Frame Ratio", result['phishing_frames_ratio'] * 100, "#ff3366"),
                        use_container_width=True)

                progress.progress(100)

            except Exception as e:
                st.markdown(f"""
                <div class='card error'>
                    <h3>Error</h3>
                    <p>{str(e)}</p>
                </div>
                """, unsafe_allow_html=True)

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

# ---------------- Voice Tab ----------------
with tabs[2]:
    st.markdown("<div class='card'><h2>Voice Phishing Detection</h2></div>", unsafe_allow_html=True)
    uploaded_audio = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"], key=f"audio_{uuid.uuid4()}")

    if uploaded_audio:
        if not check_ffmpeg():
            st.markdown("""
            <div class='card error'>
                <h3>Error</h3>
                <p>FFmpeg is not installed. Please install FFmpeg to process audio files.</p>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        with st.spinner("🔊 Transcribing audio and detecting phishing..."):
            st_lottie(lottie_spinner, width=200, height=200)
            progress = st.progress(0)
            temp_dir = "temp_files"
            os.makedirs(temp_dir, exist_ok=True)
            raw_audio_path = os.path.join(temp_dir, "uploaded_audio." + uploaded_audio.name.split('.')[-1])
            converted_audio_path = os.path.join(temp_dir, "converted_audio.wav")

            try:
                with open(raw_audio_path, "wb") as f:
                    f.write(uploaded_audio.read())
                    f.flush()
                progress.progress(25)

                ffmpeg_command = [
                    "ffmpeg", "-y", "-i", raw_audio_path,
                    "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", converted_audio_path
                ]
                result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                if result.returncode != 0:
                    raise Exception(f"FFmpeg conversion failed: {result.stderr.decode()}")

                progress.progress(50)

                audio_result = models['transcriber'].transcribe_audio(converted_audio_path)
                if audio_result['error']:
                    raise Exception(audio_result['error'])

                progress.progress(75)

                nlp_result = models['nlp'].detect_phishing_nlp(audio_result['text'])
                result_class = "phishing" if nlp_result['is_phishing'] else "safe"

                st.audio(converted_audio_path)
                st.markdown("<div class='card'><h3>📜 Transcript</h3></div>", unsafe_allow_html=True)
                st.code(audio_result['text'], language='text')

                st.markdown(f"""
                <div class='card {result_class}'>
                    <h3>Prediction: {nlp_result['label']}</h3>
                    <p><strong>Confidence:</strong> {nlp_result['confidence']:.1f}%</p>
                    <p><strong>NLP Score:</strong> {nlp_result['nlp_score']:.1f}%</p>
                    <p><strong>Suspicious Phrases:</strong></p>
                    <ul style='color: #e0e1dd;'>{''.join(f"<li>{kw[0]} ({kw[1] * 100:.1f}%)</li>" for kw in nlp_result['keyword_matches'])}</ul>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(draw_gauge("Confidence", nlp_result['confidence'], "#00ff88"),
                                    use_container_width=True)
                with col2:
                    st.plotly_chart(draw_gauge("NLP Score", nlp_result['nlp_score'], "#ff3366"),
                                    use_container_width=True)

                progress.progress(100)

            except Exception as e:
                st.markdown(f"""
                <div class='card error'>
                    <h3>Error</h3>
                    <p>{str(e)}</p>
                </div>
                """, unsafe_allow_html=True)

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

# Footer
st.markdown("""
<div class='card' style='text-align: center;'>
    <p style='color: #7780a1;'>© 2025 AURA-GUARD – Powered by AI for a Safer Digital Future</p>
</div>
""", unsafe_allow_html=True)