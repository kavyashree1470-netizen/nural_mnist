import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import plotly.express as px
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import json
import os

# ── PAGE CONFIGURATION ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MNIST CNN Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM STYLING ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #f1f5f9; }
    .main-header { color: #a855f7; font-weight: 800; margin-bottom: 5px; }
    .metric-card {
        background: #131a2e; border: 1px solid #1e293b; border-radius: 12px;
        padding: 15px; text-align: center; margin-bottom: 10px;
    }
    .metric-value { font-size: 1.8rem; font-weight: 800; color: #2dd4bf; }
    .metric-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; }
    .mnist-grid {
        display: grid; grid-template-columns: repeat(28, 8px);
        gap: 0; border: 2px solid #334155; width: fit-content; margin: 0 auto;
    }
    .mnist-cell { width: 8px; height: 8px; }
    .banner-warn {
        background: #451a03; border: 1px solid #f59e0b; border-radius: 10px;
        padding: 14px; color: #fef3c7; margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE INITIALIZATION ──────────────────────────────────────────────
def build_cnn_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3), # Prevent overfitting as requested
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if "model" not in st.session_state:
    st.session_state["model"] = build_cnn_model()
    st.session_state["is_trained"] = False
if "history" not in st.session_state: st.session_state["history"] = None
if "canvas_key" not in st.session_state: st.session_state["canvas_key"] = 0

# ── GOOGLE SHEETS UTILS ───────────────────────────────────────────────────────
def get_sheets_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    if "gcp_service_account" in st.secrets:
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
        return gspread.authorize(creds)
    return None

# ── PREPROCESSING & ANALYTICS ─────────────────────────────────────────────────
def preprocess_drawing(image_data):
    """Crops and centers the drawing into a 28x28 grayscale image."""
    gray = np.max(image_data[:, :, :3], axis=2).astype(np.uint8)
    if np.max(gray) < 20: return None
    
    # Bounding Box Extraction
    coords = np.argwhere(gray > 20)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped = gray[y_min:y_max+1, x_min:x_max+1]
    
    # Resize with aspect ratio
    pil_img = Image.fromarray(cropped, 'L')
    w, h = pil_img.size
    if w > h:
        new_w = 20
        new_h = int(20 * (h / w))
    else:
        new_h = 20
        new_w = int(20 * (w / h))
    
    pil_img = pil_img.resize((max(1, new_w), max(1, new_h)), Image.Resampling.LANCZOS)
    
    # Center in 28x28
    canvas = Image.new('L', (28, 28), 0)
    canvas.paste(pil_img, ((28 - new_w)//2, (28 - new_h)//2))
    return np.array(canvas)

def get_pixel_metrics(arr_28x28):
    active_pixels = np.count_nonzero(arr_28x28 > 30)
    mean_val = np.mean(arr_28x28)
    return active_pixels, mean_val

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_mnist_data(samples_per_digit=100):
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        
        # Balance dataset
        idx = np.concatenate([np.where(y_train == i)[0][:samples_per_digit] for i in range(10)])
        return x_train[idx], y_train[idx], x_test[:200], y_test[:200]
    except:
        # Emergency dummy data if download fails
        return np.random.rand(100, 28, 28, 1), np.random.randint(0, 10, 100), np.random.rand(20, 28, 28, 1), np.random.randint(0, 10, 20)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ CNN Settings")
epochs = st.sidebar.slider("Epochs", 5, 50, 10)
lr = st.sidebar.select_slider("Learning Rate", [0.01, 0.001, 0.0001], value=0.001)

if st.sidebar.button("🗑️ Reset Model Weights"):
    st.session_state["model"] = build_cnn_model()
    st.session_state["is_trained"] = False
    st.rerun()

# ── MAIN UI ───────────────────────────────────────────────────────────────────
st.markdown("<h1 class='main-header'>🧠 MNIST CNN Studio</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["✏️ Sandbox & Inference", "📊 Training Studio"])

with tab1:
    col1, col2, col3 = st.columns([2, 1.5, 1.5])
    
    with col1:
        st.subheader("Draw Digit")
        canvas_res = st_canvas(
            stroke_width=18, stroke_color="#FFF", background_color="#000",
            height=280, width=280, drawing_mode="freedraw", key=f"canv_{st.session_state.canvas_key}"
        )
        if st.button("Clear Canvas"):
            st.session_state.canvas_key += 1
            st.rerun()

    processed_img = None
    if canvas_res.image_data is not None:
        processed_img = preprocess_drawing(canvas_res.image_data)

    with col2:
        st.subheader("CNN Input View")
        if processed_img is not None:
            # Display metrics (Req 1)
            active, mean = get_pixel_metrics(processed_img)
            c_a, c_b = st.columns(2)
            c_a.markdown(f"<div class='metric-card'><div class='metric-label'>Active Pixels</div><div class='metric-value'>{active}</div></div>", unsafe_allow_html=True)
            c_b.markdown(f"<div class='metric-card'><div class='metric-label'>Mean Pixel</div><div class='metric-value'>{mean:.1f}</div></div>", unsafe_allow_html=True)
            
            # HTML Preview
            cells = "".join([f'<div class="mnist-cell" style="background:rgb({v},{v//2},{v})"></div>' for v in processed_img.flatten()])
            st.markdown(f'<div class="mnist-grid">{cells}</div>', unsafe_allow_html=True)
        else:
            st.info("Waiting for drawing...")

    with col3:
        st.subheader("Prediction")
        assigned_label = st.selectbox("Your Label", list(range(10)))
        
        if processed_img is not None:
            # Inference
            inp = processed_img.reshape(1, 28, 28, 1) / 255.0
            pred_probs = st.session_state["model"].predict(inp, verbose=0)
            pred_digit = np.argmax(pred_probs)
            conf = np.max(pred_probs)
            
            st.markdown(f"## Prediction: <span style='color:#a855f7'>{pred_digit}</span>", unsafe_allow_html=True)
            st.progress(float(conf))
            
            # Warning logic (Req 5)
            if pred_digit != assigned_label:
                st.markdown(f"""
                <div class='banner-warn'>
                    ⚠️ <b>Label Mismatch:</b> You marked this as <b>{assigned_label}</b>, 
                    but the CNN predicts <b>{pred_digit}</b> ({conf*100:.1f}% confidence).
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("✅ Prediction matches your label!")

with tab2:
    st.subheader("Model Training")
    if st.button("🚀 Start CNN Training"):
        with st.spinner("Loading MNIST..."):
            xt, yt, xv, yv = load_mnist_data()
            
            # Re-compile with chosen LR
            st.session_state["model"].optimizer.learning_rate.assign(lr)
            
            history = st.session_state["model"].fit(
                xt, yt, validation_data=(xv, yv),
                epochs=epochs, batch_size=32, verbose=0
            )
            st.session_state["history"] = history.history
            st.session_state["is_trained"] = True
            st.success("Training Complete!")
            
    if st.session_state["history"]:
        h = st.session_state["history"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=h['accuracy'], name='Train Accuracy'))
        fig.add_trace(go.Scatter(y=h['val_accuracy'], name='Val Accuracy'))
        fig.update_layout(template="plotly_dark", title="Accuracy Curve")
        st.plotly_chart(fig, use_container_width=True)
