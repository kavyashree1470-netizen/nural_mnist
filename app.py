import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import plotly.express as px
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import json
import os

# ── PAGE CONFIGURATION ────────────────────────────────────────────────────────
st.set_page_config(page_title="MNIST CNN Studio & Collector", page_icon="🧠", layout="wide")

# ── STYLING ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #f1f5f9; }
    .main-header { color: #a855f7; font-weight: 800; }
    .metric-card {
        background: #131a2e; border: 1px solid #1e293b; border-radius: 12px;
        padding: 16px; text-align: center;
    }
    .metric-card .value { font-size: 2rem; font-weight: 800; color: #2dd4bf; }
    .mnist-grid {
        display: grid; grid-template-columns: repeat(28, 8px);
        gap: 0; border: 3px solid #1e293b; border-radius: 8px;
        width: fit-content; margin: 0 auto; background-color: #000;
    }
    .mnist-cell { width: 8px; height: 8px; }
    .banner-warn {
        background: #451a03; border: 1px solid #f59e0b; border-radius: 10px;
        padding: 14px; color: #fef3c7; margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE INITIALIZATION ──────────────────────────────────────────────
MODEL_VERSION = "CNN_v1.1"

def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3), # Req 3: Dropout
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if "model" not in st.session_state or st.session_state.get("m_ver") != MODEL_VERSION:
    st.session_state["model"] = build_cnn()
    st.session_state["m_ver"] = MODEL_VERSION
    st.session_state["is_trained"] = False

if "canvas_key" not in st.session_state: st.session_state["canvas_key"] = 0
if "uploaded_creds" not in st.session_state: st.session_state["uploaded_creds"] = None

# ── GOOGLE SHEETS HELPER ──────────────────────────────────────────────────────
def get_sheets_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    if st.session_state["uploaded_creds"]:
        creds = Credentials.from_service_account_info(st.session_state["uploaded_creds"], scopes=scopes)
        return gspread.authorize(creds)
    if "gcp_service_account" in st.secrets:
        info = st.secrets["gcp_service_account"]
        if isinstance(info, str): info = json.loads(info)
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        return gspread.authorize(creds)
    return None

# ── PREPROCESSING & ANALYTICS ─────────────────────────────────────────────────
def preprocess_drawing(image_data):
    """Req 2: Centering."""
    gray = np.max(image_data[:, :, :3], axis=2).astype(np.uint8)
    if np.max(gray) < 20: return None
    coords = np.argwhere(gray > 20)
    y_min, x_min = coords.min(axis=0); y_max, x_max = coords.max(axis=0)
    cropped = gray[y_min:y_max+1, x_min:x_max+1]
    pil_img = Image.fromarray(cropped, 'L').resize((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (28, 28), 0)
    canvas.paste(pil_img, (4, 4))
    return np.array(canvas)

# ── MAIN UI ───────────────────────────────────────────────────────────────────
st.markdown("<h1 class='main-header'>🧠 MNIST CNN Studio</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")
uploaded_file = st.sidebar.file_uploader("Service Account JSON", type=["json"])
if uploaded_file: st.session_state["uploaded_creds"] = json.load(uploaded_file)
spreadsheet_url = st.sidebar.text_input("Spreadsheet URL/ID", value=st.secrets.get("SPREADSHEET_ID", ""))
sheet_name = st.sidebar.text_input("Sheet Title", value="Digits Data")

tab1, tab2 = st.tabs(["✏️ Drawing Sandbox", "📊 Model Training"])

with tab1:
    col1, col2, col3 = st.columns([2, 1.5, 1.5])
    
    with col1:
        st.subheader("Canvas")
        canvas_result = st_canvas(
            stroke_width=20, stroke_color="#FFF", background_color="#000",
            height=280, width=280, drawing_mode="freedraw", key=f"c_{st.session_state.canvas_key}"
        )
        if st.button("Clear Canvas"):
            st.session_state.canvas_key += 1; st.rerun()

    processed = None
    if canvas_result.image_data is not None:
        processed = preprocess_drawing(canvas_result.image_data)

    with col2:
        st.subheader("Analytics")
        if processed is not None:
            # Req 1: Active/Mean Pixel
            act = int(np.count_nonzero(processed > 30))
            avg = float(np.mean(processed))
            st.markdown(f"<div class='metric-card'><div class='value'>{act}</div><div class='label'>Active Pixels</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><div class='value'>{avg:.1f}</div><div class='label'>Mean Pixel</div></div>", unsafe_allow_html=True)
            
            cells = "".join([f'<div class="mnist-cell" style="background:rgb({v},{v//2},{v})"></div>' for v in processed.flatten()])
            st.markdown(f'<div class="mnist-grid">{cells}</div>', unsafe_allow_html=True)

    with col3:
        st.subheader("Prediction")
        operator = st.text_input("Operator", value="User")
        assigned_label = st.selectbox("Assign Label", list(range(10)))
        
        if processed is not None:
            # CNN Inference (Req 4: Accuracy)
            inp = processed.reshape(1, 28, 28, 1).astype("float32") / 255.0
            preds = st.session_state["model"].predict(inp, verbose=0)[0]
            pred_digit = int(np.argmax(preds)) # FIXED: Convert to Python int
            conf = float(preds[pred_digit])   # FIXED: Convert to Python float
            
            st.markdown(f"## Prediction: `{pred_digit}`")
            st.progress(conf)
            
            # Req 5: Correct Warning Message
            if pred_digit != assigned_label:
                st.markdown(f"<div class='banner-warn'>⚠️ Warning: Assigned label ({assigned_label}) does not match Prediction ({pred_digit}).</div>", unsafe_allow_html=True)

            if st.button("🚀 Push to Cloud"):
                client = get_sheets_client()
                if client and spreadsheet_url:
                    try:
                        sh = client.open_by_url(spreadsheet_url) if "http" in spreadsheet_url else client.open_by_key(spreadsheet_url)
                        wks = sh.worksheet(sheet_name)
                        
                        # --- THE FIX FOR YOUR ERROR ---
                        # We force all items to standard Python types (int, str)
                        row_metadata = [
                            int(len(wks.get_all_values())), # ID
                            str(operator),                 # Username
                            int(assigned_label),           # Label
                            datetime.now().isoformat(),     # Timestamp
                            int(pred_digit)                # Pred Label
                        ]
                        pixel_data = [int(p) for p in processed.flatten()]
                        full_row = row_metadata + pixel_data
                        
                        wks.append_row(full_row)
                        st.success("Data pushed successfully!")
                        st.session_state.canvas_key += 1; st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.error("Google Sheets not configured.")

with tab2:
    st.subheader("Train the CNN")
    if st.button("🔥 Start Training"):
        with st.spinner("Training on MNIST..."):
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train = x_train[:2000].reshape(-1, 28, 28, 1) / 255.0
            y_train = y_train[:2000]
            st.session_state["model"].fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
            st.session_state["is_trained"] = True
            st.success("Model trained successfully!")
