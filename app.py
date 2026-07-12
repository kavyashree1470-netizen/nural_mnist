import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import json

# ── PAGE CONFIGURATION ────────────────────────────────────────────────────────
st.set_page_config(page_title="MNIST CNN Pro Studio", page_icon="🧠", layout="wide")

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #f1f5f9; }
    .status-online { color: #10b981; font-weight: bold; border: 1px solid #10b981; padding: 2px 8px; border-radius: 10px; }
    .status-offline { color: #ef4444; font-weight: bold; border: 1px solid #ef4444; padding: 2px 8px; border-radius: 10px; }
    .metric-card {
        background: #131a2e; border: 1px solid #1e293b; border-radius: 12px;
        padding: 16px; text-align: center; margin-bottom: 10px;
    }
    .metric-card .value { font-size: 2rem; font-weight: 800; color: #2dd4bf; }
    .banner-warn {
        background: #451a03; border: 1px solid #f59e0b; border-radius: 10px;
        padding: 14px; color: #fef3c7; margin: 10px 0; border-left: 5px solid #f59e0b;
    }
    .lock-screen {
        text-align: center; padding: 40px; background: #131a2e; 
        border-radius: 20px; border: 2px dashed #a855f7; margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE INITIALIZATION ──
if "model" not in st.session_state:
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    st.session_state["model"] = model
    st.session_state["is_trained"] = False

if "canvas_key" not in st.session_state: st.session_state["canvas_key"] = 0

# ── GOOGLE SHEETS HELPER FUNCTIONS ──
@st.cache_resource
def get_sheets_client():
    try:
        if "gcp_service_account" in st.secrets:
            info = st.secrets["gcp_service_account"]
            if isinstance(info, str): info = json.loads(info)
            creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
            return gspread.authorize(creds)
    except: return None

@st.cache_data(ttl=60)
def fetch_sheet_data(url, name):
    client = get_sheets_client()
    if not client: return None
    try:
        sh = client.open_by_url(url) if "http" in url else client.open_by_key(url)
        return sh.worksheet(name).get_all_values()
    except: return None

# ── PREPROCESSING LOGIC ──
def preprocess_drawing(image_data):
    gray = np.max(image_data[:, :, :3], axis=2).astype(np.uint8)
    if np.max(gray) < 25: return None
    coords = np.argwhere(gray > 25)
    y_min, x_min = coords.min(axis=0); y_max, x_max = coords.max(axis=0)
    cropped = gray[y_min:y_max+1, x_min:x_max+1]
    pil_crop = Image.fromarray(cropped, 'L')
    pil_crop.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (28, 28), 0)
    w, h = pil_crop.size
    canvas.paste(pil_crop, ((28 - w) // 2, (28 - h) // 2))
    return np.array(canvas)

# ── OPERATOR GATEKEEPER ──
st.title("🧠 MNIST CNN Pro Studio")
op_name = st.text_input("Enter Operator Name to Unlock System", placeholder="Type name here...")

if not op_name:
    st.markdown('<div class="lock-screen"><h2 style="color: #a855f7;">🔒 System Locked</h2><p>Authentication Required: Please enter your name to access the Sandbox and Tools.</p></div>', unsafe_allow_html=True)
    st.stop()

# ── SYSTEM UNLOCKED: CONTENT BELOW ──
st.sidebar.title("📡 System Status")
spreadsheet_url = st.sidebar.text_input("Spreadsheet URL", value=st.secrets.get("SPREADSHEET_ID", ""))
sheet_name = st.sidebar.text_input("Sheet Name", value="Digits Data")

client = get_sheets_client()
if client: st.sidebar.markdown("Status: <span class='status-online'>● Online</span>", unsafe_allow_html=True)
else: st.sidebar.markdown("Status: <span class='status-offline'>○ Offline</span>", unsafe_allow_html=True)

tabs = st.tabs(["✏️ Sandbox", "📊 Training Studio", "📋 Data Explorer"])

with tabs[0]: # Sandbox
    col1, col2, col3 = st.columns([2, 1.5, 1.5])
    
    with col1:
        st.subheader("Canvas")
        canvas_result = st_canvas(
            stroke_width=22, stroke_color="#FFF", background_color="#000",
            height=300, width=300, drawing_mode="freedraw", key=f"c_{st.session_state.canvas_key}"
        )
        if st.button("Clear Canvas", use_container_width=True):
            st.session_state.canvas_key += 1; st.rerun()

    processed = preprocess_drawing(canvas_result.image_data) if canvas_result.image_data is not None else None

    with col2:
        st.subheader("Preprocessing")
        if processed is not None:
            act = int(np.count_nonzero(processed > 20))
            avg = float(np.mean(processed))
            st.markdown(f'<div class="metric-card"><small>Active Pixels</small><div class="value">{act}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><small>Mean Intensity</small><div class="value">{avg:.1f}</div></div>', unsafe_allow_html=True)
            st.image(processed, width=220, caption="CNN Input (28x28 Centered)")
        else: st.info("Waiting for drawing...")

    with col3:
        st.subheader("Prediction")
        st.write(f"Logged in as: **{op_name}**")
        label = st.selectbox("Assign True Label", list(range(10)))
        
        if processed is not None:
            inp = processed.reshape(1, 28, 28, 1).astype("float32") / 255.0
            preds = st.session_state["model"].predict(inp, verbose=0)[0]
            pred_digit = int(np.argmax(preds))
            conf = float(preds[pred_digit])
            
            st.markdown(f"## Prediction: `{pred_digit}`")
            st.progress(conf, text=f"Confidence: {conf*100:.1f}%")
            
            is_mismatch = (pred_digit != label)
            if is_mismatch:
                st.markdown(f'<div class="banner-warn">⚠️ Mismatch: Prediction ({pred_digit}) != Label ({label})</div>', unsafe_allow_html=True)

            if st.button("🚀 Push to Cloud", use_container_width=True):
                if client:
                    try:
                        sh = client.open_by_url(spreadsheet_url) if "http" in spreadsheet_url else client.open_by_key(spreadsheet_url)
                        wks = sh.worksheet(sheet_name)
