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
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
MODEL_VERSION = "CNN_PRO_V1"

def build_pro_cnn():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if "model" not in st.session_state or st.session_state.get("m_ver") != MODEL_VERSION:
    st.session_state["model"] = build_pro_cnn()
    st.session_state["m_ver"] = MODEL_VERSION
    st.session_state["is_trained"] = False

if "canvas_key" not in st.session_state: st.session_state["canvas_key"] = 0

# ── GOOGLE SHEETS CORE ───────────────────────────────────────────────────────
def get_sheets_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    try:
        if "gcp_service_account" in st.secrets:
            info = st.secrets["gcp_service_account"]
            if isinstance(info, str): info = json.loads(info)
            creds = Credentials.from_service_account_info(info, scopes=scopes)
            return gspread.authorize(creds)
    except: return None
    return None

def check_sheet_connection(spreadsheet_url, sheet_name):
    client = get_sheets_client()
    if not client: return False, "No Credentials"
    try:
        sh = client.open_by_url(spreadsheet_url) if "http" in spreadsheet_url else client.open_by_key(spreadsheet_url)
        sh.worksheet(sheet_name)
        return True, "Connected"
    except Exception as e: return False, str(e)

# ── PREPROCESSING (IMPROVED CENTERING) ────────────────────────────────────────
def preprocess_drawing(image_data):
    # Convert to grayscale
    gray = np.max(image_data[:, :, :3], axis=2).astype(np.uint8)
    if np.max(gray) < 25: return None # Ignore empty canvas
    
    # 1. Find Bounding Box
    coords = np.argwhere(gray > 25)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped = gray[y_min:y_max+1, x_min:x_max+1]
    
    # 2. Convert to PIL and resize proportionally to fit in 20x20
    pil_crop = Image.fromarray(cropped, 'L')
    pil_crop.thumbnail((20, 20), Image.Resampling.LANCZOS)
    
    # 3. Paste into center of 28x28 canvas
    canvas = Image.new('L', (28, 28), 0)
    w, h = pil_crop.size
    offset = ((28 - w) // 2, (28 - h) // 2)
    canvas.paste(pil_crop, offset)
    
    return np.array(canvas)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title("📡 System Status")
spreadsheet_url = st.sidebar.text_input("Spreadsheet URL", value=st.secrets.get("SPREADSHEET_ID", ""))
sheet_name = st.sidebar.text_input("Sheet Name", value="Digits Data")

conn_ok, conn_msg = check_sheet_connection(spreadsheet_url, sheet_name)
if conn_ok:
    st.sidebar.markdown(f"Status: <span class='status-online'>● {conn_msg}</span>", unsafe_allow_html=True)
else:
    st.sidebar.markdown(f"Status: <span class='status-offline'>○ {conn_msg}</span>", unsafe_allow_html=True)

if st.sidebar.button("🗑️ Reset Model Weights"):
    st.session_state["model"] = build_pro_cnn()
    st.session_state["is_trained"] = False
    st.rerun()

# ── MAIN DASHBOARD ────────────────────────────────────────────────────────────
st.title("🧠 MNIST CNN Pro Studio")
tab1, tab2, tab3 = st.tabs(["✏️ Drawing Sandbox", "📋 Live Database", "📊 Training Studio"])

with tab1:
    col1, col2, col3 = st.columns([2, 1.5, 1.5])
    
    with col1:
        st.subheader("Canvas")
        canvas_result = st_canvas(
            stroke_width=22, stroke_color="#FFF", background_color="#000",
            height=300, width=300, drawing_mode="freedraw", key=f"c_{st.session_state.canvas_key}"
        )
        if st.button("Clear Canvas", use_container_width=True):
            st.session_state.canvas_key += 1; st.rerun()

    processed = None
    if canvas_result.image_data is not None:
        processed = preprocess_drawing(canvas_result.image_data)

    with col2:
        st.subheader("Analytics")
        if processed is not None:
            act = int(np.count_nonzero(processed > 20))
            avg = float(np.mean(processed))
            st.markdown(f"<div class='metric-card'><small>Active Pixels</small><div class='value'>{act}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><small>Mean Intensity</small><div class='value'>{avg:.1f}</div></div>", unsafe_allow_html=True)
            
            st.image(processed, width=200, caption="Preprocessed Input (Centered)")
        else: st.info("Draw a digit to see preprocessing.")

    with col3:
        st.subheader("Prediction")
        operator = st.text_input("Operator", value="User_1")
        assigned_label = st.selectbox("True Label", list(range(10)))
        
        if processed is not None:
            # Inference
            inp = processed.reshape(1, 28, 28, 1).astype("float32") / 255.0
            preds = st.session_state["model"].predict(inp, verbose=0)[0]
            pred_digit = int(np.argmax(preds))
            conf = float(preds[pred_digit])
            
            color = "#2dd4bf" if pred_digit == assigned_label else "#ef4444"
            st.markdown(f"## Prediction: <span style='color:{color}'>{pred_digit}</span>", unsafe_allow_html=True)
            st.progress(conf, text=f"Confidence: {conf*100:.1f}%")
            
            if pred_digit != assigned_label:
                st.markdown(f"""<div class='banner-warn'>⚠️ <b>Mismatch:</b> Assigned label ({assigned_label}) does not match Prediction ({pred_digit}).</div>""", unsafe_allow_html=True)

            if st.button("🚀 Save to Google Sheet", use_container_width=True):
                client = get_sheets_client()
                if client and conn_ok:
                    try:
                        sh = client.open_by_url(spreadsheet_url) if "http" in spreadsheet_url else client.open_by_key(spreadsheet_url)
                        wks = sh.worksheet(sheet_name)
                        row = [int(len(wks.get_all_values())), operator, int(assigned_label), datetime.now().strftime("%Y-%m-%d %H:%M"), int(pred_digit)] + [int(p) for p in processed.flatten()]
                        wks.append_row(row)
                        st.toast("Success! Data Synced.", icon="✅")
                        st.session_state.canvas_key += 1; st.rerun()
                    except Exception as e: st.error(f"Sync Failed: {e}")
                else: st.error("Check Google Sheet connection in sidebar.")

with tab2:
    st.subheader("Live Google Sheet Data")
    if conn_ok:
        try:
            client = get_sheets_client()
            sh = client.open_by_url(spreadsheet_url) if "http" in spreadsheet_url else client.open_by_key(spreadsheet_url)
            wks = sh.worksheet(sheet_name)
            raw_data = wks.get_all_values()
            if len(raw_data) > 1:
                df = pd.DataFrame(raw_data[1:], columns=raw_data[0])
                # Show only first 5 columns (Metadata)
                st.dataframe(df.iloc[:, :5].sort_index(ascending=False), use_container_width=True)
                st.success(f"Total entries in sheet: {len(df)}")
            else: st.info("Sheet is empty.")
        except: st.error("Could not fetch data.")
    else: st.warning("Connect Google Sheet to see live data.")

with tab3:
    st.subheader("Model Training")
    st.write("To improve accuracy, train the model on the full MNIST dataset.")
    if st.button("🔥 Start High-Accuracy Training"):
        with st.spinner("Training CNN... (Using 5,000 samples)"):
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train = x_train[:5000].reshape(-1, 28, 28, 1) / 255.0
            y_train = y_train[:5000]
            st.session_state["model"].fit(x_train, y_train, epochs=8, batch_size=64, verbose=0)
            st.session_state["is_trained"] = True
            st.success("Training Complete! Try drawing again.")
