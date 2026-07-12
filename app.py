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
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──
MODEL_VERSION = "CNN_PRO_V1.2"
if "model" not in st.session_state or st.session_state.get("m_ver") != MODEL_VERSION:
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    st.session_state["model"] = model
    st.session_state["m_ver"] = MODEL_VERSION
if "canvas_key" not in st.session_state: st.session_state["canvas_key"] = 0

# ── CACHED GOOGLE SHEETS FUNCTIONS (Prevents 429 Quota Error) ──
@st.cache_resource
def get_sheets_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    try:
        if "gcp_service_account" in st.secrets:
            info = st.secrets["gcp_service_account"]
            if isinstance(info, str): info = json.loads(info)
            creds = Credentials.from_service_account_info(info, scopes=scopes)
            return gspread.authorize(creds)
    except: return None

@st.cache_data(ttl=300) # Cache connection status for 5 minutes
def check_sheet_connection(url, name):
    client = get_sheets_client()
    if not client: return False, "No Credentials Found"
    try:
        sh = client.open_by_url(url) if "http" in url else client.open_by_key(url)
        sh.worksheet(name)
        return True, "Connected"
    except Exception as e: return False, str(e)

@st.cache_data(ttl=60) # Cache data for 1 minute
def fetch_sheet_data(url, name):
    client = get_sheets_client()
    if not client: return None
    try:
        sh = client.open_by_url(url) if "http" in url else client.open_by_key(url)
        wks = sh.worksheet(name)
        return wks.get_all_values()
    except: return None

# ── PREPROCESSING (Centered) ──
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

# ── SIDEBAR ──
st.sidebar.title("📡 System Status")
spreadsheet_url = st.sidebar.text_input("Spreadsheet URL", value=st.secrets.get("SPREADSHEET_ID", ""))
sheet_name = st.sidebar.text_input("Sheet Name", value="Digits Data")

conn_ok, conn_msg = check_sheet_connection(spreadsheet_url, sheet_name)
if conn_ok:
    st.sidebar.markdown(f"Status: <span class='status-online'>● Online</span>", unsafe_allow_html=True)
else:
    st.sidebar.markdown(f"Status: <span class='status-offline'>○ {conn_msg[:20]}...</span>", unsafe_allow_html=True)
    st.sidebar.warning("Note: Check secrets or API limits.")

# ── MAIN DASHBOARD ──
tab1, tab2, tab3 = st.tabs(["✏️ Sandbox", "📋 Live Data", "📊 Training"])

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

    processed = preprocess_drawing(canvas_result.image_data) if canvas_result.image_data is not None else None

    with col2:
        st.subheader("Analytics")
        if processed is not None:
            act = int(np.count_nonzero(processed > 20))
            avg = float(np.mean(processed))
            st.markdown(f"<div class='metric-card'><small>Active Pixels</small><div class='value'>{act}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><small>Mean Intensity</small><div class='value'>{avg:.1f}</div></div>", unsafe_allow_html=True)
            st.image(processed, width=200, caption="Centered CNN Input")
        else: st.info("Waiting for drawing...")

    with col3:
        st.subheader("Prediction")
        op = st.text_input("Operator", value="User")
        label = st.selectbox("True Label", list(range(10)))
        
        if processed is not None:
            inp = processed.reshape(1, 28, 28, 1).astype("float32") / 255.0
            preds = st.session_state["model"].predict(inp, verbose=0)[0]
            pred_digit = int(np.argmax(preds))
            conf = float(preds[pred_digit])
            
            st.markdown(f"## Prediction: `{pred_digit}`")
            st.progress(conf)
            
            if pred_digit != label:
                st.markdown(f"<div class='banner-warn'>⚠️ Label Mismatch: Assigned {label} vs Predicted {pred_digit}</div>", unsafe_allow_html=True)

            if st.button("🚀 Save to Cloud", use_container_width=True):
                client = get_sheets_client()
                if client and conn_ok:
                    try:
                        sh = client.open_by_url(spreadsheet_url) if "http" in spreadsheet_url else client.open_by_key(spreadsheet_url)
                        wks = sh.worksheet(sheet_name)
                        # We clear cache so the table refreshes next time
                        fetch_sheet_data.clear() 
                        row = [int(len(wks.get_all_values())), op, int(label), datetime.now().strftime("%H:%M:%S"), int(pred_digit)] + [int(p) for p in processed.flatten()]
                        wks.append_row(row)
                        st.toast("Saved Successfully!", icon="✅")
                        st.session_state.canvas_key += 1; st.rerun()
                    except Exception as e: st.error(f"Error: {e}")

with tab2:
    st.subheader("Google Sheet Preview")
    if st.button("🔄 Refresh Data (Clear Cache)"):
        fetch_sheet_data.clear()
        st.rerun()

    raw_rows = fetch_sheet_data(spreadsheet_url, sheet_name)
    if raw_rows and len(raw_rows) > 1:
        df = pd.DataFrame(raw_rows[1:], columns=raw_rows[0])
        st.dataframe(df.iloc[:, :5].tail(10), use_container_width=True) # Show last 10 rows
    else:
        st.info("No data found or check connection.")

with tab3:
    if st.button("🔥 Train on MNIST (Quick)"):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        st.session_state["model"].fit(x_train[:2000].reshape(-1, 28, 28, 1)/255.0, y_train[:2000], epochs=5, verbose=0)
        st.success("Trained!")
