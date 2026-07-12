import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import plotly.express as px
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageFilter
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import json
import os

# ── PAGE CONFIGURATION ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MNIST CNN Studio & Collector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── MODERN HIGH-CONTRAST CUSTOM STYLING ───────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #f1f5f9; }
    .main-header { font-family: 'Inter', sans-serif; color: #a855f7; font-weight: 800; }
    .sub-text { color: #94a3b8; font-size: 15px; margin-bottom: 25px; }
    .metric-card {
        background: #131a2e; border: 1px solid #1e293b; border-radius: 12px;
        padding: 16px; text-align: center; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .metric-card .value { font-size: 2rem; font-weight: 800; color: #2dd4bf; }
    .metric-card .label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; }
    .mnist-grid {
        display: grid; grid-template-columns: repeat(28, 8px);
        gap: 0; border: 3px solid #1e293b; border-radius: 8px;
        width: fit-content; margin: 0 auto; background-color: #000;
    }
    .mnist-cell { width: 8px; height: 8px; }
    .banner-warn {
        background: #451a03; border: 1px solid #f59e0b; border-radius: 10px;
        padding: 14px 18px; color: #fef3c7; font-size: 0.9rem; margin: 10px 0;
    }
    .banner-success {
        background: #064e3b; border: 1px solid #10b981; border-radius: 10px;
        padding: 14px 18px; color: #a7f3d0; margin: 10px 0;
    }
    .canvas-container {
        display: flex; justify-content: center; background-color: #000;
        border-radius: 12px; border: 3px solid #1e293b; padding: 10px; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE INITIALIZATION ──────────────────────────────────────────────
# We use a version key to force a reset if you switch from the old Dense model to CNN
MODEL_VERSION = "CNN_v1"

def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3), # Req 3: Dropout to prevent overfitting
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if "model" not in st.session_state or st.session_state.get("m_ver") != MODEL_VERSION:
    st.session_state["model"] = build_cnn()
    st.session_state["m_ver"] = MODEL_VERSION
    st.session_state["is_trained"] = False

# Initialize other state variables
for key, val in {
    "history": None, "synced_count": 0, "sync_logs": [], 
    "uploaded_creds": None, "active_drawing": None, "canvas_key": 0
}.items():
    if key not in st.session_state: st.session_state[key] = val

# ── GOOGLE SHEETS API CONNECTION ──────────────────────────────────────────────
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

def get_worksheet(client, spreadsheet_url, sheet_name):
    try:
        sh = client.open_by_url(spreadsheet_url) if "http" in spreadsheet_url else client.open_by_key(spreadsheet_url)
        try: return sh.worksheet(sheet_name)
        except gspread.exceptions.WorksheetNotFound:
            headers = ["id", "username", "label", "timestamp", "pred_label"] + [f"p{i}" for i in range(784)]
            wks = sh.add_worksheet(title=sheet_name, rows=1000, cols=800)
            wks.append_row(headers)
            return wks
    except Exception as e: return str(e)

# ── PREPROCESSING & ANALYTICS ─────────────────────────────────────────────────
def preprocess_drawing(image_data):
    """Req 2: Centering. Extracts bounding box and centers in 28x28."""
    gray = np.max(image_data[:, :, :3], axis=2).astype(np.uint8)
    if np.max(gray) < 20: return None
    
    # Bounding Box Logic
    coords = np.argwhere(gray > 20)
    y_min, x_min = coords.min(axis=0); y_max, x_max = coords.max(axis=0)
    cropped = gray[y_min:y_max+1, x_min:x_max+1]
    
    # Resize keeping aspect ratio
    pil_img = Image.fromarray(cropped, 'L')
    w, h = pil_img.size
    ratio = 20.0 / max(w, h)
    new_size = (int(w * ratio), int(h * ratio))
    pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Center in 28x28
    canvas = Image.new('L', (28, 28), 0)
    canvas.paste(pil_img, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))
    return np.array(canvas)

def get_pixel_metrics(arr):
    """Req 1: Active and Mean Pixel calculation."""
    active = np.count_nonzero(arr > 30)
    mean = np.mean(arr)
    return active, mean

# ── DATA LOADING (With CNN compatibility) ─────────────────────────────────────
@st.cache_data
def load_data(samples_per_digit=50):
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        
        idx = np.concatenate([np.where(y_train == i)[0][:samples_per_digit] for i in range(10)])
        return x_train[idx], y_train[idx], x_test[:100], y_test[:100]
    except Exception:
        # Fallback procedural generator (Req 4: ensuring CNN shape)
        images = []
        labels = []
        for d in range(10):
            for _ in range(samples_per_digit):
                img = np.zeros((28, 28, 1), dtype=np.float32)
                img[10:18, 13:15] = 1.0 # Simple procedural '1' shape
                images.append(img); labels.append(d)
        return np.array(images), np.array(labels), np.array(images[:20]), np.array(labels[:20])

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("<h2 style='color:#a855f7;'>💾 Cloud Sync</h2>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload Service Account JSON", type=["json"])
if uploaded_file: st.session_state["uploaded_creds"] = json.load(uploaded_file)

spreadsheet_url = st.sidebar.text_input("Spreadsheet URL/ID", value=st.secrets.get("SPREADSHEET_ID", ""))
sheet_name = st.sidebar.text_input("Worksheet Title", value="Digits Data")

st.sidebar.markdown("<h2 style='color:#a855f7;'>⚙️ Hyperparameters</h2>", unsafe_allow_html=True)
epochs_val = st.sidebar.slider("Training Epochs", 5, 50, 15)
batch_size_val = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
samples_val = st.sidebar.slider("Samples per Digit", 10, 200, 50)

if st.sidebar.button("🔄 Hard Reset Model & State", use_container_width=True):
    for key in list(st.session_state.keys()): del st.session_state[key]
    st.rerun()

# ── MAIN DASHBOARD ────────────────────────────────────────────────────────────
st.markdown("<h1 class='main-header'>🧠 MNIST CNN Studio & Database Collector</h1>", unsafe_allow_html=True)

tab_sandbox, tab_train, tab_database = st.tabs(["✏️ Sandbox", "📊 CNN Training", "📋 DB Explorer"])

with tab_sandbox:
    col_draw, col_monitor, col_save = st.columns([2, 1.5, 1.5], gap="large")
    
    with col_draw:
        st.subheader("🖊️ Canvas")
        st.markdown("<div class='canvas-container'>", unsafe_allow_html=True)
        canvas_result = st_canvas(
            stroke_width=20, stroke_color="#FFF", background_color="#000",
            height=280, width=280, drawing_mode="freedraw", key=f"cnv_{st.session_state.canvas_key}"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Canvas", use_container_width=True):
            st.session_state.canvas_key += 1; st.rerun()

    with col_monitor:
        st.subheader("🔍 Preprocessed")
        if canvas_result.image_data is not None:
            processed = preprocess_drawing(canvas_result.image_data)
            if processed is not None:
                st.session_state.active_drawing = processed
                # Req 1: Pixel Metrics
                act, avg = get_pixel_metrics(processed)
                c1, c2 = st.columns(2)
                c1.markdown(f"<div class='metric-card'><div class='value'>{act}</div><div class='label'>Active Pixels</div></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-card'><div class='value'>{avg:.1f}</div><div class='label'>Mean Pixel</div></div>", unsafe_allow_html=True)
                
                # Visual Preview
                cells = "".join([f'<div class="mnist-cell" style="background:rgb({v},{v//2},{v})"></div>' for v in processed.flatten()])
                st.markdown(f'<div class="mnist-grid">{cells}</div>', unsafe_allow_html=True)
        else: st.info("Draw to begin.")

    with col_save:
        st.subheader("🚀 Prediction")
        operator = st.text_input("Operator Name", value="Dev")
        assigned_label = st.selectbox("Assign True Digit Label", list(range(10)))
        
        if st.session_state.active_drawing is not None:
            # Inference
            inp = st.session_state.active_drawing.reshape(1, 28, 28, 1).astype("float32") / 255.0
            preds = st.session_state["model"].predict(inp, verbose=0)[0]
            pred_digit = np.argmax(preds)
            conf = preds[pred_digit]
            
            st.markdown(f"### Predicted: <span style='color:#a855f7; font-size:32px;'>{pred_digit}</span>", unsafe_allow_html=True)
            st.progress(float(conf), text=f"Confidence: {conf*100:.1f}%")
            
            # Req 5: Mismatch Warning
            if pred_digit != assigned_label:
                st.markdown(f"""
                <div class='banner-warn'>
                    ⚠️ <b>Mismatch Warning:</b> You assigned <b>{assigned_label}</b>, but the CNN predicts <b>{pred_digit}</b>. 
                    Please check the label or redraw.
                </div>
                """, unsafe_allow_html=True)

            if st.button("🚀 Push to Google Sheets DB", use_container_width=True):
                client = get_sheets_client()
                if client:
                    wks = get_worksheet(client, spreadsheet_url, sheet_name)
                    if not isinstance(wks, str):
                        row = [len(wks.get_all_values()), operator, assigned_label, datetime.now().isoformat(), pred_digit] + st.session_state.active_drawing.flatten().tolist()
                        wks.append_row(row)
                        st.session_state.synced_count += 1
                        st.toast("Saved to Cloud!", icon="🔥")
                        st.session_state.canvas_key += 1; st.rerun()
                else: st.error("Sheets Offline - check sidebar.")

with tab_train:
    st.subheader("🧠 Model Operations")
    if st.button("🔥 Launch CNN Training", use_container_width=True):
        xt, yt, xv, yv = load_data(samples_val)
        hist = st.session_state["model"].fit(xt, yt, validation_data=(xv, yv), epochs=epochs_val, batch_size=batch_size_val, verbose=0)
        st.session_state["history"] = pd.DataFrame(hist.history)
        st.session_state["is_trained"] = True
        st.balloons()

    if st.session_state["history"] is not None:
        h = st.session_state["history"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=h['accuracy'], name="Train Acc", line=dict(color="#a855f7", width=3)))
        fig.add_trace(go.Scatter(y=h['val_accuracy'], name="Val Acc", line=dict(color="#2dd4bf", dash='dash')))
        fig.update_layout(template="plotly_dark", title="CNN Training Progress", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

with tab_database:
    st.subheader("📋 Dataset Explorer")
    client = get_sheets_client()
    if client and spreadsheet_url:
        wks = get_worksheet(client, spreadsheet_url, sheet_name)
        if not isinstance(wks, str):
            data = wks.get_all_values()
            if len(data) > 1:
                df = pd.DataFrame(data[1:], columns=data[0])
                st.dataframe(df.iloc[:, :5], use_container_width=True)
                
                # Digit distribution
                dist = df['label'].value_counts().reset_index()
                fig_dist = px.bar(dist, x='label', y='count', color='count', color_continuous_scale="Purples")
                fig_dist.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_dist, use_container_width=True)
    else: st.warning("Connect Google Sheets to explore the live database.")
