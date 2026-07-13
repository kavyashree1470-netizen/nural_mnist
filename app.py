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
import graphviz # Added for the Playground Visualization

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

# ── SESSION STATE (PRO MODEL ARCHITECTURE) ──
MODEL_VERSION = "CNN_PRO_V2.5_PLAYGROUND"
if "model" not in st.session_state or st.session_state.get("m_ver") != MODEL_VERSION:
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    st.session_state["model"] = model
    st.session_state["m_ver"] = MODEL_VERSION
    st.session_state["is_trained"] = False
    st.session_state["train_history"] = {"acc": [], "loss": []}

if "canvas_key" not in st.session_state: st.session_state["canvas_key"] = 0

# ── VISUALIZATION: NN PLAYGROUND GRAPH ──
def draw_neural_playground():
    dot = graphviz.Digraph(comment='Neural Network Playground')
    dot.attr(rankdir='LR', bgcolor='#131a2e', fontcolor='white')
    dot.attr('node', shape='circle', color='#2dd4bf', fontcolor='white', style='filled', fillcolor='#1e293b')
    
    # Layers visualization
    dot.node('In', 'Input\n28x28')
    dot.node('C1', 'Conv2D\n32 Filters')
    dot.node('P1', 'MaxPool')
    dot.node('C2', 'Conv2D\n64 Filters')
    dot.node('P2', 'MaxPool')
    dot.node('F', 'Flatten')
    dot.node('D1', 'Dense\n128 Neurons')
    dot.node('Out', 'Output\n(0-9)')
    
    # Edges
    dot.edges(['InC1', 'C1P1', 'P1C2', 'C2P2', 'P2F', 'FD1', 'D1Out'])
    return dot

# ── GOOGLE SHEETS CORE ──
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

# ── PREPROCESSING (7 vs 2 Accuracy preserved) ──
def preprocess_drawing(image_data):
    gray = np.max(image_data[:, :, :3], axis=2).astype(np.uint8)
    if np.max(gray) < 30: return None
    coords = np.argwhere(gray > 30)
    y_min, x_min = coords.min(axis=0); y_max, x_max = coords.max(axis=0)
    cropped = gray[y_min:y_max+1, x_min:x_max+1]
    img = Image.fromarray(cropped, 'L')
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)
    new_img = Image.new('L', (28, 28), 0)
    new_img.paste(img, ((28 - img.size[0]) // 2, (28 - img.size[1]) // 2))
    return np.array(new_img)

# ── AUTOMATED TRAINING LOGIC ──
def perform_automated_training():
    if not st.session_state["is_trained"]:
        with st.status("🚀 Automated Training Active...", expanded=False) as status:
            (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
            x_train = x_train[:8000].reshape(-1, 28, 28, 1) / 255.0
            y_train = y_train[:8000]
            history = st.session_state["model"].fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)
            
            # Store history for playground chart
            st.session_state["train_history"]["acc"].extend(history.history['accuracy'])
            st.session_state["train_history"]["loss"].extend(history.history['loss'])
            
            st.session_state["is_trained"] = True
            status.update(label="Initial Model Trained!", state="complete")

def train_on_live_recording(spreadsheet_url, sheet_name):
    data = fetch_sheet_data(spreadsheet_url, sheet_name)
    if data and len(data) > 3:
        try:
            df = pd.DataFrame(data[1:], columns=data[0])
            y_live = df.iloc[:, 2].astype(int).values
            x_live = df.iloc[:, 5:].astype(float).values / 255.0
            x_live = x_live.reshape(-1, 28, 28, 1)
            history = st.session_state["model"].fit(x_live, y_live, epochs=2, batch_size=4, verbose=0)
            # Update history for live recording
            st.session_state["train_history"]["acc"].append(history.history['accuracy'][-1])
            st.session_state["train_history"]["loss"].append(history.history['loss'][-1])
            return True
        except: return False
    return False

# ── LOGIN SYSTEM ──
st.title("🧠 MNIST CNN Pro Studio")
op_name = st.text_input("Operator Login", placeholder="Enter name...")

if not op_name:
    st.markdown('<div class="lock-screen"><h2>🔒 System Locked</h2><p>Please enter your name to access the Studio.</p></div>', unsafe_allow_html=True)
    st.stop()

perform_automated_training()

# ── APP CONTENT ──
st.sidebar.title("📡 System Status")
spreadsheet_url = st.sidebar.text_input("Spreadsheet URL", value=st.secrets.get("SPREADSHEET_ID", ""))
sheet_name = st.sidebar.text_input("Sheet Name", value="Digits Data")

client = get_sheets_client()
if client: st.sidebar.markdown("Status: <span class='status-online'>● Online</span>", unsafe_allow_html=True)
else: st.sidebar.markdown("Status: <span class='status-offline'>○ Offline</span>", unsafe_allow_html=True)

tabs = st.tabs(["✏️ Sandbox", "📊 Neural Network Studio", "📋 Database Explorer"])

with tabs[0]:
    col1, col2, col3 = st.columns([2, 1.5, 1.5])
    with col1:
        st.subheader("Canvas")
        canvas_result = st_canvas(stroke_width=22, stroke_color="#FFF", background_color="#000", height=300, width=300, key=f"c_{st.session_state.canvas_key}")
        if st.button("Clear", use_container_width=True):
            st.session_state.canvas_key += 1; st.rerun()

    processed = preprocess_drawing(canvas_result.image_data) if canvas_result.image_data is not None else None

    with col2:
        st.subheader("Preprocessing")
        if processed is not None:
            st.image(processed, width=220, caption="CNN Ready Input")
        else: st.info("Draw on canvas.")

    with col3:
        st.subheader("Prediction")
        label = st.selectbox("True Label", list(range(10)))
        if processed is not None:
            inp = processed.reshape(1, 28, 28, 1).astype("float32") / 255.0
            preds = st.session_state["model"].predict(inp, verbose=0)[0]
            pred_digit = int(np.argmax(preds))
            st.markdown(f"## Prediction: `{pred_digit}`")
            st.progress(float(preds[pred_digit]))
            
            if st.button("🚀 Push & Auto-Train", use_container_width=True):
                if client:
                    sh = client.open_by_url(spreadsheet_url) if "http" in spreadsheet_url else client.open_by_key(spreadsheet_url)
                    wks = sh.worksheet(sheet_name)
                    row = [int(len(wks.get_all_values())), op_name, int(label), datetime.now().strftime("%H:%M:%S"), "NA"] + [int(p) for p in processed.flatten()]
                    wks.append_row(row)
                    with st.spinner("Live Training..."):
                        fetch_sheet_data.clear()
                        train_on_live_recording(spreadsheet_url, sheet_name)
                    st.toast("Saved & Re-trained!"); st.session_state.canvas_key += 1; st.rerun()

with tabs[1]:
    st.subheader("📊 Neural Network Playground")
    
    # NEW: Visual Playground Layout
    col_play, col_metrics = st.columns([1, 1])
    
    with col_play:
        st.markdown("#### Network Topology")
        st.graphviz_chart(draw_neural_playground())
        
        with st.expander("🔍 Layer Details (Playground Inspector)"):
            summary = []
            st.session_state["model"].summary(print_fn=lambda x: summary.append(x))
            st.code("\n".join(summary))

    with col_metrics:
        st.markdown("#### Training Metrics Chat/Log")
        if st.session_state["train_history"]["acc"]:
            # Display live accuracy chart
            metrics_df = pd.DataFrame({
                "Accuracy": st.session_state["train_history"]["acc"],
                "Loss": st.session_state["train_history"]["loss"]
            })
            st.line_chart(metrics_df, height=250)
            
            # Show specific values
            curr_acc = st.session_state["train_history"]["acc"][-1]
            curr_loss = st.session_state["train_history"]["loss"][-1]
            c1, c2 = st.columns(2)
            c1.metric("Current Accuracy", f"{curr_acc:.2%}")
            c2.metric("Current Loss", f"{curr_loss:.4f}")
        else:
            st.info("Log in to see training metrics.")

    if st.button("🔥 Force Full MNIST Refresh", use_container_width=True):
        st.session_state["is_trained"] = False; st.rerun()

with tabs[2]:
    st.subheader("📋 Database Explorer")
    if st.button("🔄 Refresh"): fetch_sheet_data.clear(); st.rerun()
    raw_data = fetch_sheet_data(spreadsheet_url, sheet_name)
    if raw_data and len(raw_data) > 1:
        df = pd.DataFrame(raw_data[1:], columns=raw_data[0])
        st.dataframe(df.iloc[:, :5].tail(15), use_container_width=True)
