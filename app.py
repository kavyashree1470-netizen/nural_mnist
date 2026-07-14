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
        background: #ff4b4b; border: 1px solid #ffffff; border-radius: 10px;
        padding: 15px; color: white; margin: 15px 0; border-left: 8px solid #800000;
        font-size: 1.1rem; font-weight: 900; box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
    }
    .lock-screen {
        text-align: center; padding: 40px; background: #131a2e; 
        border-radius: 20px; border: 2px dashed #a855f7; margin-top: 20px;
    }
    .svg-container {
        background: #f8fafc; border-radius: 15px; padding: 20px; 
        box-shadow: 0 10px 25px rgba(0,0,0,0.3); text-align: center;
    }
    @keyframes pulse {
        0% { stroke-width: 2; r: 15; }
        50% { stroke-width: 4; r: 17; }
        100% { stroke-width: 2; r: 15; }
    }
    .active-node { animation: pulse 1.5s infinite; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE (PRO MODEL ARCHITECTURE) ──
MODEL_VERSION = "CNN_PRO_V3.0_FINAL"
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
    st.session_state["last_preds"] = None # Store live activations

if "canvas_key" not in st.session_state: st.session_state["canvas_key"] = 0
if "predict_clicked" not in st.session_state: st.session_state["predict_clicked"] = False

# ── DYNAMIC NN GRAPH GENERATOR (REAL-TIME SVG) ──
def generate_nn_svg(activations=None):
    # layers_def: [node_count, color, label]
    # Representing a simplified view of the actual model
    layers_def = [
        [5, "#4a76c0", "Input (28x28)"], 
        [8, "#6366f1", "Conv+Pool"], 
        [6, "#8b5cf6", "Dense (128)"], 
        [10, "#f59e0b", "Output (0-9)"]
    ]
    width, height = 600, 350
    svg = f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
    
    # Draw Connections
    for i in range(len(layers_def) - 1):
        curr_l, next_l = layers_def[i], layers_def[i+1]
        x1, x2 = 60 + i * 160, 60 + (i + 1) * 160
        for c in range(curr_l[0]):
            y1 = (height / (curr_l[0] + 1)) * (c + 1)
            for n in range(next_l[0]):
                y2 = (height / (next_l[0] + 1)) * (n + 1)
                opacity = 0.15
                # If it's the connection to the output, make it highlight based on activation
                if i == 2 and activations is not None:
                    opacity = max(0.1, activations[n])
                svg += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#cbd5e1" stroke-width="1" stroke-opacity="{opacity}" />'

    # Draw Nodes
    for i, layer in enumerate(layers_def):
        x = 60 + i * 160
        svg += f'<text x="{x}" y="25" font-family="sans-serif" font-weight="bold" font-size="12" text-anchor="middle" fill="#475569">{layer[2]}</text>'
        for n in range(layer[0]):
            y = (height / (layer[0] + 1)) * (n + 1)
            fill_color = layer[1]
            extra_attr = ""
            
            # Real-time update for Output Layer
            if i == 3 and activations is not None:
                alpha = hex(int(max(0.2, activations[n]) * 255))[2:].zfill(2)
                fill_color = f"#10b981{alpha}" # Green highlight for active
                if activations[n] == max(activations):
                    extra_attr = 'class="active-node" stroke="#059669" stroke-width="3"'
            
            svg += f'<circle cx="{x}" cy="{y}" r="12" fill="{fill_color}" stroke="white" stroke-width="2" {extra_attr} />'
            
            # Label Output nodes 0-9
            if i == 3:
                svg += f'<text x="{x}" y="{y+4}" font-family="sans-serif" font-size="9" font-weight="bold" text-anchor="middle" fill="white">{n}</text>'
                
    svg += '</svg>'
    return svg

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

# ── PREPROCESSING (7 vs 2 Optimization) ──
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
        with st.status("🚀 Neural Network Initializing...", expanded=False) as status:
            (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
            x_train = x_train[:8000].reshape(-1, 28, 28, 1) / 255.0
            y_train = y_train[:8000]
            history = st.session_state["model"].fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)
            st.session_state["train_history"]["acc"].extend(history.history['accuracy'])
            st.session_state["train_history"]["loss"].extend(history.history['loss'])
            st.session_state["is_trained"] = True
            status.update(label="Training Complete!", state="complete")

def train_on_live_recording(spreadsheet_url, sheet_name):
    data = fetch_sheet_data(spreadsheet_url, sheet_name)
    if data and len(data) > 3:
        try:
            df = pd.DataFrame(data[1:], columns=data[0])
            y_live = df.iloc[:, 2].astype(int).values
            x_live = df.iloc[:, 5:].astype(float).values / 255.0
            x_live = x_live.reshape(-1, 28, 28, 1)
            history = st.session_state["model"].fit(x_live, y_live, epochs=2, batch_size=4, verbose=0)
            st.session_state["train_history"]["acc"].append(history.history['accuracy'][-1])
            st.session_state["train_history"]["loss"].append(history.history['loss'][-1])
            return True
        except: return False
    return False

# ── LOGIN SYSTEM ──
st.title("🧠 MNIST CNN Pro Studio")
op_name = st.text_input("Operator Login", placeholder="Enter name...")
if not op_name:
    st.markdown('<div class="lock-screen"><h2>🔒 System Locked</h2><p>Enter name to unlock.</p></div>', unsafe_allow_html=True)
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
        canvas_result = st_canvas(
            stroke_width=22, stroke_color="#FFF", background_color="#000", 
            height=300, width=300, drawing_mode="freedraw", key=f"c_{st.session_state.canvas_key}"
        )
        
        btn_c1, btn_c2 = st.columns(2)
        if btn_c1.button("Clear", use_container_width=True):
            st.session_state.canvas_key += 1
            st.session_state.predict_clicked = False
            st.session_state["last_preds"] = None
            st.rerun()
        
        if btn_c2.button("Check digit", use_container_width=True):
            st.session_state.predict_clicked = True

    processed = preprocess_drawing(canvas_result.image_data) if canvas_result.image_data is not None else None

    with col2:
        st.subheader("Preprocessing")
        if processed is not None:
            st.image(processed, width=220, caption="CNN Ready Input")
        else: st.info("Draw on canvas.")

    with col3:
        st.subheader("Prediction")
        true_label = st.selectbox("Assign True Label", list(range(10)))
        
        if processed is not None and st.session_state.predict_clicked:
            inp = processed.reshape(1, 28, 28, 1).astype("float32") / 255.0
            preds = st.session_state["model"].predict(inp, verbose=0)[0]
            st.session_state["last_preds"] = preds # Update global activations
            pred_digit = int(np.argmax(preds))
            conf = float(preds[pred_digit])
            
            st.markdown(f"## Prediction: `{pred_digit}`")
            st.progress(conf)
            
            if pred_digit != true_label:
                st.markdown(f'<div class="banner-warn">⚠️ MISMATCH DETECTED<br>Predicted: {pred_digit} | Label: {true_label}</div>', unsafe_allow_html=True)
            
            if st.button("🚀 Push & Live Train", use_container_width=True):
                if client:
                    try:
                        sh = client.open_by_url(spreadsheet_url) if "http" in spreadsheet_url else client.open_by_key(spreadsheet_url)
                        wks = sh.worksheet(sheet_name)
                        mismatch_text = "TRUE" if (pred_digit != true_label) else "FALSE"
                        row = [int(len(wks.get_all_values())), op_name, int(true_label), datetime.now().strftime("%H:%M:%S"), mismatch_text] + [int(p) for p in processed.flatten()]
                        wks.append_row(row)
                        with st.spinner("Fine-tuning..."):
                            fetch_sheet_data.clear()
                            train_on_live_recording(spreadsheet_url, sheet_name)
                        st.toast("Sync Success!")
                        st.session_state.canvas_key += 1
                        st.session_state.predict_clicked = False
                        st.rerun()
                    except Exception as e: st.error(f"Sync error: {e}")

with tabs[1]:
    st.subheader("📊 Neural Network Studio")
    col_graph, col_metrics = st.columns([1.2, 0.8])
    with col_graph:
        st.markdown("**Live Architecture Visualizer**")
        st.markdown('<div class="svg-container">', unsafe_allow_html=True)
        # Pass live activations to the SVG generator
        st.write(generate_nn_svg(st.session_state.get("last_preds")), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("The graph updates in real-time when 'Check Digit' is pressed in the Sandbox.")
    with col_metrics:
        if st.session_state["train_history"]["acc"]:
            metrics_df = pd.DataFrame({"Accuracy": st.session_state["train_history"]["acc"], "Loss": st.session_state["train_history"]["loss"]})
            st.line_chart(metrics_df, height=220)
            st.metric("Global Model Confidence", f"{st.session_state['train_history']['acc'][-1]:.1%}")

with tabs[2]:
    st.subheader("📋 Database Explorer")
    if st.button("🔄 Refresh"): fetch_sheet_data.clear(); st.rerun()
    raw_data = fetch_sheet_data(spreadsheet_url, sheet_name)
    if raw_data and len(raw_data) > 1:
        df = pd.DataFrame(raw_data[1:], columns=raw_data[0])
        st.dataframe(df.iloc[:, :5].tail(15), use_container_width=True)
