import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import plotly.express as px
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import json

# ── PAGE CONFIG ──
st.set_page_config(page_title="MNIST CNN Studio", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #f1f5f9; }
    .metric-card {
        background: #131a2e; border: 1px solid #1e293b; border-radius: 12px;
        padding: 15px; text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 800; color: #2dd4bf; }
    .banner-warn {
        background: #451a03; border: 1px solid #f59e0b; border-radius: 10px;
        padding: 14px; color: #fef3c7; margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── MODEL BUILDING (CNN) ──
def build_cnn_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2), # Req 3: Dropout added
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize or Re-initialize if shape is wrong
if "model" not in st.session_state or st.session_state.get("model_type") != "CNN_V2":
    st.session_state["model"] = build_cnn_model()
    st.session_state["model_type"] = "CNN_V2"
    st.session_state["is_trained"] = False

if "canvas_key" not in st.session_state: st.session_state["canvas_key"] = 0

# ── PREPROCESSING (Req 2: Centering) ──
def preprocess_drawing(image_data):
    # Convert RGBA to Grayscale
    gray = np.max(image_data[:, :, :3], axis=2).astype(np.uint8)
    if np.max(gray) < 20: return None
    
    # Find bounding box to center the digit
    coords = np.argwhere(gray > 20)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Crop and Resize to 20x20 (standard MNIST practice before padding)
    cropped = gray[y_min:y_max+1, x_min:x_max+1]
    pil_img = Image.fromarray(cropped, 'L')
    pil_img = pil_img.resize((20, 20), Image.Resampling.LANCZOS)
    
    # Pad to 28x28 (Centering)
    canvas = Image.new('L', (28, 28), 0)
    canvas.paste(pil_img, (4, 4)) 
    return np.array(canvas)

# ── MAIN UI ──
st.title("🧠 MNIST CNN Studio")

tab1, tab2 = st.tabs(["✏️ Sandbox", "📊 Training"])

with tab1:
    col1, col2, col3 = st.columns([2, 1, 1.5])
    
    with col1:
        st.subheader("Draw Digit")
        canvas_res = st_canvas(
            stroke_width=20, stroke_color="#FFF", background_color="#000",
            height=280, width=280, drawing_mode="freedraw", key=f"canv_{st.session_state.canvas_key}"
        )
        if st.button("Clear"):
            st.session_state.canvas_key += 1
            st.rerun()

    processed_img = None
    if canvas_res.image_data is not None:
        processed_img = preprocess_drawing(canvas_res.image_data)

    with col2:
        st.subheader("Metrics")
        if processed_img is not None:
            # Req 1: Active and Mean Pixel
            active_px = np.count_nonzero(processed_img > 30)
            mean_px = np.mean(processed_img)
            
            st.markdown(f"<div class='metric-card'><small>ACTIVE PIXELS</small><div class='metric-value'>{active_px}</div></div>", unsafe_allow_html=True)
            st.write("")
            st.markdown(f"<div class='metric-card'><small>MEAN PIXEL</small><div class='metric-value'>{mean_px:.1f}</div></div>", unsafe_allow_html=True)
            
            # Show small preview
            st.image(processed_img, width=150, caption="CNN Input")
        else:
            st.info("Draw something!")

    with col3:
        st.subheader("Prediction")
        assigned_label = st.selectbox("Assign Label", list(range(10)))
        
        if processed_img is not None:
            # Inference - Reshape to (1, 28, 28, 1) for CNN
            inp = processed_img.reshape(1, 28, 28, 1).astype("float32") / 255.0
            
            # Robust prediction call
            try:
                preds = st.session_state["model"].predict(inp, verbose=0)
                pred_digit = np.argmax(preds)
                conf = np.max(preds)
                
                st.markdown(f"## Digit: `{pred_digit}`")
                st.caption(f"Confidence: {conf*100:.1f}%")
                
                # Req 5: Warning if Label != Prediction
                if pred_digit != assigned_label:
                    st.markdown(f"""
                    <div class='banner-warn'>
                        ⚠️ <b>Warning:</b> You assigned <b>{assigned_label}</b>, 
                        but the CNN predicts <b>{pred_digit}</b>.
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error("Model error. Please click 'Hard Reset' in the Training tab.")

with tab2:
    if st.button("🚀 Train CNN Model (Quick)"):
        with st.spinner("Training on MNIST..."):
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train = x_train[:2000].reshape(-1, 28, 28, 1) / 255.0
            y_train = y_train[:2000]
            
            st.session_state["model"].fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
            st.session_state["is_trained"] = True
            st.success("Trained successfully!")

    if st.button("🗑️ Hard Reset App State"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
