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
st.set_page_config(
    page_title="MNIST Neural Network Studio & Collector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── MODERN HIGH-CONTRAST CUSTOM STYLING ───────────────────────────────────────
st.markdown("""
<style>
    /* Dark Theme Base overrides */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    .main-header {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        color: #a855f7;
        font-weight: 800;
        letter-spacing: -0.75px;
        margin-bottom: 5px;
    }
    .sub-text {
        color: #94a3b8;
        font-size: 15px;
        margin-bottom: 25px;
    }
    /* Info badge classes */
    .info-badge {
        display: inline-block;
        background: #1e1548;
        color: #c084fc;
        border: 1px solid #7c3aed;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    /* Metric cards */
    .metric-card {
        background: #131a2e;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .metric-card .value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2dd4bf;
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    /* Interactive Mini Pixel Grid for MNIST Preview */
    .mnist-grid {
        display: grid;
        grid-template-columns: repeat(28, 8px);
        gap: 0;
        border: 3px solid #1e293b;
        border-radius: 8px;
        width: fit-content;
        margin: 0 auto;
        background-color: #000;
    }
    .mnist-cell {
        width: 8px;
        height: 8px;
    }
    /* Banner blocks */
    .banner-success {
        background: #064e3b;
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 14px 18px;
        color: #a7f3d0;
        font-size: 0.9rem;
        margin: 10px 0;
    }
    .banner-warn {
        background: #451a03;
        border: 1px solid #f59e0b;
        border-radius: 10px;
        padding: 14px 18px;
        color: #fef3c7;
        font-size: 0.9rem;
        margin: 10px 0;
    }
    .banner-info {
        background: #1e1b4b;
        border: 1px solid #6366f1;
        border-radius: 10px;
        padding: 14px 18px;
        color: #e0e7ff;
        font-size: 0.9rem;
        margin: 10px 0;
    }
    /* Canvas container centering */
    .canvas-container {
        display: flex;
        justify-content: center;
        background-color: #000000;
        border-radius: 12px;
        border: 3px solid #1e293b;
        padding: 10px;
        margin-bottom: 10px;
    }
    /* Streamlit Sidebar customization */
    .sidebar-section {
        background-color: #0f172a;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE INITIALIZATION ──────────────────────────────────────────────
if "model" not in st.session_state:
    # Set up our default Keras Sequential Model
    model = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, activation='relu', name='hidden_1'),
        layers.Dense(64, activation='relu', name='hidden_2'),
        layers.Dense(10, activation='softmax', name='output')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    st.session_state["model"] = model
    st.session_state["is_trained"] = False

if "history" not in st.session_state:
    st.session_state["history"] = None

if "synced_count" not in st.session_state:
    st.session_state["synced_count"] = 0

if "sync_logs" not in st.session_state:
    st.session_state["sync_logs"] = []

if "uploaded_creds" not in st.session_state:
    st.session_state["uploaded_creds"] = None

if "active_drawing" not in st.session_state:
    st.session_state["active_drawing"] = None

if "canvas_key" not in st.session_state:
    st.session_state["canvas_key"] = 0

# ── GOOGLE SHEETS API CONNECTION MANAGER ──────────────────────────────────────
def get_sheets_client():
    """
    Highly robust authentication that checks uploaded JSON files,
    Streamlit secrets, or local credentials to find service accounts.
    """
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    
    # Method 1: Check session-uploaded credentials
    if st.session_state["uploaded_creds"] is not None:
        try:
            creds = Credentials.from_service_account_info(st.session_state["uploaded_creds"], scopes=scopes)
            return gspread.authorize(creds)
        except Exception as e:
            st.error(f"Error with uploaded JSON credentials: {e}")
            
    # Method 2: Check Streamlit Secrets (gcp_service_account)
    if "gcp_service_account" in st.secrets:
        try:
            creds_info = st.secrets["gcp_service_account"]
            if isinstance(creds_info, str):
                creds_info = json.loads(creds_info)
            creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
            return gspread.authorize(creds)
        except Exception as e:
            st.sidebar.error(f"Error with secrets credentials: {e}")

    # Method 3: Check local service_account.json file
    if os.path.exists("service_account.json"):
        try:
            with open("service_account.json", "r") as f:
                creds_info = json.load(f)
            creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
            return gspread.authorize(creds)
        except Exception as e:
            st.sidebar.error(f"Error reading service_account.json: {e}")

    # Method 4: Check local credentials.json file
    if os.path.exists("credentials.json"):
        try:
            creds = Credentials.from_service_account_file("credentials.json", scopes=scopes)
            return gspread.authorize(creds)
        except Exception as e:
            st.sidebar.error(f"Error reading credentials.json: {e}")

    return None

def get_worksheet(client, spreadsheet_url, sheet_name):
    """
    Opens the requested spreadsheet and worksheet. If the worksheet doesn't exist,
    automatically provisions it with standard column headers!
    """
    try:
        if "docs.google.com/spreadsheets" in spreadsheet_url:
            sh = client.open_by_url(spreadsheet_url)
        else:
            sh = client.open_by_key(spreadsheet_url)
            
        try:
            return sh.worksheet(sheet_name)
        except gspread.exceptions.WorksheetNotFound:
            # Create a new worksheet automatically with headers
            headers = ["id", "username", "label", "timestamp", "is_mismatch"] + [f"pixel_{i}" for i in range(784)]
            wks = sh.add_worksheet(title=sheet_name, rows=1000, cols=800)
            wks.append_row(headers)
            return wks
    except Exception as e:
        return f"Spreadsheet access failed: {e}"

@st.cache_data(ttl=10)
def load_gsheet_records(spreadsheet_url, sheet_name):
    """Loads and returns dataset metadata/records from the live Google Sheet."""
    client = get_sheets_client()
    if not client or not spreadsheet_url:
        return pd.DataFrame()
    
    wks_or_err = get_worksheet(client, spreadsheet_url, sheet_name)
    if isinstance(wks_or_err, str):
        return pd.DataFrame()
        
    try:
        all_cells = wks_or_err.get_all_values()
        if len(all_cells) <= 1:
            return pd.DataFrame()
            
        headers = [h.strip() for h in all_cells[0][:5]]
        rows = [row[:5] for row in all_cells[1:]]
        return pd.DataFrame(rows, columns=headers)
    except Exception:
        return pd.DataFrame()

# ── SYNTHETIC DIGIT GENERATOR & MNIST LOADER ──────────────────────────────────
@st.cache_data
def load_or_generate_mnist_dataset(samples_per_digit=50):
    """
    Downloads the real MNIST dataset, or falls back to highly-detailed synthetic
    procedural digits generator if internet/Keras servers are unreachable.
    """
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Flatten and normalize
        x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
        
        # Balance dataset with exact samples per digit label
        sliced_x, sliced_y = [], []
        for digit in range(10):
            idxs = np.where(y_train == digit)[0][:samples_per_digit]
            sliced_x.append(x_train[idxs])
            sliced_y.append(y_train[idxs])
            
        x_train_sliced = np.concatenate(sliced_x, axis=0)
        y_train_sliced = np.concatenate(sliced_y, axis=0)
        
        # Shuffle
        shf = np.random.permutation(len(x_train_sliced))
        return x_train_sliced[shf], y_train_sliced[shf], x_test[:100], y_test[:100]
    except Exception as e:
        # Procedural fallback generator with geometric noise
        images, labels = [], []
        for digit in range(10):
            for _ in range(samples_per_digit):
                canvas = np.zeros((28, 28), dtype=np.float32)
                
                # Draw stylized geometric approximations of digits
                if digit == 0:
                    for theta in np.linspace(0, 2*np.pi, 20):
                        r_y, r_x = int(14 + 6*np.sin(theta)), int(14 + 5*np.cos(theta))
                        canvas[max(0, min(27, r_y)), max(0, min(27, r_x))] = 0.8
                elif digit == 1:
                    canvas[5:23, 14] = 0.95
                elif digit == 2:
                    canvas[5:8, 11:17] = 0.9
                    canvas[8:14, 16] = 0.9
                    canvas[14:19, 11:17] = 0.9
                    canvas[19:22, 11] = 0.9
                    canvas[22, 11:18] = 0.95
                elif digit == 3:
                    canvas[5, 10:18] = 0.9
                    canvas[5:23, 17] = 0.9
                    canvas[14, 11:18] = 0.9
                    canvas[22, 10:18] = 0.9
                elif digit == 4:
                    canvas[5:14, 11] = 0.9
                    canvas[14, 11:18] = 0.9
                    canvas[5:23, 16] = 0.95
                elif digit == 5:
                    canvas[5, 11:18] = 0.9
                    canvas[5:14, 11] = 0.9
                    canvas[14, 11:18] = 0.9
                    canvas[14:23, 17] = 0.9
                    canvas[22, 11:17] = 0.9
                elif digit == 6:
                    canvas[5:23, 11] = 0.9
                    canvas[13:23, 17] = 0.9
                    canvas[13, 11:18] = 0.9
                    canvas[22, 11:18] = 0.9
                elif digit == 7:
                    canvas[5, 10:18] = 0.95
                    for dy in range(5, 23):
                        dx = int(17 - (dy-5)*0.35)
                        canvas[dy, max(0, min(27, dx))] = 0.9
                elif digit == 8:
                    for theta in np.linspace(0, 2*np.pi, 15):
                        r_y, r_x = int(10 + 4*np.sin(theta)), int(14 + 4*np.cos(theta))
                        canvas[r_y, r_x] = 0.8
                        r_y2, r_x2 = int(18 + 4*np.sin(theta)), int(14 + 4*np.cos(theta))
                        canvas[r_y2, r_x2] = 0.8
                elif digit == 9:
                    canvas[5:14, 11] = 0.9
                    canvas[5:14, 17] = 0.9
                    canvas[5, 11:18] = 0.9
                    canvas[13, 11:18] = 0.9
                    canvas[5:23, 17] = 0.95
                
                # Add Gaussian-like blur using Pillow, and uniform noise
                from PIL import ImageFilter
                pil_canvas = Image.fromarray((canvas.reshape(28, 28) * 255).astype(np.uint8), 'L')
                pil_canvas = pil_canvas.filter(ImageFilter.GaussianBlur(radius=0.8))
                canvas = np.array(pil_canvas).astype(np.float32) / 255.0
                canvas += np.random.uniform(0.0, 0.12, (28, 28))
                canvas = np.clip(canvas, 0.0, 1.0).flatten()
                images.append(canvas)
                labels.append(digit)
                
        images = np.array(images)
        labels = np.array(labels)
        shf = np.random.permutation(len(images))
        return images[shf], labels[shf], images[:100], labels[:100]

# ── INTELLIGENT HANDWRITING IMAGE PREPROCESSING ───────────────────────────────
def preprocess_drawing(image_data):
    """
    Downsamples the raw RGBA streamlit drawing canvas into a standard 28x28 grayscale
    MNIST pixel representation (cropping and geometrically centering the digit).
    """
    rgb = image_data[:, :, :3]
    gray = np.max(rgb, axis=2) # Convert to 2D grayscale array
    
    # Clip background noise
    gray[gray < 15] = 0
    
    if np.max(gray) == 0:
        return None # Empty board
        
    # Find bounding box of the actual drawing coordinates
    coords = np.argwhere(gray > 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Extract the cropped region of interest
    cropped = gray[y_min:y_max+1, x_min:x_max+1]
    
    # Resize the crop to fit inside a standard 20x20 bounding box (keeping aspect ratio)
    h, w = cropped.shape
    pil_crop = Image.fromarray(cropped.astype('uint8'), 'L')
    
    if h > w:
        new_h = 20
        new_w = max(1, int(20 * (w / h)))
    else:
        new_w = 20
        new_h = max(1, int(20 * (h / w)))
        
    pil_resized = pil_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_arr = np.array(pil_resized)
    
    # Place inside standard black 28x28 grid, centered geometrically
    mnist_grid = np.zeros((28, 28), dtype=np.uint8)
    offset_y = (28 - new_h) // 2
    offset_x = (28 - new_w) // 2
    
    mnist_grid[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized_arr
    return mnist_grid

# ── DIGIT HEURISTIC MISMATCH VALIDATOR ────────────────────────────────────────
def check_digit_mismatch(img_28x28, label):
    """
    Performs defensive quadrant analysis on drawn pixels to warn users if their drawing
    is highly mismatched with the selected target label.
    """
    if img_28x28 is None:
        return False
        
    mask = img_28x28 > 30
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
        
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    h = (y_max - y_min) + 1
    w = (x_max - x_min) + 1
    aspect_ratio = w / float(h) if h > 0 else 1.0
    
    # Broad structural bounds checks
    if label == 1 and aspect_ratio > 0.5:
        return True # "1" should be thin and vertical
    if label != 1 and aspect_ratio < 0.2:
        return True # Non-"1" digits cannot be a single thin vertical strip
        
    # Segment image into quadrants
    mid_y, mid_x = y_min + h//2, x_min + w//2
    q_top_left = float(img_28x28[y_min:mid_y, x_min:mid_x].sum())
    q_top_right = float(img_28x28[y_min:mid_y, mid_x:x_max+1].sum())
    q_bot_left = float(img_28x28[mid_y:y_max+1, x_min:mid_x].sum())
    q_bot_right = float(img_28x28[mid_y:y_max+1, mid_x:x_max+1].sum())
    
    total = q_top_left + q_top_right + q_bot_left + q_bot_right
    if total == 0:
        return False
        
    p1, p2, p3, p4 = q_top_left/total, q_top_right/total, q_bot_left/total, q_bot_right/total
    
    # Specific Digit Structural Rules
    if label == 0:
        if min(p1, p2, p3, p4) < 0.10:
            return True # Missing ink in one of the outer bounds
    elif label == 3:
        if (p1 + p3) > (p2 + p4) * 1.3:
            return True # Too much ink on the left
    elif label == 7:
        if (p1 + p2) < (p3 + p4) * 0.7:
            return True # Not top-heavy enough
            
    return False

# ── RENDER MINI MNIST GRAPHICAL PREVIEW ───────────────────────────────────────
def get_mnist_html_preview(arr):
    """Generates styled HTML grid displaying normalized grayscale intensity."""
    cells = ""
    for row in arr:
        for val in row:
            val = int(val)
            if val > 15:
                # Color code intensity gradient using custom purple-violet scale
                r = min(255, 60 + int(val * 0.75))
                g = min(255, int(val * 0.3))
                b = min(255, 120 + int(val * 0.5))
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
            else:
                hex_color = "#070a13"
            cells += f'<div class="mnist-cell" style="background: {hex_color};"></div>'
    return f'<div class="mnist-grid">{cells}</div>'


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("<h2 style='color:#a855f7; margin-bottom:10px;'>💾 Data Storage Sync</h2>", unsafe_allow_html=True)

# 1. Credentials File Config
st.sidebar.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
st.sidebar.markdown("**🔑 Google Service Account Account**")
uploaded_file = st.sidebar.file_uploader("Upload credentials.json or service_account.json", type=["json"], label_visibility="collapsed")
if uploaded_file is not None:
    try:
        st.session_state["uploaded_creds"] = json.load(uploaded_file)
        st.sidebar.success("🎉 Custom Credentials uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Invalid JSON format: {e}")

# 2. Spreadsheet Setup
sheets_client = get_sheets_client()
if sheets_client:
    st.sidebar.success("🟢 Google API Authorized!")
else:
    st.sidebar.warning("🟡 Sheets Status: Offline")
    st.sidebar.info("Upload your credentials.json file above to activate live cloud syncing.")

spreadsheet_url = st.sidebar.text_input(
    "Spreadsheet URL / ID",
    value=st.secrets.get("SPREADSHEET_ID", ""),
    placeholder="Paste your target spreadsheet link..."
)
sheet_name = st.sidebar.text_input("Worksheet Title", value="Digits Data")

if sheets_client and spreadsheet_url:
    st.sidebar.info("💡 Tip: Share your Google Sheet with the `client_email` listed in your JSON credentials so the app has permission to write data.")
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# 3. Model Parameters Sidebar
st.sidebar.markdown("<h2 style='color:#a855f7; margin-top:20px; margin-bottom:10px;'>⚙️ Hyperparameters</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
epochs_val = st.sidebar.slider("Training Epochs", min_value=5, max_value=50, value=15, step=5)
batch_size_val = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
samples_val = st.sidebar.slider("Samples per Digit Label", min_value=10, max_value=200, value=50, step=10)
learning_rate_val = st.sidebar.select_slider("Learning Rate", [0.0001, 0.001, 0.01, 0.1], value=0.001)

if st.sidebar.button("🔄 Reset Keras Network Weights", use_container_width=True):
    # Reinitialize weights
    st.session_state["model"] = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, activation='relu', name='hidden_1'),
        layers.Dense(64, activation='relu', name='hidden_2'),
        layers.Dense(10, activation='softmax', name='output')
    ])
    st.session_state["model"].compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_val),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    st.session_state["history"] = None
    st.session_state["is_trained"] = False
    st.sidebar.success("🧠 Weights successfully reinitialized!")
st.sidebar.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<h1 class='main-header'>🧠 MNIST Neural Network Sandbox & Database Collector</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>A fully featured neural network client enabling drawing and syncing handwriting samples directly to Google Sheets, with an integrated real-time TensorFlow/Keras training dashboard.</p>", unsafe_allow_html=True)

# Main Navigation Tabs
tab_sandbox, tab_train, tab_database = st.tabs([
    "✏️ Hand-Written Digit Sandbox",
    "📊 Keras Neural Network Studio",
    "📋 Google Sheets Live Database Explorer"
])

# ──────────────────────────────────────────────────────────────────────────────
#  TAB 1: HAND-WRITTEN DIGIT SANDBOX
# ──────────────────────────────────────────────────────────────────────────────
with tab_sandbox:
    col_draw, col_monitor, col_save = st.columns([2, 1.5, 1.5], gap="large")
    
    with col_draw:
        st.subheader("🖊️ Draw Digit Canvas")
        st.markdown("<p style='color:#94a3b8; font-size:13px;'>Draw a single digit (0-9) inside the block. Center it clearly and use standard thick stroke widths for maximum prediction accuracy.</p>", unsafe_allow_html=True)
        
        # Draw board component
        st.markdown("<div class='canvas-container'>", unsafe_allow_html=True)
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            update_streamlit=True,
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
            display_toolbar=False
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Clear button logic
        if st.button("🗑️ Clear Active Canvas", use_container_width=True):
            st.session_state.canvas_key += 1
            st.session_state.active_drawing = None
            st.rerun()
            
        # Extract canvas drawing pixels
        if canvas_result.image_data is not None:
            rgba_pixels = canvas_result.image_data.astype(np.uint8)
            mnist_pixels = preprocess_drawing(rgba_pixels)
            st.session_state.active_drawing = mnist_pixels
            
    with col_monitor:
        st.subheader("🔍 Preprocessed MNIST Grid")
        st.markdown("<p style='color:#94a3b8; font-size:13px;'>View your handwriting downsampled and centered geometrically into the standard 28x28 grayscale format.</p>", unsafe_allow_html=True)
        
        if st.session_state.active_drawing is not None:
            arr_28x28 = st.session_state.active_drawing
            st.markdown(get_mnist_html_preview(arr_28x28), unsafe_allow_html=True)
            st.write("")
            
            # Download Button for 28x28 image
            pil_img = Image.fromarray(arr_28x28, 'L')
            import io
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            st.download_button(
                label="📥 Download 28x28 Grayscale PNG",
                data=img_byte_arr,
                file_name=f"mnist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True
            )
            
            with st.expander("📊 Show Raw 28x28 Numerical Array"):
                st.dataframe(pd.DataFrame(arr_28x28), use_container_width=True)
        else:
            st.info("Draw a digit on the left board to render real-time preprocessed previews.")
            
    with col_save:
        st.subheader("🚀 Prediction & Cloud Sync")
        
        username_val = st.text_input("👤 Operator Name", value="Developer", placeholder="Enter your name...")
        assigned_label = st.selectbox("Assign True Digit Label", list(range(10)))
        
        st.markdown("---")
        
        if st.session_state.active_drawing is not None:
            arr_28x28 = st.session_state.active_drawing
            
            # Run inference on Keras Model
            flat_vector = arr_28x28.flatten() / 255.0
            model = st.session_state["model"]
            predictions = model.predict(np.expand_dims(flat_vector, axis=0), verbose=0)[0]
            pred_digit = np.argmax(predictions)
            confidence = predictions[pred_digit]
            
            is_trained = st.session_state.get("is_trained", False)
            trained_status_note = "" if is_trained else " *(Untrained Initial Weights)*"
            
            st.markdown(f"### Predicted: <span style='color:#a855f7; font-size:32px;'>{pred_digit}</span> {trained_status_note}", unsafe_allow_html=True)
            st.progress(float(confidence), text=f"Confidence: {confidence*100:.1f}%")
            
            # Trigger Mismatch Heuristic warning
            mismatch_detected = check_digit_mismatch(arr_28x28, assigned_label)
            if mismatch_detected:
                st.markdown("""
                <div class='banner-warn'>
                    ⚠️ <b>Heuristic Mismatch Detected:</b> The drawing shape does not closely resemble the digit label you assigned! Please verify the digit label selection or redraw carefully.
                </div>
                """, unsafe_allow_html=True)
                
            force_override = False
            if mismatch_detected:
                force_override = st.checkbox("⚠️ Force push mismatched entry anyway?", value=False)
                
            # Submit to spreadsheet database button
            if st.button("🚀 Push to Google Sheets DB", use_container_width=True):
                if not sheets_client:
                    st.error("Google Sheets API Connection is offline. Please upload or configure credentials in the sidebar.")
                elif not spreadsheet_url:
                    st.error("Please configure your Spreadsheet ID / URL in the sidebar.")
                elif mismatch_detected and not force_override:
                    st.warning("Please correct your digit label or check the 'Force push' box above to save this record.")
                else:
                    with st.spinner("Uploading row to Google Sheet..."):
                        try:
                            sh_wks = get_worksheet(sheets_client, spreadsheet_url, sheet_name)
                            if isinstance(sh_wks, str):
                                st.error(sh_wks)
                            else:
                                all_cells = sh_wks.get_all_values()
                                next_id = len(all_cells)
                                
                                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                mismatch_flag = "TRUE" if mismatch_detected else "FALSE"
                                pixels_list = [int(v) for v in arr_28x28.flatten()]
                                
                                row_data = [next_id, username_val, assigned_label, timestamp_str, mismatch_flag] + pixels_list
                                sh_wks.append_row(row_data)
                                
                                st.session_state["synced_count"] += 1
                                st.session_state["sync_logs"].insert(0, f"Digit {assigned_label} (Pred: {pred_digit}) successfully pushed!")
                                
                                st.toast("Sample saved successfully to Cloud Sheet!", icon="🔥")
                                st.markdown("""
                                <div class='banner-success'>
                                    🎉 <b>Database Success:</b> Entry added securely to your active sheet worksheet!
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Auto-clear canvas
                                st.session_state.canvas_key += 1
                                st.session_state.active_drawing = None
                                st.rerun()
                                
                        except Exception as ex:
                            st.error(f"Failed to push entry: {ex}")
        else:
            st.info("Start drawing in the left coordinate panel to calculate real-time Keras predictions.")

# ──────────────────────────────────────────────────────────────────────────────
#  TAB 2: KERAS NEURAL NETWORK STUDIO
# ──────────────────────────────────────────────────────────────────────────────
with tab_train:
    col_t_btn, col_charts = st.columns([1.5, 2], gap="large")
    
    with col_t_btn:
        st.subheader("🧠 Model Operations")
        st.markdown("<p style='color:#94a3b8; font-size:13px;'>Train a multi-layer fully-connected feedforward deep neural network inside Streamlit. Choose settings from the sidebar, then click Train.</p>", unsafe_allow_html=True)
        
        col_grid1, col_grid2 = st.columns(2)
        with col_grid1:
            st.metric("Model Architecture", "Dense (784 -> 128 -> 64 -> 10)")
        with col_grid2:
            st.metric("Total Params", "109,386 Trainable")
            
        st.write("")
        
        # Launch training
        if st.button("🔥 Launch Real-Time Model Training", use_container_width=True):
            with st.spinner("Slicing balanced datasets and preparing SGD optimizers..."):
                x_train, y_train, x_val, y_val = load_or_generate_mnist_dataset(samples_val)
                prog_bar = st.progress(0.0, text="Initializing training epochs...")
                
                # Custom callback to stream accuracies/losses directly to Streamlit UI
                class StreamProgressCallback(tf.keras.callbacks.Callback):
                    def __init__(self, total):
                        super().__init__()
                        self.total = total
                        self.loss = []
                        self.acc = []
                        self.v_loss = []
                        self.v_acc = []
                        
                    def on_epoch_end(self, epoch, logs=None):
                        logs = logs or {}
                        self.loss.append(logs.get('loss', 0))
                        self.acc.append(logs.get('accuracy', 0))
                        self.v_loss.append(logs.get('val_loss', 0))
                        self.v_acc.append(logs.get('val_accuracy', 0))
                        
                        ratio = (epoch + 1) / self.total
                        prog_bar.progress(
                            ratio, 
                            text=f"Epoch {epoch+1}/{self.total} | Loss: {logs.get('loss',0):.3f} | Accuracy: {logs.get('accuracy',0)*100:.1f}%"
                        )
                
                cb = StreamProgressCallback(epochs_val)
                st.session_state["model"].optimizer.learning_rate.assign(learning_rate_val)
                
                # Train model
                st.session_state["model"].fit(
                    x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=epochs_val,
                    batch_size=batch_size_val,
                    callbacks=[cb],
                    verbose=0
                )
                
                # Cache results
                metrics_df = pd.DataFrame({
                    "Epoch": list(range(1, epochs_val + 1)),
                    "Train Loss": cb.loss,
                    "Train Accuracy": cb.acc,
                    "Val Loss": cb.v_loss,
                    "Val Accuracy": cb.v_acc
                })
                st.session_state["history"] = metrics_df
                st.session_state["is_trained"] = True
                
                st.balloons()
                st.toast("Model trained successfully!", icon="🚀")
                
        if st.session_state["is_trained"]:
            st.success("🟢 Current Keras Model is Trained & Validated!")
        else:
            st.warning("🟡 Model state: Using Untrained/Randomized weights.")
            
    with col_charts:
        st.subheader("📈 Real-Time Optimization Curves")
        
        if st.session_state["history"] is not None:
            hist = st.session_state["history"]
            
            # Plot Accuracy Curves
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=hist["Epoch"], y=hist["Train Accuracy"], name="Training Accuracy", line=dict(color="#a855f7", width=3)))
            fig_acc.add_trace(go.Scatter(x=hist["Epoch"], y=hist["Val Accuracy"], name="Validation Accuracy", line=dict(color="#2dd4bf", width=2.5, dash='dash')))
            fig_acc.update_layout(
                title="Model Accuracy Convergence",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Epoch",
                yaxis_title="Accuracy"
            )
            st.plotly_chart(fig_acc, use_container_width=True)
            
            # Plot Loss Curves
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=hist["Epoch"], y=hist["Train Loss"], name="Training Loss", line=dict(color="#f59e0b", width=3)))
            fig_loss.add_trace(go.Scatter(x=hist["Epoch"], y=hist["Val Loss"], name="Validation Loss", line=dict(color="#ef4444", width=2.5, dash='dash')))
            fig_loss.update_layout(
                title="Cross-Entropy Loss Minimization",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Epoch",
                yaxis_title="Loss Value"
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("Train the network to generate custom validation loss and accuracy curves automatically.")

    # CONFUSION HEATMAP SECTION
    st.write("---")
    st.subheader("🎯 Test Dataset Performance Matrix")
    
    if st.session_state["is_trained"]:
        with st.spinner("Calculating test classification matrices..."):
            _, _, x_test, y_test = load_or_generate_mnist_dataset(30)
            predictions = st.session_state["model"].predict(x_test, verbose=0)
            predicted_labels = np.argmax(predictions, axis=1)
            
            # Calculate 10x10 matrix
            matrix = np.zeros((10, 10), dtype=int)
            for true_label, predicted_label in zip(y_test, predicted_labels):
                matrix[true_label, predicted_label] += 1
                
            fig_heatmap = px.imshow(
                matrix,
                labels=dict(x="Predicted Digit Label", y="True Digit Label", color="Sample Count"),
                x=[str(i) for i in range(10)],
                y=[str(i) for i in range(10)],
                text_auto=True,
                color_continuous_scale="Purples"
            )
            fig_heatmap.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            col_heatmap, col_explanation = st.columns([2, 1.5], gap="large")
            with col_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
            with col_explanation:
                diagonal_correct = np.sum(np.diag(matrix))
                total_samples = np.sum(matrix)
                accuracy_rate = (diagonal_correct / total_samples) * 100 if total_samples > 0 else 0
                
                st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
                st.metric("Test Categorization Accuracy", f"{accuracy_rate:.1f}%")
                
                st.markdown("""
                **Interpreting the Performance Heatmap:**
                - **Diagonal Elements (Purple Squares):** Show correct classification instances. The brighter the color, the more times the network successfully matched the drawing.
                - **Off-Diagonal Elements:** Show error clusters (e.g. if the model frequently mistakes a drawn `3` as an `8`). Use this to study structural patterns!
                """)
    else:
        st.info("Train your Neural Network above to display the validation heatmap.")

# ──────────────────────────────────────────────────────────────────────────────
#  TAB 3: GOOGLE SHEETS LIVE DATABASE EXPLORER
# ──────────────────────────────────────────────────────────────────────────────
with tab_database:
    st.subheader("📋 Dataset Explorer (Live Google Sheet Sync)")
    st.markdown("<p style='color:#94a3b8; font-size:13px;'>Browse and audit handdrawn records directly fetched from the linked Google Sheets database spreadsheet in real-time.</p>", unsafe_allow_html=True)
    
    if not sheets_client:
        st.warning("Google Sheets connection is offline. Connect using the sidebar configuration.")
    elif not spreadsheet_url:
        st.warning("Please configure your Spreadsheet ID / URL in the sidebar first.")
    else:
        records_df = load_gsheet_records(spreadsheet_url, sheet_name)
        
        if len(records_df) > 0:
            m_total = len(records_df)
            m_users = records_df["username"].nunique() if "username" in records_df.columns else 0
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(f'<div class="metric-card"><div class="value">{m_total}</div><div class="label">Total Gathered Samples</div></div>', unsafe_allow_html=True)
            with col_m2:
                st.markdown(f'<div class="metric-card"><div class="value">{m_users}</div><div class="label">Distinct Operators Contributing</div></div>', unsafe_allow_html=True)
                
            st.write("")
            
            # Live Data Grid
            st.markdown("### 🔍 Live Data Table")
            st.dataframe(
                records_df.sort_values(by="id", ascending=False) if "id" in records_df.columns else records_df, 
                use_container_width=True, 
                hide_index=True
            )
            
            # Draw visual distribution of gathered classes
            if "label" in records_df.columns:
                st.write("")
                st.markdown("### 📊 Distribution of Handdrawn Digits")
                
                dist_counts = records_df["label"].value_counts().reset_index()
                dist_counts.columns = ["Label", "Count"]
                dist_counts = dist_counts.sort_values(by="Label")
                
                fig_dist = px.bar(
                    dist_counts, 
                    x="Label", 
                    y="Count", 
                    labels={"Label": "Digit Label", "Count": "Number of Samples"},
                    color="Count",
                    color_continuous_scale="Purples"
                )
                fig_dist.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Spreadsheet worksheet is currently empty or contains no valid rows. Start drawing and push records to begin populate the tables!")

    st.markdown("---")
    st.subheader("📝 Local Session Event Log")
    if st.session_state["synced_count"] > 0:
        st.write(f"**Total pushed drawings in this browser instance:** `{st.session_state['synced_count']}`")
        st.code("\n".join(st.session_state["sync_logs"]))
    else:
        st.info("Pushed entries and sync actions will log here in real-time.")