# app/streamlit_app.py

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys

# -------------------------
# Project root
# -------------------------
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))

# -------------------------
# Imports
# -------------------------
from models.cnn_feature_extractor import CNNFeatureExtractor
from models.gnn_model import GNNClassifier
from models.utils import build_graph

# -------------------------
# Streamlit UI config
# -------------------------
st.set_page_config(
    page_title="FurSight 🐾",
    page_icon="🐱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.image("https://img.icons8.com/fluency/48/cat.png", use_container_width=False)
st.sidebar.title("FurSight")
st.sidebar.write("Detect cat & dog skin problems using AI!")
st.sidebar.info("💡 Tip: Upload a clear image showing the affected area, not necessarily the whole pet.")

# -------------------------
# Main UI
# -------------------------
st.markdown(
    """
    <div style="text-align: center; background-color:#F5F5F5; padding: 20px; border-radius:10px">
        <h1 style="color:#4B0082; font-family:Arial, sans-serif;">
            🐾 FurSight: Cat & Dog Disease Detection 🐶
        </h1>
        <p style="color:#333333; font-size:18px;">
            Upload an image of your pet to predict its skin disease quickly!
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("---")

st.info("💡 Tip: Make sure the affected area or skin problem is clearly visible in the image!")

# -------------------------
# Device & labels
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAP = {
    0: "demodicosis",
    1: "dermatitis",
    2: "healthy"
}

# -------------------------
# Load models
# -------------------------
cnn = CNNFeatureExtractor(output_dim=128).to(DEVICE)
gnn = GNNClassifier(input_dim=128, hidden_dim=64, num_classes=len(LABEL_MAP)).to(DEVICE)

cnn.load_state_dict(torch.load(PROJECT_ROOT / "models/cnn.pth", map_location=DEVICE))
gnn.load_state_dict(torch.load(PROJECT_ROOT / "models/gnn.pth", map_location=DEVICE))
cnn.eval()
gnn.eval()

# -------------------------
# Image upload & prediction
# -------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        # CNN feature extraction
        with torch.no_grad():
            feature = cnn(img_tensor)
            graph = build_graph(feature, edges=None)
            graph = graph.to(DEVICE)
            batch_index = torch.zeros(graph.x.size(0), dtype=torch.long).to(DEVICE)

            # GNN prediction
            pred = gnn(graph.x, graph.edge_index, batch_index)
            pred_idx = torch.argmax(pred, dim=1).item()
            pred_name = LABEL_MAP.get(pred_idx, "Unknown")

        st.success(f"Predicted class: **{pred_name}**")

    except Exception as e:
        st.error(f"Error processing image: {e}")

# -------------------------
# Footer
# -------------------------
st.markdown(
    """
    <div style="text-align:center; margin-top:20px; color:#888;">
        Developed with ❤️ by Myriam
    </div>
    """,
    unsafe_allow_html=True
)
