import os, json
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
import plotly.graph_objects as go

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "saved_models/best_model_full.pth"
NAMES_PATH = "saved_models/class_names.json"
RESULTS    = "results"
IMG_SIZE   = 224

st.set_page_config(page_title="Fish Classifier", page_icon="🐟", layout="wide")

@st.cache_resource
def load_model():
    model = torch.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    return model

@st.cache_data
def load_names():
    with open(NAMES_PATH) as f: return json.load(f)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

with st.sidebar:
    st.title("🐟 Fish Classifier")
    st.markdown(f"**Device:** {DEVICE}")
    info_path = "saved_models/best_model_name.json"
    if os.path.exists(info_path):
        with open(info_path) as f: info = json.load(f)
        st.success(f"Best Model: {info['best_model']}")
        st.metric("Val Accuracy", f"{info['accuracy']*100:.2f}%")
    page = st.radio("Navigate", ["🔍 Predict", "📊 Comparison", "📈 History"])

model, class_names = load_model(), load_names()

if page == "🔍 Predict":
    st.title("🐟 Fish Species Classifier")
    uploaded = st.file_uploader("Upload a fish image", type=["jpg","jpeg","png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)
        with col2:
            inp  = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out   = model(inp)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            st.success(f"**Prediction: {class_names[idx]}**")
            st.metric("Confidence", f"{probs[idx]*100:.2f}%")
            st.progress(float(probs[idx]))
            fig = go.Figure(go.Bar(
                x=list(probs), y=class_names, orientation="h",
                marker_color=["#2ecc71" if i==idx else "#3498db" for i in range(len(class_names))],
                text=[f"{v*100:.1f}%" for v in probs], textposition="outside"
            ))
            fig.update_layout(title="Class Probabilities", xaxis_range=[0,1.2],
                              height=max(400, len(class_names)*35))
            st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Comparison":
    st.title("📊 Model Comparison")
    rpt_path = os.path.join(RESULTS, "comparison_report.json")
    if os.path.exists(rpt_path):
        df = pd.read_json(rpt_path)
        st.dataframe(df.style.highlight_max(subset=["Accuracy","Precision","Recall","F1-Score"],
                     color="lightgreen"), use_container_width=True)
        chart = os.path.join(RESULTS, "model_comparison_chart.png")
        if os.path.exists(chart): st.image(chart, use_container_width=True)
    else:
        st.warning("Run the notebook first to generate results.")

elif page == "📈 History":
    st.title("📈 Training History")
    all_hist = os.path.join(RESULTS, "all_models_history.png")
    if os.path.exists(all_hist):
        st.image(all_hist, caption="All Models", use_container_width=True)
    for hf in sorted(f for f in os.listdir(RESULTS) if f.endswith("_history.png") and "all" not in f):
        st.markdown(f"**{hf.replace('_history.png', '')}**")
        st.image(os.path.join(RESULTS, hf), use_container_width=True)