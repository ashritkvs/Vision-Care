import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import numpy as np

# Set page config
st.set_page_config(page_title="VisionCare AI", layout="centered")

# Title
st.title("👁️ VisionCare AI: Diabetic Retinopathy Detection")

# Load model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load("models/best_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Label mapping
class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(image).unsqueeze(0)

# Upload image
uploaded_file = st.file_uploader("Upload a retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0).numpy()
        pred_class = np.argmax(probs)

    # Show prediction
    st.subheader(f"Prediction: **{class_names[pred_class]}**")
    st.bar_chart(probs)
