import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np

# --- IMPORTANT ---
# Make sure 'model.py' is in the same directory as this script
from model import Encoder # Importing your Encoder class

# --- Page Configuration ---
st.set_page_config(
    page_title="Few-Shot Learning Demo",
    page_icon="🚀",
    layout="wide"
)

# --- Helper Functions ---

# Function to load the model (cached so it only runs once)
@st.cache_resource
def load_model(model_path):
    """Loads the trained Encoder model."""
    model = Encoder()
    # Load the model on CPU, you can change to 'cuda' if you have a GPU
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Image transformation
transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(image):
    """Converts a user-uploaded image to a tensor."""
    img = Image.open(image).convert('RGB')
    return transform(img).unsqueeze(0) # Add batch dimension

def calculate_prototypes(support_embeddings, support_labels):
    """Calculates class prototypes from support embeddings."""
    unique_labels = sorted(list(set(support_labels)))
    prototypes = []
    for label in unique_labels:
        # Get embeddings for the current class
        class_embeddings = support_embeddings[np.array(support_labels) == label]
        # Calculate the mean to get the prototype
        prototype = class_embeddings.mean(dim=0)
        prototypes.append(prototype)
    return torch.stack(prototypes)

# --- Main App Interface ---

st.title("🚀 Few-Shot Image Classification Demo")
st.write(
    "This app demonstrates **rapid adaptation**. Teach the AI to recognize new image "
    "classes by providing just a few examples ('shots')."
)

# --- Sidebar for Instructions and Model Loading ---
with st.sidebar:
    st.header("How it Works")
    st.info(
        "1. **Define Classes**: Create up to 5 new classes.\n"
        "2. **Upload Support Images**: For each class, upload 1-5 example images (the 'support set').\n"
        "3. **Upload Query Image**: Upload a new image you want to classify.\n"
        "4. **Predict**: The model will classify the new image based on the examples you provided."
    )
    # NOTE: Update the path to your actual model file
    model_path = 'best_model.pth' 
    model = load_model(model_path)
    st.success("Model loaded successfully!")

# --- UI for Class Definition and Image Uploads ---
st.header("1. Define Your Classes and Upload Support Images")

# Use columns for a clean layout
num_classes = st.slider("Select number of classes (2-5):", 2, 5, 3)
cols = st.columns(num_classes)

support_images = []
support_labels = []

for i, col in enumerate(cols):
    with col:
        class_name = st.text_input(f"Class {i+1} Name", f"Class {i+1}", key=f"class_{i}")
        uploaded_files = st.file_uploader(f"Upload examples for {class_name}", 
                                          type=['png', 'jpg', 'jpeg'], 
                                          accept_multiple_files=True, 
                                          key=f"uploader_{i}")
        
        for file in uploaded_files:
            support_images.append(file)
            support_labels.append(i) # Assign integer label
            st.image(file, width=100)

# --- UI for Query Image Upload ---
st.header("2. Upload an Image to Classify")
query_image_file = st.file_uploader("Upload your 'query' image here", type=['png', 'jpg', 'jpeg'])

if query_image_file:
    st.image(query_image_file, caption="Query Image", width=200)

# --- Prediction Logic ---
if st.button("🚀 Classify Image", type="primary", use_container_width=True):
    if not support_images:
        st.error("Please upload at least one support image.")
    elif not query_image_file:
        st.error("Please upload a query image.")
    else:
        with st.spinner("Analyzing..."):
            # Process all support images and get embeddings
            support_tensors = torch.cat([process_image(img) for img in support_images])
            support_embeddings = model(support_tensors)
            
            # Calculate class prototypes
            prototypes = calculate_prototypes(support_embeddings, support_labels)
            
            # Process query image and get embedding
            query_tensor = process_image(query_image_file)
            query_embedding = model(query_tensor)
            
            # Calculate distances and predict
            distances = torch.cdist(query_embedding, prototypes)
            prediction = torch.argmin(distances, dim=1).item()
            
            # Get class names from text input fields
            class_names = [st.session_state[f"class_{i}"] for i in range(num_classes)]
            predicted_class_name = class_names[prediction]
            
            st.success(f"### Predicted Class: **{predicted_class_name}**")