# upload_image.py
import os
import streamlit as st
import torch
from analysis import analyze_medical_image
from diffusers import DiffusionPipeline

# Cache Stable Diffusion pipeline so it loads only once
@st.cache_resource
def load_pipeline():
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

def generate_simplified_image(prompt):
    """Generate patient-friendly simplified diagram using Stable Diffusion."""
    pipe = load_pipeline()
    image = pipe(prompt, height=512, width=512).images[0]  # keep stable size
    return image


# ---------------- Streamlit UI ----------------
st.title("🩺 Medical Image Analysis 🔬")

st.markdown(
    """
    Upload a medical image (**X-ray, MRI, CT, Ultrasound, etc.**) and our AI-powered system will:
    
    - Analyze it and provide a **detailed structured report** 📋  
    - Explain findings in **patient-friendly language** 🧑‍🤝‍🧑  
    - Generate a **simplified diagram** for easy understanding 🖼️
    """
)

# File uploader in main area
st.subheader("📤 Upload Your Medical Image")
uploaded_file = st.file_uploader(
    "Choose a medical image file",
    type=["jpg", "jpeg", "png", "bmp", "gif"]
)

# Process when file is uploaded
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=300)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing the image... Please wait ⏳"):
            # Save uploaded file temporarily
            ext = uploaded_file.type.split("/")[-1]
            image_path = f"temp_image.{ext}"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Run AI analysis
            report = analyze_medical_image(image_path)

            # Show report
            st.subheader("📋 AI Analysis Report")
            st.markdown(report, unsafe_allow_html=True)

        # Generate simplified diagram
        st.subheader("🖼️ Patient-Friendly Visual Explanation")
        with st.spinner("Generating simplified diagram..."):
            diagram_prompt = (
                "Create a simplified, labeled medical diagram for patient understanding based on this explanation: "
                + report
            )
            try:
                diagram_image = generate_simplified_image(diagram_prompt)
                st.image(diagram_image, caption="AI-Generated Simplified Diagram")
            except Exception as e:
                st.error(f"⚠️ Diagram generation failed: {e}")

        # Clean up temp file
        if os.path.exists(image_path):
            os.remove(image_path)
else:
    st.info("👉 Please upload a medical image to start analysis.")
