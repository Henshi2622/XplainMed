import streamlit as st
import pathlib
import base64

# Load CSS
def load_css(file_path):
    with open(file_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Convert image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

css_path = pathlib.Path("home.css")
load_css(css_path)


col1, col2 = st.columns(2, gap="small", vertical_alignment="center")
with col1:
    # Convert image to base64 and embed it
    img_base64 = get_base64_image("assets/Robot.png")
    st.markdown(f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_base64}" width="300" 
             style="max-width: 100%; height: auto; pointer-events: none; cursor: default;">
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <h1 style="text-align:center; font-size: 2.8rem; margin-bottom: 0.5rem;">
         XplainMed
    </h1>
    <h3 style="text-align:center; color: #6c63ff; margin-top: 0;">
        Understand Your Medical Report with Ease
    </h3>
    <p style="font-size: 1.1rem; text-align: center; max-width: 800px; margin: auto;">
        Welcome to <strong>XplainMed</strong>, your trusted AI companion in decoding medical reports.
        Confused by medical jargon? Not sure what that diagnosis means?
        <br><br>
        Let us simplify complex terms into human language and even visualize your report with helpful diagrams.
        <br><br>
        <em>Your health, explained clearly.</em>
    </p>
""", unsafe_allow_html=True)
# Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    



# Inject styling (optional gradient or center alignment)
