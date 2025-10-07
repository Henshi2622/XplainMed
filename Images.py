import streamlit as st

st.title("Medical Illustrations")

image_urls = st.session_state.get("image_urls", [])
diagnosis = st.session_state.get("diagnosis")

if image_urls and diagnosis:
    for i, url in enumerate(image_urls, start=1):
        st.image(url, caption=f"Related to {diagnosis} - Image {i}", use_container_width=True)
        st.code(url, language="text")
elif diagnosis:
    st.warning("Images not available yet. Try regenerating from the Upload_Report tab.")
else:
    st.markdown("""
    <div style="
        background-color: #e0f3ff;
        border-left: 6px solid #2196F3;
        padding: 1rem;
        border-radius: 0.5rem;
        font-size: 1rem;
        color: #0b3954;
        margin: 1rem 0;
    ">
        💡 <strong>Info:</strong> Please upload and generate a report first.
    </div>
""", unsafe_allow_html=True)