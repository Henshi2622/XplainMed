import streamlit as st
import pathlib

#  laod the CSS file
def load_css(file_path):
    with open (file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the CSS file
css_path = pathlib.Path("style.css")
load_css(css_path)


# naviagtion bar 
tab1, tab2, tab3 = st.tabs(["Home", "Report", "Images"])

with tab1:
    # Display content from Home.py
    st.empty()  # Optional: placeholder for clarity
    try:
        exec(open("Home.py").read())
    except Exception as e:
        st.error(f"Error loading Home.py: {e}")

with tab2:
    # Display content from Report.py
    try:
        exec(open("Report.py", encoding="utf-8").read())
    except Exception as e:
        st.error(f"Error loading Report.py: {e}")

with tab3:
    # Display content from Images.py
    try:
        exec(open("Images.py", encoding="utf-8").read())
    except Exception as e:
        st.error(f"Error loading Images.py: {e}")
# with tab4:
#     # Display content from Doctor_Search.py
#     try:
#         exec(open("Doctor_Search.py").read())
#     except Exception as e:
#         st.error(f"Error loading Doctor_Search.py: {e}")
