import os
import json
import re
import time
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from duckduckgo_search import DDGS

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA

import pathlib

# Load environment variables
load_dotenv()

# Load CSS
def load_css(file_path):
    with open(file_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = pathlib.Path("report.css")
load_css(css_path)


st.title("🩺Understand Your Medical Report")

pdf_file = st.file_uploader("📄 Upload your Medical Report (PDF)", type=["pdf"], key="report_uploader")
generate_button = st.button("🔍 Generate Explanation")

# PDF Text Extraction
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    return "".join([page.extract_text() or "" for page in reader.pages])

# Image search (return 3 images)
def search_multiple_medical_images(diagnosis, explanation):
    focused_prompt = f"{diagnosis} medical diagram"
    try:
        with DDGS() as ddgs:
            time.sleep(2)
            results = ddgs.images(focused_prompt, max_results=3)
            return [result["image"] for result in results]
    except Exception as e:
        print("Image search failed:", e)
    return []

# Embedding model
embedding_model = AzureOpenAIEmbeddings(
    openai_api_key=os.getenv("EMBEDDING_AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("EMBEDDING_AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("EMBEDDING_AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("EMBEDDING_AZURE_OPENAI_API_VERSION"),
    chunk_size=1000
)

# Main logic
if pdf_file and generate_button:
    with st.spinner("⚙️ Extracting and interpreting your report..."):
        raw_text = extract_text_from_pdf(pdf_file)
        if not raw_text.strip():
            st.error("No text found in the uploaded PDF.")
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_text(raw_text)
            documents = [Document(page_content=chunk) for chunk in chunks]
            vector_db = FAISS.from_documents(documents, embedding_model)
            retriever = vector_db.as_retriever()

            llm = AzureChatOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                temperature=0.7,
                max_tokens=1500,
            )

            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

            user_prompt = (
                "From this medical report, extract only the main diagnosis in 2-3 words. "
                "Return the diagnosis as JSON like this:\n"
                '{"diagnosis": "..."}\n\n'
                "After the JSON, write a clear explanation in simple layman language that anyone without medical knowledge can understand. "
                "First, explain what the patient is suffering from in general terms. "
                "Then, describe *which part of the body* or *which area* is affected by the diagnosis. "
                "After that, mention the *possible causes* of this condition in simple words. "
                "Finally, list some *precautions* or lifestyle suggestions the patient should follow to manage or avoid complications. "
                "Avoid medical jargon; keep it friendly and easy to understand."
            )

            try:
                response = qa_chain.invoke(user_prompt)
                result_text = response["result"]
                json_match = re.search(r'{.*}', result_text, re.DOTALL)

                if json_match:
                    diagnosis_block = json.loads(json_match.group())
                    diagnosis = diagnosis_block.get("diagnosis", "").strip()
                    explanation = result_text.replace(json_match.group(), "").strip()
                else:
                    diagnosis = ""
                    explanation = result_text.strip()
            except Exception:
                st.warning("⚠️ Could not extract structured data. Showing full response.")
                diagnosis = ""
                explanation = response.get("result", "")

            # Save to session for later use
            st.session_state["diagnosis"] = diagnosis
            st.session_state["explanation"] = explanation
            st.session_state["image_urls"] = search_multiple_medical_images(diagnosis, explanation)

            # Display only explanation here
            st.subheader("Layman Explanation")
            if diagnosis:
                st.markdown(f"**🧠 Diagnosis:** `{diagnosis}`")
            st.markdown(explanation)
