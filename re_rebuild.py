import os
import streamlit as st
from typing import Dict, Optional
from groq import Groq
from PIL import Image
import PyPDF2
from docx import Document

# Streamlit page configuration
st.set_page_config(layout="wide", page_title="AI Resume Rebuilder", initial_sidebar_state="expanded")

# Supported models
SUPPORTED_MODELS: Dict[str, str] = {
    "Llama 3.2 1B (Preview)": "llama-3.2-1b-preview",
    "Llama 3 70B": "llama3-70b-8192",
    "Llama 3 8B": "llama3-8b-8192",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "Gemma 2 9B": "gemma2-9b-it",
    "Llama 3.2 11B Vision (Preview)": "llama-3.2-11b-vision-preview",
    "Llama 3.2 11B Text (Preview)": "llama-3.2-11b-text-preview",
    "Llama 3.1 8B Instant (Text-Only Workloads)": "llama-3.1-8b-instant",
    "Llama 3.2 90B Vision (Preview)": "llama-3.2-90b-vision-preview",  # New addition
    "LLaVA v1.5 7B Vision (Deprecated)": "llava-v1.5-7b-4096-preview",  # Deprecated but included
}

MAX_TOKENS: int = 1000

# Initialize Groq client with API key
@st.cache_resource
def get_groq_client() -> Optional[Groq]:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables. Please set it and restart the app.")
        return None
    return Groq(api_key=groq_api_key)

client = get_groq_client()

# Sidebar - Model Configuration
st.sidebar.image("icon.PNG", width=100)
st.sidebar.image("p2.png")
st.sidebar.title("Model Configuration")
selected_model = st.sidebar.selectbox("Choose an AI Model", list(SUPPORTED_MODELS.keys()))

# Sidebar - Temperature Slider
st.sidebar.subheader("Temperature")
temperature = st.sidebar.slider("Set temperature for response variability:", min_value=0.0, max_value=1.0, value=0.7)

# Sidebar - Optional Features
st.sidebar.subheader("Optional Features")
ats_optimization = st.sidebar.checkbox("ATS Optimization")
cms_optimization = st.sidebar.checkbox("CMS Optimization")
use_kiss_principle = st.sidebar.checkbox("Simplification using KISS principle")
achievements_star = st.sidebar.checkbox("Achievements in STAR format")
seo_keywords = st.sidebar.checkbox("Include SEO Keywords")
personal_branding = st.sidebar.checkbox("Personal Branding Statement")

# Sidebar - File Upload
st.sidebar.subheader("Upload Resume")
resume_file = st.sidebar.file_uploader("Upload PDF or Word Document", type=["pdf", "docx"])

# Initialize session state for outputs
if 'rebuilt_resume' not in st.session_state:
    st.session_state.rebuilt_resume = None

# Function to extract text from uploaded file
def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(uploaded_file)
        text = "".join(page.extract_text() for page in reader.pages)
        return text
    elif uploaded_file.name.endswith(".docx"):
        doc = Document(uploaded_file)
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        return text
    else:
        st.error("Unsupported file type. Please upload a PDF or Word document.")
        return ""

# Function to get response from Groq API
def get_groq_response(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=SUPPORTED_MODELS[selected_model],
            messages=[
                {"role": "system", "content": "You are an AI assistant that generates outputs based on inputs."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error: {e}")
        return ""

# Main Content
st.title("AI Resume Rebuilder")

# Resume rebuilding
st.subheader("Resume Rebuilder")
default_prompt = f"""
Using the following input, rebuild and optimize the resume to target modern hiring practices and specific job descriptions:
"""

if resume_file:
    resume_text = extract_text_from_file(resume_file)
    if st.sidebar.button("Rebuild Resume"):
        optional_features = []
        if ats_optimization:
            optional_features.append("Optimize for ATS.")
        if cms_optimization:
            optional_features.append("Optimize for CMS.")
        if use_kiss_principle:
            optional_features.append("Simplify content using KISS principle.")
        if achievements_star:
            optional_features.append("Present achievements in STAR format.")
        if seo_keywords:
            optional_features.append("Include industry-specific SEO keywords.")
        if personal_branding:
            optional_features.append("Add a strong personal branding statement.")

        optional_features_text = "\n".join(optional_features)
        resume_prompt = f"{default_prompt}\n{optional_features_text}\n\n{resume_text}"
        if client:
            with st.spinner("Rebuilding your resume..."):
                st.session_state.rebuilt_resume = get_groq_response(resume_prompt)
        else:
            st.error("Groq client not initialized.")

# Display Rebuilt Resume
if st.session_state.rebuilt_resume:
    st.subheader("Rebuilt Resume")
    st.text_area("Optimized Resume:", st.session_state.rebuilt_resume, height=400)
    st.download_button(
        label="Download Rebuilt Resume",
        data=st.session_state.rebuilt_resume,
        file_name="rebuilt_resume.txt",
        mime="text/plain"
    )

st.info("build by dw- Upload your resume file or enter details manually to rebuild it using AI-powered optimization.")
