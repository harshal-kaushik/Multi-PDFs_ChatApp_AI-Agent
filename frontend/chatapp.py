import streamlit as st
import requests

# 1. Page Global Setup
st.set_page_config(
    page_title="Doc-to-Decks AI Workspace",
    page_icon="📊",
    layout="wide"
)

# Core communication pointer mapping to your minimal FastAPI network port
BACKEND_URL = "http://127.0.0.1:8000"

# 2. Sidebar Workspace: Document Ingestion & Storage Management
with st.sidebar:
    st.header("📥 Knowledge Base Management")
    st.markdown("Upload corporate or business PDFs to update the local FAISS index.")

    uploaded_files = st.file_uploader(
        "Select Business PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    # Process Button Action
    if st.button("🚀 Process & Index Files", use_container_width=True):
        if not uploaded_files:
            st.sidebar.warning("Please select at least one PDF file.")
        else:
            for uploaded_file in uploaded_files:
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    # Extract file stream data payload to transfer across ports
                    file_payload = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
                    }
                    try:
                        response = requests.post(f"{BACKEND_URL}/ingest", files=file_payload)
                        if response.status_code == 200:
                            st.sidebar.success(f"Indexed: {uploaded_file.name}")
                        else:
                            error_msg = response.json().get("detail", "Unknown server contraction.")
                            st.sidebar.error(f"Failed {uploaded_file.name}: {error_msg}")
                    except requests.exceptions.ConnectionError:
                        st.sidebar.error("Backend offline! Please run your FastAPI server on port 8000.")
                        break

    st.markdown("---")
    st.markdown("### System Status")
    try:
        # Fast non-blocking verification pin hitting our core system root
        check = requests.get(BACKEND_URL, timeout=2)
        if check.status_code == 200:
            st.success("● FastAPI Connected")
        else:
            st.warning("● API Status Irregular")
    except:
        st.error("○ FastAPI Disconnected")

# 3. Main Operational Workspace: Deck Generation & Output Delivery
st.title("📊 Doc-to-Decks: Structured RAG Slide Builder")
st.markdown(
    "This pipeline extracts key insights across your corporate documents and structures them "
    "into an executive presentation layout without exposing your system to loose conversational chatbot fluff."
)

st.markdown("### 🎯 Define Presentation Requirements")
user_prompt = st.text_area(
    "What should this business presentation cover?",
    placeholder="e.g., Extract our core financial metrics, growth drivers, and market expansion strategies for a 5-slide investment deck.",
    height=150,
    help="Be specific about the parameters, pillars, or sections you want to highlight from the ingested files."
)

# Compilation Execution Trigger
if st.button("🎬 Synthesize & Compile Presentation Deck", use_container_width=True):
    if not user_prompt.strip():
        st.warning("Please fill out your presentation requirements before compiling.")
    else:
        # Create an operational loading layer block while waiting for LLM + binary generation
        with st.spinner(
                "Executing Parent-Child lookup, running schema validation, and assembling presentation layout..."):
            form_payload = {"topic": user_prompt}
            try:
                response = requests.post(f"{BACKEND_URL}/generate-deck", data=form_payload)

                if response.status_code == 200:
                    st.success("🎉 Presentation deck compiled successfully!")

                    # Provide the generated binary memory buffer as a native download button link
                    st.download_button(
                        label="📥 Download Native Widescreen Presentation (.pptx)",
                        data=response.content,
                        file_name="generated_executive_presentation.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        use_container_width=True
                    )
                else:
                    # Parse out errors if the parsing structure fails validation tags
                    error_detail = response.json().get('detail', 'Generation collapse.')
                    st.error(f"Compilation Failed: {error_detail}")
                    st.info("💡 Tip: Ensure your document text is extractable and contains relevant business metrics.")

            except requests.exceptions.ConnectionError:
                st.error("Network Exception: Could not stream parameters to FastAPI. Verify port 8000 is active.")