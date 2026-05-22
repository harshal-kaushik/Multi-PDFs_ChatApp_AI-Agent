import streamlit as st

from pdf_handler import get_pdf_text
from vector_store import (
    get_text_chunks,
    create_vector_store
)

from runnable_chain import ask_question


def main():
    st.set_page_config(
        page_title="Multi PDF RAG Chatbot"
    )

    st.header(
        "Multi-PDF Chatbot 📚🤖"
    )

    # Session state to track PDF processing
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

    with st.sidebar:
        st.title("PDF Upload Section")

        pdf_docs = st.file_uploader(
            "Upload your PDF files",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):

            if not pdf_docs:
                st.warning("Please upload at least one PDF first.")

            else:
                with st.spinner("Processing PDFs..."):

                    # Step 1 → Extract text
                    raw_text = get_pdf_text(pdf_docs)

                    # Step 2 → Create chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Step 3 → Create vector store
                    create_vector_store(text_chunks)

                    # Step 4 → Mark as processed
                    st.session_state.pdf_processed = True

                    st.success(
                        "PDFs processed successfully. Now you can ask questions."
                    )

    # Only allow questions after PDF upload + processing
    if st.session_state.pdf_processed:

        user_question = st.text_input(
            "Ask a question from your uploaded PDFs"
        )

        if user_question:
            with st.spinner("Generating answer..."):

                answer = ask_question(
                    user_question
                )

                st.write("### Reply:")
                st.write(answer)

    else:
        st.info(
            "Please upload and process PDFs first to start asking questions."
        )


if __name__ == "__main__":
    main()