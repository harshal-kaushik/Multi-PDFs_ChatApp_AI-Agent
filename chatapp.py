import streamlit as st
<<<<<<< HEAD

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

=======
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline

# ✅ Cache embeddings (important)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Cache LLM
@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_length=512,
        temperature=0.3
    )
    return HuggingFacePipeline(pipeline=pipe)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # ✅ FIXED
        chunk_overlap=200
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = load_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_qa_chain(vector_store):
    llm = load_llm()

    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, say:
    "answer is not available in the context"

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    from langchain.chains import RetrievalQA

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    return chain

def main():
    st.set_page_config("Multi PDF Chatbot", page_icon="📚")
    st.header("Multi-PDF Chat Agent 🤖")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    user_question = st.text_input("Ask a question:")

    if user_question and st.session_state.vector_store:
        chain = get_qa_chain(st.session_state.vector_store)
        response = chain.invoke({"query": user_question})
        st.write("Reply:", response["result"])

    with st.sidebar:
        st.title("Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                st.session_state.vector_store = get_vector_store(chunks)
                st.success("Ready!")
>>>>>>> origin/main

if __name__ == "__main__":
    main()
