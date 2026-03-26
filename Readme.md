# Multi-PDF-s 📚 ChatApp AI Agent 🤖

Meet MultiPDF Chat AI App! 🚀 Chat seamlessly with Multiple PDFs using LangChain, Hugging Face Models & FAISS Vector DB with Streamlit deployment. Get instant, accurate responses using open-source LLMs running locally. 📚💬

---

## 📝 Description

The Multi-PDF Chat Agent is a Streamlit-based web application that allows users to interact with multiple PDF documents through a chatbot interface. It uses a Retrieval-Augmented Generation (RAG) pipeline powered by Hugging Face models running locally.

---

## 🎯 How It Works

1. **PDF Loading** – Extracts text from uploaded PDFs  
2. **Text Chunking** – Splits text into smaller chunks  
3. **Embeddings** – Converts text into vectors using Hugging Face models  
4. **Similarity Search (FAISS)** – Finds relevant chunks  
5. **Response Generation** – Generates answers using a local LLM  

---

## 🎯 Key Features

- 📄 Multi-PDF Question Answering  
- 🤖 Fully Local LLM (No API required)  
- ⚡ Fast retrieval using FAISS  
- 🧠 Hugging Face open-source models  
- 🌐 Streamlit-based UI  
- 🔐 Better data privacy (runs locally)  

---

## ▶️ Run the App

```bash
streamlit run app.py