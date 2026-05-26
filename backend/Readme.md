# Multi-PDF ChatApp AI Agent 🤖📚

An enterprise-grade, fully local **Advanced RAG (Retrieval-Augmented Generation)** pipeline that transforms dense corporate PDF reports into professional, structured presentation slide decks. Built with **FastAPI**, **Streamlit**, **LangChain**, and **FAISS**, this application completely eliminates context fragmentation and formatting breaks using a multi-stage retrieval and type-safe schema validation engine.

---

## 🚀 Advanced RAG Architecture & Features

Unlike standard tutorial chatbots that copy text blindly, this agent uses a production-grade **Two-Stage Retrieval & Guardrail** workflow:

* **Day 3: Hierarchical Parent-Child Indexing** – Solves context fragmentation. The system embeds small, granular 400-character child nodes into FAISS for sharp mathematical search precision, while mapping them to 2,000-character parent text blocks in metadata to feed the LLM broad, unbroken sentences.
* **Day 4: Cross-Encoder Re-ranking** – Boosts data priority. Uses the local `ms-marco-MiniLM-L-6-v2` Cross-Encoder model to simultaneously evaluate prompt-passage relationships, bubbles up critical metrics, and filters out irrelevant vector noise.
* **Day 5: Pydantic Structural Guardrails** – Prevents application crashes. Binds a rigid JSON data blueprint directly to the Gemini LLM using `.with_structured_output()`. The API is mathematically constrained to return type-safe objects, completely removing brittle text regex parsing scripts.
* **Day 6: Decoupled API Gateway** – A stateless backend powered by FastAPI that processes matrix math locally on your CPU and streams presentation data back over HTTP via memory-efficient binary byte streams.

---

## 🛠️ Tech Stack

* **Frontend UI:** Streamlit
* **API Gateway Backend:** FastAPI + Uvicorn
* **Orchestration Framework:** LangChain Expression Language (LCEL)
* **Vector Database Library:** FAISS (Local File Persistence)
* **Dense Embeddings Model:** Hugging Face Transformers (`all-MiniLM-L6-v2`)
* **Reranker Model:** Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
* **Generation Core:** Google Gemini Pro (`with_structured_output`)
* **Data Validation:** Pydantic v2
* **Presentation Compiler:** Python-PPTX

---

## 🎯 How the Production Pipeline Works

1. **Ingest & Parse:** PDF text is extracted and split into a dual-layer hierarchy (Broad Parents ──► Tiny Overlapping Children).
2. **Metadata Inlining:** Children are saved to the local FAISS index carrying their full raw parent text blocks inside their metadata dictionaries.
3. **Stage 1 Retrieval:** FAISS executes a fast dense vector lookup to catch a wide pool of candidate child chunks ($k=15$).
4. **Stage 2 Rerank:** The local Cross-Encoder scores the candidates. The top 5 high-priority matches are kept, and their child text is swapped for their broad parent context blocks.
5. **Schema Synthesis:** The pristine parent blocks are passed to the Gemini API, which synthesizes the content directly into a validated Pydantic object model structure.
6. **Byte-Streaming:** The PowerPoint compiler maps object properties natively to layout canvas shapes, streaming a raw binary PPTX payload back to the user client interface.

---

## 💻 Local Installation & Setup

### 1. Clone and Install Dependencies
```bash
git clone [https://github.com/your-username/multi-pdf-slide-agent.git](https://github.com/your-username/multi-pdf-slide-agent.git)
cd multi-pdf-slide-agent
pip install -r requirements.txt