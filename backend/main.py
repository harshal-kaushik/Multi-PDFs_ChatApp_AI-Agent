from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from rag_engine import RagEngine

app = FastAPI(title="Doc-to-Decks Agentic Pipeline")

# Enable Streamlit frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RagEngine()
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handles secure local storage saving and executes Day 3 Parent-Child indexing."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        total_child_chunks = engine.ingest_pdf(file_path)
        return {
            "status": "Success",
            "filename": file.filename,
            "message": f"Hierarchical index built successfully with {total_child_chunks} child nodes."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-deck")
async def generate_deck(topic: str = Form(...)):
    """Executes Day 4 Re-ranking, applies Day 5 Guardrails, and streams back clean binary PPTX data."""
    if engine.vector_store is None:
        raise HTTPException(status_code=400, detail="No active document matrix found. Please upload a PDF first.")

    try:
        # 1. Fetch structured Pydantic model data directly from the RAG engine
        presentation_object_data = engine.generate_presentation_data(topic)

        # 2. Feed the verified object model properties into the PPTX compiler
        pptx_stream = engine.export_to_pptx(presentation_object_data)

        return StreamingResponse(
            pptx_stream,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={"Content-Disposition": "attachment; filename=presentation.pptx"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))