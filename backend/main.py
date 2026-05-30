from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

# Import your RAG logic file
from rag_engine import RagEngine

app = FastAPI()

# Tell your backend it is allowed to talk to your Streamlit frontend port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the RAG engine once when the server starts
engine = RagEngine()


# Step 1: Health Check Route (Fixes your "API Status Irregular" issue)
@app.get("/")
def home():
    return {"status": "online"}


# Step 2: Basic PDF Upload Route
@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    # Make sure our local folder exists
    os.makedirs("uploaded_docs", exist_ok=True)

    # Define where to save the file
    file_path = os.path.join("uploaded_docs", file.filename)

    # Save the uploaded file to your hard drive
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Pass the local file path to your RAG engine to read it
    try:
        chunks_created = engine.ingest_pdf(file_path)
        return {"message": "Success", "chunks": chunks_created}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Step 3: Simplified Slide Generation Route (Stability Upgrade)
@app.post("/generate-deck")
def make_presentation(topic: str = Form(...)):
    if engine.vector_store is None:
        raise HTTPException(status_code=400, detail="Please upload a PDF first!")

    try:
        # 1. Ask Gemini to compile the data using your existing engine pipeline
        presentation_data = engine.generate_presentation_data(topic)

        # 2. Build the PowerPoint file on disk safely
        ppt_stream = engine.export_to_pptx(presentation_data)
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        ppt_path = os.path.join(backend_dir, "presentation.pptx")
        with open(ppt_path, "wb") as f:
            f.write(ppt_stream.getbuffer())

        # 3. Return a clean success flag so the connection stays open
        return {"status": "success", "message": "Presentation compiled successfully on disk!"}

    except Exception as e:
        # If anything goes wrong inside the engine, catch it here so Uvicorn doesn't disconnect!
        print(f"⚠️ Caught an engine error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Engine compilation failed: {str(e)}")