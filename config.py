CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyAO_DL5b8XCrcs2nimzcBgmZu6C9GrwOiM"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"

FAISS_PATH = "faiss_index"

TOP_K = 4
MAX_LENGTH = 512
TEMPERATURE = 0.3