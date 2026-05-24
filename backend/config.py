CHUNK_SIZE_PARENT = 2000
CHUNK_OVERLAP_PARENT = 200
CHUNK_SIZE_CHILD = 400
CHUNK_OVERLAP_CHILD = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"

FAISS_PATH = "faiss_index"
# because the context window is small so we need to extract several child chunks
TOP_K = 12
MAX_LENGTH = 512
TEMPERATURE = 0.3