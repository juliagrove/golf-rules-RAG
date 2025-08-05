import os

from dotenv import load_dotenv

load_dotenv()

import os

# directories
PDF_FOLDER = os.path.join("data", "pdf")
SCRAPED_FOLDER = os.path.join("data", "scraped")
CHUNKED_FOLDER = os.path.join("data", "chunked_output")
FAISS_FOLDER = os.path.join("data", "faiss")

# embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# chunk setttings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# LLM Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
