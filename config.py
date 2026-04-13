import os

# Gmail
CREDENTIALS_FILE = os.getenv("CREDENTIALS_FILE", "gmail_main_new_secret.json")
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# ChromaDB
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "emails")

# Embeddings
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

# Pipeline
N_EMAILS = int(os.getenv("N_EMAILS", "50"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
