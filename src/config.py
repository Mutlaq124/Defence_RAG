# LATER -> Pydantic validation for config values to ensure correct types/formats

'''
env values -> would be given highest priority (over defaults here)
generator tokenizer that matches with generator model
Option A: Docker Qdrant (recommended for speed + persistence) -> QDRANT_URL
Option B: Local file-based storage (slower, no Docker required) -> QDRANT_PATH
Only ONE of QDRANT_URL or QDRANT_PATH should be active at a time.
'''

from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env file into os.environ (values here are fallback defaults)
load_dotenv()

# --- System ---
# Change to 4 for balance on lower-end machines
MAX_CORES = int(os.getenv("MAX_CORES", "6"))   # CPU threads for Docling accelerator
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))  # for parallel processing

# --- Infrastructure ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "F16_Manual_Collection")

# Option A: Docker Qdrant (recommended for speed)
# QDRANT_URL = "http://localhost:6333"
# QDRANT_PATH = None

# Option B: Local file-based (slower, but no Docker)
# QDRANT_URL = None
# QDRANT_PATH = "./qdrant_data"

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)   # None for local Qdrant
QDRANT_PATH = os.getenv("QDRANT_PATH", None)          # Only used if QDRANT_URL is None

# --- Models ---
# MxBAI large (best MTEB score) has token limit of 512 only.
# nomic-embed-text supports up to 2048 tokens -> better for long technical chunks.
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")

# Must be under max_trained_positions of model (2048 for nomic -> 200 buffer
# due to different tokenizers between tiktoken and the model's own BPE).
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))  # Qdrant collection size must match

# ModelFile -> If answer not from contexts -> "NOT in context"
GEN_MODEL = os.getenv("GEN_MODEL", "llama3.1:8b-instruct-q4_K_S")

# Alibaba-NLP/gte-reranker-modernbert-base: fast, 8192 tokens limit, extremely strong performance
# Not using ms-marco-MiniLM due to 512 token input limit (too short for our chunks)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Alibaba-NLP/gte-reranker-modernbert-base")

# Select 13b or bigger LLM for better quality golden answers (Temperature = 0)
EVAL_GROUND_TRUTH_MODEL = os.getenv("EVAL_GROUND_TRUTH_MODEL", "llama3.1:8b-instruct-q4_K_S")

# --- Embedder Parameters ---
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

# --- Generation Parameters ---
MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "512"))

# --- Chunking Parameters ---
# Decreased from 1000 to 500 to respect reranker (bge-reranker-v2-m3) input limits.
# LATER -> tune per domain: technical manuals benefit from larger chunks (more context).
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "1000"))
MIN_CHUNK_TOKENS = int(os.getenv("MIN_CHUNK_TOKENS", "20"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))

# cl100k_base: tiktoken encoding compatible with GPT-4/Llama-3 models.
# LATER -> switch to AutoTokenizer.from_pretrained(GEN_MODEL) for exact token counts.
TOKENIZER_MODEL = os.getenv("TOKENIZER_MODEL", "cl100k_base")

# --- Cache ---
# Cache directory for HuggingFace model weights (reranker, etc.)
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", "./models_cache")

# --- Retrieval Parameters ---
TOP_K_INITIAL = int(os.getenv("TOP_K_INITIAL", "8"))   # candidates before reranking
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "2"))        # final chunks passed to generator

# --- Processing ---
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
EVAL_QUERIES = int(os.getenv("EVAL_QUERIES", "100"))

# --- File and Directory Paths ---
DATA_DIR = os.getenv("DATA_DIR", "./data")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
EVAL_DIR = os.getenv("EVAL_DIR", "./dataset")         # output path for eval CSV + results
PARSED_FILENAME = "parsed_doc.json"
EVAL_DATASET_PATH = os.getenv("EVAL_DATASET_PATH", "./dataset/eval_dataset.csv")

# --- MLflow Configuration ---
# LATER -> set to absolute path or remote URI (e.g., http://mlflow-server:5000)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
MLFLOW_ARTIFACT_LOCATION = os.getenv("MLFLOW_ARTIFACT_LOCATION", None)
PIPELINE_VERSION = "1.0.0"
ENV = os.getenv("ENV", "dev")  # dev | staging | prod

# --- Create necessary directories at import time ---
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(EVAL_DIR).mkdir(parents=True, exist_ok=True)
# MAX_TABLE_TOKENS -> for chunking tables separately if needed (LATER)
