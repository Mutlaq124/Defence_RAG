'''
s3/gcs -> raw PDFs use
redis cache + Prometheus (metrics tracking) -> token counts, chunking, embedding cache (avoid recompute)
LATER -> replace tiktoken with transformers AutoTokenizer for model-exact token counts
Later -> Cache parsed docs by file hash (avoid re-parse unchanged files)
LLama 3.1 -> cl100k_base-compatible tokenizer (okay for approximate counts),
# (LATER) Must use tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B") + use_fast=True + cache_dir
Sentencizer -> Faster (rule-based: ., !?), struggles with abbreviations.
en_core_web_sm ML model (only parser pipeline active) -> better accuracy for technical text.
'''
from __future__ import annotations
import json
import uuid
from pathlib import Path
import regex as re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from . import config
import structlog
import tiktoken
import hashlib
import spacy
from transformers import AutoTokenizer
from spacy.language import Language

# Configure structlog for structured (JSON) logs;
# ConsoleRenderer -> readable in dev, JSONRenderer -> machine-friendly in prod
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,   # Format exception info if present
        # structlog.processors.JSONRenderer()  # for production
        structlog.dev.ConsoleRenderer()          # for development
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logger = structlog.get_logger()

TOKENIZER_MODEL = config.TOKENIZER_MODEL
# LATER -> USE LAZY LOADING; tiktoken.get_encoding() executes at import time
# (problematic in Docker/CI/CD -- crashes early if vocab file not cached)
encoding = tiktoken.get_encoding("cl100k_base")


@lru_cache(maxsize=4096)
def count_tokens(text: str) -> int:
    """
    Count tokens using tiktoken cl100k_base encoding.

    LRU-cached (maxsize=4096) to avoid recomputing token counts for repeated
    chunk texts. Uses Python's functools.lru_cache -- LRU evicts least-recently-
    used entries when cache is full. For production, replace with Redis to share
    cache across workers.

    Note: cl100k_base is an approximation for Llama-3 models. The actual token
    count may differ by ~5-10% due to different BPE vocabs. Good enough for
    chunking decisions; use AutoTokenizer for generation input budget.
    """
    if not text:
        return 0
    try:
        return len(encoding.encode(text))
    except Exception as e:
        logger.error("token_count_failed", text_preview=text[:100], error=str(e))
        raise


def truncate_to_tokens(text: str, max_tokens: int, suffix: str = "...") -> str:
    """Return text truncated to at most max_tokens tokens (using tiktoken)."""
    if count_tokens(text) <= max_tokens:
        return text
    tokens = encoding.encode(text, disallowed_special=())[:max_tokens]
    return encoding.decode(tokens) + suffix


# ============================================================================
# Lazy-loaded spaCy model for sentence splitting
# en_core_web_trf -> High accuracy (GPU required)
# en_core_web_sm  -> Fast CPU, good accuracy for English technical text
# Fallback        -> sentencizer pipe (rule-based, no model download needed)
# ============================================================================
_nlp: Language | None = None


def get_nlp() -> Language:
    """
    Lazy-load spaCy with en_core_web_sm fallback to sentencizer.

    Loading is deferred to first call to avoid import-time overhead
    and allow Docker containers to start without model files present.
    LATER -> preload at container startup for consistent first-request latency.
    """
    global _nlp
    if _nlp is None:
        try:
            # disable unused pipes -> faster inference (we only need senter/parser)
            _nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
            _nlp.max_length = 2_000_000
            logger.info("spacy_model_loaded", model="en_core_web_sm")
        except OSError:
            logger.warning("spacy_model_missing", fallback="sentencizer")
            # Fallback: blank model with rule-based sentencizer (no download needed)
            _nlp = spacy.blank("en")
            _nlp.add_pipe("sentencizer")
            _nlp.max_length = 2_000_000
            logger.info("spacy_sentencizer_loaded", pipes=_nlp.pipe_names)
        except Exception as e:
            logger.error("spacy_failed", error=str(e))
            _nlp = None
    return _nlp


@lru_cache(maxsize=1024)
def split_into_sentences(text: str) -> List[str]:
    """
    Military-grade sentence splitter for technical manual text.

    Handles: "A/A", "Fig. 1", "1.2.3.", bullet points, OCR noise.
    Uses spaCy for ML-based sentence boundary detection, falls back to
    a regex pattern on failure.

    LATER -> don't use full parser for speed (just sentencizer component).
    LATER -> language detection per element for multilingual chunking.
    """
    if not text or not text.strip():
        return []

    nlp = get_nlp()
    if nlp:
        try:
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except Exception as e:
            logger.warning("spacy_sentence_split_failed", error=str(e))

    # Ultimate fallback: robust regex (handles common English sentence boundaries)
    pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*(?=[A-Z0-9])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def get_pdf_hash(path: str) -> str:
    """
    Return SHA-256 hash of file contents for change detection.

    Used to cache parsed output: if the hash matches the stored hash,
    we skip re-parsing. Reads in 8KB chunks to avoid loading the full
    PDF into RAM.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ============================================================================
# ID & Hashing Utilities
# ============================================================================
'''
Hash_ID -> Chunk + metadata keys -> unique ID per chunk.
Pros -> Same content + metadata = same ID across runs (deduplication, change tracking).
Cons -> Slightly slower than random UUIDs (negligible for our scale).
PRO  -> Normalize (punctuation, whitespace) before hashing for better deduplication.
        Hash length (64-bit truncated prefix) trades collision risk for readability.

Position_ID -> based on doc_id + element_id + chunk index.
Pros -> Human-readable, traceable to source doc/element.
Cons -> Changes if doc structure changes (new elements added upstream).

Decision: use content hash (UUIDv5) so the same text always gets the same Qdrant ID.
This enables safe re-indexing without creating duplicate points (upsert is idempotent).
'''

# Fixed namespace - same across all machines and runs for deterministic IDs
RAG_NAMESPACE = uuid.UUID("1a2b3c4d-5e6f-7a8b-9c0d-e1f2a3b4c5d6")


def generate_deterministic_id(
    text: str,
    metadata: Dict[str, Any] = None,
    prefix: str = "chunk"
) -> Dict[str, str]:
    """
    Generate deterministic chunk_id and qdrant_id from text + metadata.

    Returns both IDs in one call:
        - chunk_id  -> pretty, short, human-readable (prefix + 16-char hex)
        - qdrant_id -> valid UUIDv5 string for Qdrant point ID

    The stable_input combines text content + key metadata fields so that:
    - Same chunk in same position -> same ID (idempotent upsert)
    - Same text in different doc/page/section -> different ID (no collision)

    LATER -> normalize punctuation/whitespace before hashing for better deduplication
    across slight OCR variations of the same content.
    """
    clean_text = text.strip() if text else ""
    stable_input = clean_text

    # Add stable metadata (same chunk context -> same ID forever)
    if metadata:
        source_doc = metadata.get("source_doc")
        if source_doc:
            stable_input += f"|source_doc:{source_doc}"
        # Sort keys for determinism (dict ordering not guaranteed in older Python)
        for key in sorted(['page', 'section_heading', 'element_id', 'split_index']):
            val = metadata.get(key)
            if val is not None:
                stable_input += f"|{key}:{val}"

    hash_hex = hashlib.sha256(stable_input.encode('utf-8')).hexdigest()
    qdrant_id = str(uuid.uuid5(RAG_NAMESPACE, stable_input))

    return {
        "chunk_id": f"{prefix}_{hash_hex[:16]}",
        "qdrant_id": qdrant_id
    }


def save_json(path: str, obj: Any):
    """Save object as indented JSON, creating parent directories as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    """Load JSON from file with descriptive error on parse failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error("invalid_json", path=path, error=str(e))
        raise


# ============================================================================
# Sliding Window Token-Based Chunking Utility
# ============================================================================
# Produces (start, end) index pairs over a list of per-sentence token counts.
# Each window satisfies: sum(token_counts[start:end]) <= max_tokens
# Adjacent windows overlap by >= overlap_tokens (sentence-boundary aligned).
#
# Rationale for token-based (not character-based) overlap:
#   Token counts are what the embedding/generator model actually sees.
#   Character overlap can wildly over- or under-estimate the real overlap
#   depending on token length distribution in the text.
#
# Edge cases handled:
#   - Single sentence > max_tokens: included anyway with a warning
#   - Empty input: returns []
#   - overlap_tokens > max_tokens: clamped by the backtrack loop
#   LATER -> add explicit guard for overlap >= max_tokens config error

"""
Create sliding windows over sentences with token-based overlap.

Args:
    token_counts: List of token counts per sentence
    max_tokens: Maximum tokens per window
    overlap_tokens: Minimum overlap tokens between consecutive windows

Returns:
    List of (start_idx, end_idx) tuples (end_idx is exclusive)

Example:
    token_counts = [50, 100, 150, 200, 100]
    max_tokens = 300, overlap_tokens = 100
    Returns: [(0, 3), (2, 5)]
    Window 1: sentences 0-2 (50+100+150=300 tokens)
    Window 2: sentences 2-4 (150+200+100=450 tokens)
    Overlap:  sentence 2 (150 tokens)

Algorithm:
    1. Extend window right until max_tokens reached
    2. Record window (i, j)
    3. Backtrack from j-1 to maintain overlap_tokens
    4. Set next start ensuring forward progress
"""


def sliding_window_token_indices(
    token_counts: List[int],
    max_tokens: int,
    overlap_tokens: int = 0,
) -> List[Tuple[int, int]]:
    if not token_counts:
        return []

    windows = []
    n = len(token_counts)
    i = 0

    while i < n:
        j = i
        current_tokens = 0

        while j < n:
            tokens = token_counts[j]

            # Giant sentence safety valve: include oversized sentence as its own chunk
            if tokens > max_tokens:
                if current_tokens == 0:
                    j += 1      # accept it anyway (can't split individual sentences here)
                break

            if current_tokens + tokens > max_tokens:
                break

            current_tokens += tokens
            j += 1

        windows.append((i, j))

        if j >= n:
            break

        # Backtrack to achieve token-level overlap with previous window
        overlap_accumulated = 0
        next_i = j
        for k in range(j - 1, i, -1):
            overlap_accumulated += token_counts[k]
            if overlap_accumulated >= overlap_tokens:
                next_i = k
                break
            next_i = k  # keep moving back

        # Always make forward progress (prevents infinite loop on degenerate input)
        if next_i <= i:
            next_i = i + 1

        i = next_i

    return windows


'''
Example trace:
token_counts = [50, 100, 150, 200, 100]   (5 sentences)
max_tokens=300, overlap_tokens=100

Iteration 1: i=0
  extend j: 0->1(50), 1->2(150), 2->3(300=max), stop at j=3
  window = (0, 3)  -> sentences 0,1,2 (300 tokens)
  backtrack from j-1=2: accum=150>100, next_i=2

Iteration 2: i=2
  extend j: 2->3(150), 3->4(350>300), stop at j=3... wait
  next extend: j=2, current=0; tok[2]=150<=300, add; j=3; tok[3]=200, 150+200=350>300, stop
  window = (2, 3)  -> sentence 2 only (150 tokens)
  j=3; backtrack from 2: only one step, next_i=2 <= i=2, so next_i=3

Iteration 3: i=3
  extend: tok[3]=200<=300, j=4; tok[4]=100, 200+100=300<=300, j=5; j>=n, stop
  window = (3, 5)  -> sentences 3,4 (300 tokens)
  j>=n -> break

Final windows = [(0,3), (2,3), (3,5)]
'''