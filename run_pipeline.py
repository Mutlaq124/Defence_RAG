"""
Pipeline Runner

Runs each pipeline stage independently or all at once.

Usage:
  --mode parse   ->  python run_pipeline.py --pdf data/manual.pdf --mode parse
  --mode chunk   ->  python run_pipeline.py --mode chunk
  --mode index   ->  python run_pipeline.py --mode index
  --mode all     ->  python run_pipeline.py --pdf data/manual.pdf --mode all

"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.utils import logger, save_json, load_json


def print_header():
    print("=" * 70)
    print("  F-16 MANUAL RAG — INDEXING PIPELINE")
    print("=" * 70)
    print()


def print_step(step_num, title):
    print(f"\n{'='*70}")
    print(f"  STEP {step_num}: {title}")
    print(f"{'='*70}\n")


def check_qdrant() -> bool:
    """Verify Qdrant is reachable before indexing."""
    try:
        import requests
        resp = requests.get(f"{config.QDRANT_URL}/collections", timeout=5)
        if resp.status_code == 200:
            print("  Qdrant is accessible")
            return True
        else:
            print(f"  Qdrant responded with status {resp.status_code}")
            return False
    except Exception as e:
        print(f"  Qdrant not accessible: {e}")
        print(f"  Start with: docker run -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant")
        return False


def check_ollama() -> bool:
    """Verify Ollama is reachable and required models are available."""
    try:
        import requests
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            print(f"  Ollama responded with status {resp.status_code}")
            return False

        print("  Ollama is accessible")
        models = resp.json().get('models', [])
        model_names = [m['name'] for m in models]

        embed_ok = any(config.EMBED_MODEL in m for m in model_names)
        gen_ok = any(config.GEN_MODEL in m for m in model_names)

        if not embed_ok:
            print(f"  Embedding model not found: {config.EMBED_MODEL}")
            print(f"  Pull with: ollama pull {config.EMBED_MODEL}")
        else:
            print(f"  Embedding model ready: {config.EMBED_MODEL}")

        if not gen_ok:
            print(f"  Generator model not found: {config.GEN_MODEL}")
            print(f"  Pull with: ollama pull {config.GEN_MODEL}")
        else:
            print(f"  Generator model ready: {config.GEN_MODEL}")

        return embed_ok and gen_ok

    except Exception as e:
        print(f"  Ollama not accessible: {e}")
        print(f"  Start with: ollama serve")
        return False


# ============================================================================
# STEP 1: PARSE
# ============================================================================
def run_parse(pdf_path: str) -> dict:
    """
    Run Docling parser on PDF. Output saved to output/{stem}_parsed.json.
    Hash file written to output/{stem}_hash.txt for cache detection.

    Returns:
        parsed dict (doc_meta + elements)
    """
    from src.parser_docling import parse_docling

    print_step(1, "PARSE PDF WITH DOCLING")
    print(f"  Input: {pdf_path}")
    print(f"  OCR: enabled (EasyOCR)")
    print(f"  Table extraction: ACCURATE mode\n")

    start = time.time()
    try:
        parsed = parse_docling(pdf_path)
        elapsed = time.time() - start

        elements = parsed.get("elements", [])
        by_type = {}
        for e in elements:
            t = e.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1

        print(f"  Parsing complete in {elapsed:.1f}s")
        print(f"  Total elements: {len(elements)}")
        for t, count in by_type.items():
            print(f"    {t}: {count}")
        print(f"\n  Output: output/{Path(pdf_path).stem}_parsed.json")

        return parsed

    except Exception as e:
        print(f"  Parsing failed: {e}")
        logger.error("parsing_failed", error=str(e))
        raise


# ============================================================================
# STEP 2: CHUNK
# ============================================================================
def run_chunk(parsed: dict = None, parsed_path: str = None) -> list:
    """
    Run chunker on parsed document. Output saved to data/chunks.json.

    Args:
        parsed: Already-loaded parsed dict (if running after parse step)
        parsed_path: Path to _parsed.json file (if running chunk standalone)

    Returns:
        chunks list
    """
    from src.chunker import chunk_document

    print_step(2, "CHUNK DOCUMENT")
    print(f"  Max tokens/chunk: {config.MAX_CHUNK_TOKENS}")
    print(f"  Min tokens/chunk: {config.MIN_CHUNK_TOKENS}")
    print(f"  Overlap tokens:   {config.CHUNK_OVERLAP_TOKENS}\n")

    if parsed is None:
        if parsed_path is None:
            # Find the parsed JSON in output dir
            output_files = list(Path(config.OUTPUT_DIR).glob("*_parsed.json"))
            if not output_files:
                raise FileNotFoundError(
                    f"No _parsed.json found in {config.OUTPUT_DIR}. "
                    f"Run: python run_pipeline.py --pdf data/F_16_manual.pdf --mode parse"
                )
            parsed_path = str(output_files[-1])  # Use most recent
            print(f"  Loading parsed file: {parsed_path}")

        parsed = load_json(parsed_path)

    start = time.time()
    try:
        chunks = chunk_document(parsed)
        elapsed = time.time() - start

        total_tokens = sum(c.get("tokens", 0) for c in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0

        chunk_types = {}
        for c in chunks:
            t = c.get("chunk_type", "unknown")
            chunk_types[t] = chunk_types.get(t, 0) + 1

        print(f"  Chunking complete in {elapsed:.2f}s")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Avg tokens/chunk: {avg_tokens:.0f}")
        for t, count in chunk_types.items():
            print(f"    {t}: {count}")
        print(f"\n  Output: data/chunks.json")

        return chunks

    except Exception as e:
        print(f"  Chunking failed: {e}")
        logger.error("chunking_failed", error=str(e))
        raise


# ============================================================================
# STEP 3: INDEX (Embed + Store in Qdrant)
# ============================================================================
def run_index(chunks: list = None, tracker=None) -> dict:
    """
    Embed chunks and upsert into Qdrant.

    Args:
        chunks: Already-loaded chunks list (if running after chunk step)
        tracker: Optional MLflow tracker for metric logging

    Returns:
        collection info dict
    """
    from src.embedder_ollama import OllamaEmbedder
    from src.qdrant_store import VectorStoreQdrant

    print_step(3, "EMBED + INDEX IN QDRANT")

    if chunks is None:
        chunks_path = Path(config.DATA_DIR) / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"chunks.json not found at {chunks_path}. "
                f"Run: python run_pipeline.py --mode chunk"
            )
        print(f"  Loading chunks: {chunks_path}")
        chunks = load_json(str(chunks_path))

    texts = [c["text"] for c in chunks]
    print(f"  Chunks to embed: {len(chunks)}")
    print(f"  Embed model: {config.EMBED_MODEL}")
    print(f"  Qdrant collection: {config.QDRANT_COLLECTION}\n")

    # Embed
    embedder = OllamaEmbedder()
    embed_start = time.time()
    embeddings = embedder.embed_batch(texts)
    embed_time = time.time() - embed_start

    print(f"  Embedding complete in {embed_time:.1f}s")
    print(f"  Throughput: {len(embeddings)/embed_time:.1f} chunks/sec")
    print(f"  Vector dimension: {len(embeddings[0]) if embeddings else 0}")

    # Index
    qdrant = VectorStoreQdrant()
    index_start = time.time()
    qdrant.add(embeddings, chunks)
    index_time = time.time() - index_start

    # Verify
    info = qdrant.get_collection_info()
    print(f"\n  Indexing complete in {index_time:.1f}s")
    print(f"\n  Collection: {config.QDRANT_COLLECTION}")
    print(f"  Points indexed: {info.get('points_count', 0)}")
    print(f"  Vector dimension: {info.get('dimension', 0)}")
    print(f"\n  Dashboard: http://localhost:6333/dashboard")

    if tracker:
        tracker.log_metrics({
            "embed_time_sec": embed_time,
            "index_time_sec": index_time,
            "embed_throughput_chunks_per_sec": len(chunks) / embed_time,
        })

    return info


# ============================================================================
# MAIN
# ============================================================================
def main():
    print_header()

    parser = argparse.ArgumentParser(
        description="F-16 RAG Pipeline Indexing Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full pipeline:
    python run_pipeline.py --pdf data/F_16_manual.pdf --mode all

  Step by step (recommended for iteration):
    python run_pipeline.py --pdf data/F_16_manual.pdf --mode parse
    python run_pipeline.py --mode chunk
    python run_pipeline.py --mode index

  Re-chunk with different params (edit .env first, then):
    python run_pipeline.py --mode chunk
    python run_pipeline.py --mode index
        """
    )
    parser.add_argument(
        "--pdf", type=str, default=None,
        help="Path to PDF file (required for --mode parse or --mode all)"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["parse", "chunk", "index", "all"],
        help="Pipeline stage to run (default: all)"
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Skip MLflow tracking (faster for quick tests)"
    )

    args = parser.parse_args()

    # Validate args
    if args.mode in ("parse", "all") and not args.pdf:
        parser.error("--pdf is required for --mode parse or --mode all")

    if args.mode in ("parse", "all") and not Path(args.pdf).exists():
        print(f"  Error: PDF not found: {args.pdf}")
        return 1

    # Setup MLflow
    tracker = None
    if not args.no_mlflow:
        try:
            from src.mlflow_tracker import get_tracker
            tracker = get_tracker(experiment_name="F16_RAG_Pipeline")
        except Exception as e:
            print(f"  MLflow init failed (continuing without tracking): {e}")

    # Print config summary
    print(f"  Mode:       {args.mode}")
    print(f"  PDF:        {args.pdf or 'N/A'}")
    print(f"  Chunk size: {config.MAX_CHUNK_TOKENS} tokens")
    print(f"  Overlap:    {config.CHUNK_OVERLAP_TOKENS} tokens")
    print(f"  MLflow:     {'disabled' if args.no_mlflow else config.MLFLOW_TRACKING_URI}")

    total_start = time.time()
    run_name = f"{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    ctx = tracker.start_run(run_name=run_name) if tracker else None

    try:
        if ctx:
            ctx.__enter__()

        parsed = None
        chunks = None

        if args.mode in ("parse", "all"):
            print("\n  Checking dependencies...")
            if not check_ollama():
                print("\n  Fix Ollama issues and retry.")
                return 1
            parsed = run_parse(args.pdf)

        if args.mode in ("chunk", "all"):
            chunks = run_chunk(parsed=parsed)

        if args.mode in ("index", "all"):
            print("\n  Checking dependencies...")
            if not check_qdrant():
                print("\n  Fix Qdrant issues and retry.")
                return 1
            if not check_ollama():
                print("\n  Fix Ollama issues and retry.")
                return 1
            run_index(chunks=chunks, tracker=tracker)

        total_time = time.time() - total_start

        # Log artifacts
        if tracker:
            parsed_files = list(Path(config.OUTPUT_DIR).glob("*_parsed.json"))
            if parsed_files:
                tracker.log_artifact(str(parsed_files[-1]), "intermediate")
            chunks_file = Path(config.DATA_DIR) / "chunks.json"
            if chunks_file.exists():
                tracker.log_artifact(str(chunks_file), "intermediate")

        if ctx:
            ctx.__exit__(None, None, None)

        print(f"\n{'='*70}")
        print(f"  DONE — mode={args.mode} | total time: {total_time:.1f}s")
        print(f"{'='*70}")
        print(f"\n  Next step:")

        if args.mode == "parse":
            print(f"    python run_pipeline.py --mode chunk")
        elif args.mode == "chunk":
            print(f"    python run_pipeline.py --mode index")
        elif args.mode in ("index", "all"):
            print(f"    streamlit run app.py")
            print(f"    mlflow ui --port 5000  (in separate terminal)")

        return 0

    except Exception as e:
        print(f"\n  Pipeline failed: {e}")
        logger.exception("pipeline_failed")
        if tracker:
            tracker.end_run(status="FAILED")
        if ctx:
            ctx.__exit__(type(e), e, e.__traceback__)
        return 1


if __name__ == "__main__":
    sys.exit(main())