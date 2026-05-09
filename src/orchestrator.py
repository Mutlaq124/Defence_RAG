"""
RAG Pipeline Orchestrator — Query-Time Only

This module handles query-time orchestration (embed -> retrieve -> rerank -> generate).
Indexing is performed separately via run_pipeline.py --mode steps.

Senior developer decision: keeping query path and indexing path separate prevents
accidental re-indexing during serving, and allows the UI to query without needing
write permissions to Qdrant or the filesystem.

LATER -> P99 latency tracking (how slow are the slowest 1% of requests)
LATER -> Query normalization (lowercase, remove punctuation, domain-specific keywords)
LATER -> Wrap in FastAPI for production serving with health checks
LATER -> Multiple services -> docker-compose (qdrant, redis, ollama, worker)
LATER -> Monitoring with Prometheus + Grafana (latencies, errors, usage)
LATER -> Celery for task queues (async indexing jobs)
LATER -> Redis cache: if same query within N minutes, return cached result (avoid re-embed/retrieve)
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time

from .utils import logger
from .embedder_ollama import OllamaEmbedder
from .qdrant_store import VectorStoreQdrant
from .reranker import CrossEncoderReranker
from .generator_ollama import OllamaGenerator
from . import config


@dataclass
class QueryMetrics:
    """Performance metrics for a single query execution."""
    total_time: float
    retrieval_time: float
    rerank_time: float
    generation_time: float
    chunks_retrieved: int
    chunks_reranked: int


class RAGPipeline:
    """
    Production RAG query orchestrator.

    Responsibilities:
      - embed_query -> search Qdrant -> rerank -> generate answer
      - NOT responsible for indexing (see run_pipeline.py)

    Initialization loads all models into memory (embedder, reranker, generator).
    Keep this as a singleton in Streamlit session_state to avoid reloading on
    every rerun. Model loading is the most expensive operation (~5-30s depending
    on hardware and model size).

    LATER -> Lazy-load models on first query (not at __init__) to reduce startup time.
    """

    def __init__(self):
        logger.info("initializing_rag_pipeline")
        self.embedder = OllamaEmbedder()
        self.qdrant = VectorStoreQdrant()
        self.reranker = CrossEncoderReranker()
        self.generator = OllamaGenerator()
        logger.info("pipeline_ready")

    def get_collection_stats(self) -> Dict:
        """
        Return Qdrant collection statistics for UI display.

        Returns:
            dict with points_count, vectors_count, status, dimension
        """
        return self.qdrant.get_collection_info()

    def query(
        self,
        question: str,
        top_k: int = 2,
        retrieve_k: int = 10
    ) -> Tuple[str, List[Dict], QueryMetrics]:
        """
        Full RAG query: embed -> retrieve -> rerank -> generate.

        Args:
            question: User question string
            top_k: Final number of chunks to pass to generator after reranking
            retrieve_k: Initial candidate count from vector search (before reranking)

        Returns:
            (answer_str, reranked_docs, QueryMetrics)

        Pipeline:
            1. Embed question with same model used for indexing (critical: must match)
            2. ANN search in Qdrant (top retrieve_k candidates)
            3. Cross-encoder rerank (computationally heavier, more accurate)
            4. Concatenate top_k chunk texts as context
            5. Generate answer with prompt template (RAG grounded, no hallucination)
        """
        start_time = time.perf_counter()

        logger.info("query_start", question=question[:100])

        # Step 1: Embed query
        # CRITICAL: must use same embedder model as indexing (nomic-embed-text)
        # Different model -> different vector space -> garbage retrieval results
        try:
            q_vec = self.embedder.embed_batch([question])[0]
        except Exception as e:
            logger.error("query_embedding_failed", error=str(e))
            raise RuntimeError(f"Query embedding failed: {e}") from e

        # Step 2: Vector search (ANN)
        retrieval_start = time.perf_counter()
        try:
            raw_hits = self.qdrant.search(q_vec, top_k=retrieve_k)
        except Exception as e:
            logger.error("retrieval_failed", error=str(e))
            raise RuntimeError(f"Retrieval failed: {e}") from e

        retrieval_time = time.perf_counter() - retrieval_start
        logger.info("retrieved", count=len(raw_hits))

        if not raw_hits:
            logger.warning("no_results_found", question=question[:100])
            # Return early with empty metrics (avoid metrics-before-assignment bug)
            empty_metrics = QueryMetrics(
                total_time=time.perf_counter() - start_time,
                retrieval_time=retrieval_time,
                rerank_time=0.0,
                generation_time=0.0,
                chunks_retrieved=0,
                chunks_reranked=0,
            )
            return "No relevant information found in the F-16 manual.", [], empty_metrics

        # Convert raw hits to doc format for reranker
        docs = []
        for hit in raw_hits:
            docs.append({
                "text": hit.get("text", ""),
                "metadata": hit.get("metadata", {}),
                "vector_score": hit.get("vector_score", 0.0),
                "rerank_score": hit.get("vector_score", 0.0),  # overwritten by reranker
            })

        # Step 3: Cross-encoder rerank
        # Reranker sees (query, chunk) pairs and scores relevance directly.
        # More accurate than vector similarity but O(n*tokens) per pair.
        rerank_start = time.perf_counter()
        try:
            reranked = self.reranker.rerank(question, docs, top_k=top_k)
        except Exception as e:
            logger.error("reranking_failed", error=str(e))
            logger.warning("using_vector_scores_only")
            reranked = docs[:top_k]

        rerank_time = time.perf_counter() - rerank_start
        logger.info("reranked", top_k=len(reranked))

        # Step 4: Build context string
        context = "\n\n---\n\n".join([d["text"] for d in reranked])

        # Step 5: Generate answer
        generation_start = time.perf_counter()
        try:
            answer = self.generator.generate_with_template(question=question, context=context)
        except Exception as e:
            logger.error("generation_failed", error=str(e))
            raise RuntimeError(f"Answer generation failed: {e}") from e

        generation_time = time.perf_counter() - generation_start
        total_time = time.perf_counter() - start_time

        metrics = QueryMetrics(
            total_time=total_time,
            retrieval_time=retrieval_time,
            rerank_time=rerank_time,
            generation_time=generation_time,
            chunks_retrieved=len(raw_hits),
            chunks_reranked=len(reranked),
        )

        logger.info("query_complete", metrics=metrics.__dict__)
        return answer, reranked, metrics