'''
Important concepts:
Collection -> Top-level dataset. Named group of points + settings (vector size,
             distance metric, indexing strategy). Tune at creation time.

Point      -> Individual data record: {id, vector: float[], payload: JSON metadata}
             PointStruct -> transport object for defining points with vectors + payloads.

Payload    -> Flexible JSON attached to each point. Prefer payload indexing for
             frequently queried fields (e.g., page, chunk_type) to avoid full scans.

Distance metrics:
  COSINE   -> Qdrant normalizes both stored and query vectors automatically at
              query time. Use this unless vectors are already unit-length.
  DOT      -> Faster if you pre-normalize (dot product of unit vectors = cosine).
  EUCLID   -> L2 distance; good for dense float spaces (less common for text).

  Prefer to keep on_disk false (keep in RAM) (latency-critical applications)
on_disk=True -> Vectors stored in memory-mapped files, not RAM.
              Good for large collections (>1M vectors) or RAM-limited machines.
              Slight latency increase (~2-3x vs RAM), but scales to TB of vectors.
              LATER -> check recall/speed/memory tradeoffs for your collection size.

HNSW index -> Approximate Nearest Neighbor graph (default in Qdrant).
             Built lazily as points are upserted. For small collections (<50k points),
             consider flat index (brute-force) for exact search with lower overhead.

Payload indexes -> B-tree (integer, float) or inverted index (keyword, text).
                 Create for fields used in filter conditions (page, chunk_type, source_doc).
                 Avoid indexing full-text blob fields -> store those in external DB.

Batch upsert:
  wait=True  -> Server confirms all batches are indexed before responding.
               Safe but slower (each batch blocks until indexed).
  wait=False -> Fire-and-forget; 10x faster, but search may miss points briefly.
               Risk: race condition if you search immediately after upsert.
  LATER -> use wait=False + poll collection status for production bulk ingestion.

For Hybrid Search: (Later)
1. Create collection with `vectors_config` containing both dense and sparse vector spaces.
2. Encode sparse vectors at index time (BM25 term weights per chunk).
3. Use `query_points(query=dense_vec, sparse_query=sparse_vec, fusion=RRF)`.


Connection management (FastAPI / async):
  Use singleton pattern or DI to reuse QdrantClient instance.
  Avoid creating new TCP connections per request (high overhead).
  LATER -> Connection pooling for high-throughput applications.

OpenTelemetry integration (LATER):
  Trace each operation (embed -> upsert -> search) as spans.
  Export to Prometheus + Grafana for latency/error dashboards.
'''

import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams,
    Distance,
    PointStruct,       # Transport object for defining points with vectors and payloads
    PayloadSchemaType, # For payload index type constants
)
from qdrant_client.models import Filter, FieldCondition, MatchValue, Record

from typing import List, Dict, Any, Optional
from . import config
from .utils import logger


class VectorStoreQdrant:
    """
    Wrapper around Qdrant for:
      - Indexing: batch upsert embeddings with chunk metadata
      - Querying: ANN search with optional payload filtering
      - Inspection: collection info, full payload scroll for eval/debug

    Designed as a stateless wrapper (no internal cache). For production,
    use as a singleton to reuse the underlying QdrantClient TCP connection.

    LATER -> Add connection pooling + async client for FastAPI serving.
    """

    def __init__(self, use_flat_index: bool = True):
        # Connect to Qdrant (URL mode for Docker; set QDRANT_PATH for file-based mode)
        # LATER -> ONLY ONE of URL or PATH should be active (enforced in config)
        self.client = QdrantClient(
            # host="localhost",
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            # path=config.QDRANT_PATH,   # Uncomment for local file-based storage
            timeout=60,
            # prefer_grpc=True,          # Use for production (lower latency than HTTP)
            # grpc_port=6334
        )
        self.collection = config.QDRANT_COLLECTION
        self.use_flat_index = use_flat_index
        self._ensure_collection()
        self._create_payload_indexes()

    def _ensure_collection(self):
        """
        Create Qdrant collection if it doesn't exist.

        Vector config decisions:
          - size: must match EMBEDDING_DIM from config (nomic-embed-text -> 768)
          - distance: COSINE chosen because nomic-embed-text is not pre-normalized.
            Qdrant auto-normalizes at query time for COSINE, so no manual normalization needed.
          - on_disk=True: vectors stored in memory-mapped files, not RAM.
            Good for our ~300-500 chunk collection; scales to millions without RAM spike.
          LATER -> For >100k chunks, tune HNSW m/ef_construct for recall vs latency tradeoff.
        """
        if not self.client.collection_exists(self.collection):
            logger.info("creating_qdrant_collection", name=self.collection)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=config.EMBEDDING_DIM,
                    distance=Distance.COSINE,
                    on_disk=True,               # Store vectors on disk to save RAM
                    quantization_config=None,   # LATER -> scalar/product quantization for 4-8x memory reduction
                )
            )

    def add(self, embeddings: List[List[float]], metadatas: List[Dict], batch_size: int = 64):
        """
        Batch upsert embeddings with chunk metadata into Qdrant.

        Uses upsert (not insert) so re-indexing is idempotent: same qdrant_id
        overwrites the existing point without creating duplicates.

        Batch size of 64: balances RAM usage vs HTTP overhead.
        For 5000 chunks: 5000 PointStruct objects in memory -> one HTTP req per batch.
        LATER -> use wait=False + status polling for faster bulk ingestion.
        """
        points = []
        for vec, meta in zip(embeddings, metadatas):
            points.append(
                PointStruct(
                    id=meta["qdrant_id"],   # UUIDv5 string -- correct format for Qdrant
                    vector=vec,
                    payload=meta,
                )
            )

        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=points[i:i + batch_size],
                wait=True   # Ensure all points are indexed before confirming
                            # wait=False -> 10x faster but risks race conditions on immediate search
            )
        logger.info("points_upserted", count=len(points))
# verify collection immediately after indexing 
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Return collection statistics for verification and UI display.

        Returns dict with keys: points_count, vectors_count, status, dimension.
        LATER -> add segment count, disk usage for monitoring dashboards.
        """
        try:
            info = self.client.get_collection(self.collection)
            logger.info("collection_info_retrieved", points=info.points_count)
            return {
                "points_count": info.points_count,
                "vectors_count": info.points_count,   # same as points_count for single-vector collections
                "status": str(info.status),
                "dimension": info.config.params.vectors.size,
            }
        except Exception as e:
            logger.exception("get_collection_info_failed")
            return {"error": str(e)}

    def delete_collection(self):
        """Delete collection for clean reruns (use before re-indexing with different params)."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
            logger.info("collection_deleted", name=self.collection)

    def search(self, query_vector: List[float], top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict]:
        """
        Search using qdrant-client 1.9+ query_points() API.

        The older client.search() is deprecated in 1.9+. query_points() is the
        unified search interface that supports vectors, sparse vectors, and multivec.

        Args:
            query_vector: Dense embedding of the query (must match collection dimension)
            top_k: Number of results to return
            filters: Optional dict of {field: value} for payload filtering
                     e.g., {"chunk_type": "table"} to restrict search to tables

        Returns:
            List of dicts with keys: id, vector_score, text, metadata, rerank_score
        """
        search_kwargs = {
            "collection_name": self.collection,
            "query": query_vector,   # 'query' parameter name for query_points()
            "limit": top_k,
            "with_payload": True,
            "with_vectors": False,   # Don't return vectors (saves bandwidth)
        }

        if filters:
            conditions = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filters.items()
            ]
            search_kwargs["query_filter"] = Filter(must=conditions)

        try:
            response = self.client.query_points(**search_kwargs)
        except Exception as e:
            logger.error("qdrant_query_failed", error=str(e))
            return []

        # Convert QueryResponse -> python list expected by RAG pipeline
        results = []
        for p in response.points:
            results.append({
                "id": p.id,
                "vector_score": p.score,
                "text": p.payload.get("text", ""),
                "metadata": p.payload,
                "rerank_score": p.score,    # overwritten by reranker after search
            })
        return results

    def get_all_payloads(self) -> List[Record]:
        """
        Scroll all points from collection for eval dataset generation.

        Uses Qdrant scroll API (cursor-based pagination, 1000 points/page).
        Inefficient for large collections (>50k chunks) -- for those,
        use random sampling via Qdrant's /points/sample endpoint (LATER).

        Returns list of Record objects with .payload attribute.
        """
        all_points = []
        offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=self.collection,
                with_payload=True,
                with_vectors=False,
                limit=1000,
                offset=offset
            )
            all_points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset
        logger.info("get_all_payloads", count=len(all_points))
        return all_points

    def _create_payload_indexes(self):
        """
        Create payload indexes for frequently filtered fields.

        Payload indexes work like database column indexes:
          - integer index: range queries (page >= 5 AND page <= 10)
          - keyword index: exact match (chunk_type == "table")
        Without indexes, Qdrant scans all payloads -> O(n) per filter.
        With indexes, filter is O(log n) for integer, O(1) for keyword hash.

        LATER -> Add text index on 'text' field for hybrid vector+keyword search.
        LATER -> Enforce payload schema types at collection creation for strict validation.
        """
        indexes = [
            ("page", "integer"), # For page range queries 
            ("section_heading", "keyword"), # Exact match like Heading -> Fuel system
            ("chunk_type", "keyword"), # retrieve only tables 
            ("source_doc", "keyword") # Used for multi-document filtering
        ]
        for field, schema in indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=schema
                )
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning("index_creation_failed", field=field, error=str(e))

    def hybrid_search(self, query_vector: List[float], query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Hybrid search: combine dense vector scores with sparse keyword scores.

        LATER -> Qdrant supports sparse vectors (BM25/SPLADE) alongside dense vectors.
        Implementation requires:
          1. Create collection with both dense + sparse vector configs
          2. Index sparse vectors (BM25 term weights) alongside dense embeddings
          3. Use query_points() with both vector types + RRF fusion
        This enables exact keyword matching (e.g., part numbers like "F110-GE-100")
        combined with semantic similarity. Critical for technical manual search.

        Reference: https://qdrant.tech/documentation/concepts/hybrid-queries/
        """
        raise NotImplementedError(
            "Hybrid search not yet implemented. "
            "LATER -> Add sparse vector support + RRF fusion. "
            "See: https://qdrant.tech/documentation/concepts/hybrid-queries/"
        )