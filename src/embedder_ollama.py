'''
Embedder (nomic-embed-text) use tokenizer (bpe) or else, gen usediff tokenizer -> token count mismatch
1000-token chunk per tiktoken -> 1200 for llama
# LATER -> use same model's tokenizer (check counts as well)
PRO -> Batching(Model vs latncy), Pull models via ollama cli, in docker container, Rate limiting handled by ollama server
Tenacity -> for retrying requests (exponential backoff)
# PRO -> use ollama asynch api 

Embedder selection: 
Make sure that model trained for MTEB/semantic similarity(STS) (MTEB score check) (for retrieval) tasks
Check if model expects flash_att, then your runtime supports it
Hidden size/ # layers/ #heads -> capacity vs latency tradeoff
(LATER) Max_trained positions/ max_pos_embeddings -> context length (nomic -> 2048 tokens)
(later) -> positional encoding type (rotary/absolute/relative) -> check model docs
Config.json -> Torch dtype/precision -> f16/ f32 (memory vs speed difference)
Use of optimized kernels (use_flash_attn) -> Speedup, but can be deployment headache...
Test model similarity -> Input length changing (robustness)
model.safetensors -> actual model weights (binary format)
tokenizer.json -> bpe merges/vocab + rules (hf tokenizer use this at runtime)
special tokens -> cls, sep, pad, unk, mask token ids
Must be under max_trained_positions of model, but can support longer due to rotary/nTk tricks (but quality may degrade)

'''

import requests
import json
from typing import List
from . import config
from .utils import logger
# HTTP POST to ollama embedding endpoint
# Nomic -> Not support batching...  (FIX --- USE LATER -> ELSE EMBEDDER)
class OllamaEmbedder:

    def __init__(self, model: str = None, batch_size: int = 1):
        self.model = model or config.EMBED_MODEL
        self.base_url = f"{config.OLLAMA_BASE_URL}/api/embed"
        self.batch_size = config.EMBED_BATCH_SIZE or batch_size # Must as ollama can't handle all at once
# List of texts -> list of embeddings (batch input), 
# LATER -> add batch size & cache additions

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            logger.warning("embed_batch_empty_input")
            return []

        total = len(texts)
        all_embeddings = []
        total_batches = (total + self.batch_size - 1) // self.batch_size

        logger.info("embedding_started", total_chunks=total, batch_size=self.batch_size)

        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1

            logger.info("embedding_batch", batch=batch_num, of=total_batches, size=len(batch))

            try:
                embeddings = self._embed_single_batch(batch)
                all_embeddings.extend(embeddings)

                logger.info("embedding_batch_success", batch=batch_num, embeddings=len(embeddings))

            except Exception as e:
                logger.error("embedding_batch_failed", batch=batch_num, error=str(e))
                # Don't die — try to continue with zeros or skip?
                # For POC: fail fast (better to know)
                raise RuntimeError(f"Embedding failed at batch {batch_num}/{total_batches}: {e}") from e

        if len(all_embeddings) != total:
            logger.error("embedding_count_mismatch", expected=total, got=len(all_embeddings))
            raise ValueError(f"Embedding mismatch: {len(all_embeddings)} vs {total}")

        logger.info("embedding_complete", total_embeddings=len(all_embeddings))
        return all_embeddings

    def _embed_single_batch(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model,
            "input": texts
        }

        try:
            resp = requests.post(self.base_url, json=payload, timeout=300)  # 5 min
            resp.raise_for_status()
        except requests.Timeout:
            raise RuntimeError("Ollama timed out (300s). Try smaller BATCH_SIZE.")
        except requests.ConnectionError:
            raise RuntimeError(f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}")
        except Exception as e:
            raise RuntimeError(f"HTTP error: {e}")

        try:
            data = resp.json()
        except json.JSONDecodeError:
            raise ValueError("Ollama returned invalid JSON")

        embeddings = data.get("embeddings")
        if not embeddings or not isinstance(embeddings, list):
            raise ValueError(f"Invalid embeddings response: {data}")

        # Optional: validate dimension
        if embeddings:
            first_dim = len(embeddings[0])
            if first_dim != config.EMBEDDING_DIM:
                logger.warning("embedding_dim_mismatch", got=first_dim, expected=config.EMBEDDING_DIM)

        return embeddings
