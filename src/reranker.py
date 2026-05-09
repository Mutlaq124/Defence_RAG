'''
MS marco datasaet -> used for cross-encoder training and evaluation
Compute token lengths with reranker tokenizer before -> ensure within model limits
Q-chunk each pair process seperately (must be under 8192 tokens for bge-reranker-v2-m3)
)
Per token activation cost -> 1024*4=4096 (4kb), per pair tokens -> 6MB, batchsize 6 -> 36MB
1500 tokens (max_chunk+ 500 query) -> batch_size*1500 must fits memory (check memory)


bge reranker v2 m3-> Very heavy (use lighter one's with context)
'''

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict
import logging
from . import config

logger = logging.getLogger(__name__)

# # Quantization configuration for 8-bit loading
# quantization_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     llm_int8_threshold=6.0
# )

class CrossEncoderReranker:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading reranker model on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.RERANKER_MODEL,
            cache_dir=config.HF_CACHE_DIR, # Avoid re-downloading
            trust_remote_code=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.RERANKER_MODEL,
            cache_dir=config.HF_CACHE_DIR, # Avoid re-downloading
            trust_remote_code=True
            # quantization_config=quantization_config
        ).to(self.device)
# Pairs query with docs -> tokenize -> model run -> logit scores sort
# Moves encoded tensors to the appropriate device (CPU or GPU) for model inference.

    @torch.inference_mode()
    def rerank(self, query: str, docs: List[Dict], top_k=2):
        """
        docs = [{"text": "...", "metadata": {...}}, ...]
        returns sorted docs
        """
        pairs = [(query, d["text"]) for d in docs]

        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=8192, # Change it as per model limits
# combined query + doc length 512 for miniLM (using )
            return_tensors="pt"
        ).to(self.device)
# If output logits -> single score per pair (batch,1), if shape differ -> change accordingly
        scores = self.model(**encoded).logits.squeeze(-1)
        scores = scores.tolist()

        # Attach scores and sort
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)

        docs = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
        return docs[:top_k]
