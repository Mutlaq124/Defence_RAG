'''
Sentence Splitting Strategies: (split sentence -> token counts per sentence -> sliding window chunking)
Regex -> split() -> splits on delimiters (whitespace, punctuation)
Faster, Use with regex for better splits (Can use sentence based regex for better accuracy)
spacy can be used as well.

punkt (NLTK) (depreciated in newer nltk versions -> use punkt_tab) -> unsupervised ML model for sentence boundary detection
Cons -> needs to download punkt data, slower than simple split, fails on ocr, domain-specific text (legal, medical)
Struggle with langs lacking spaces (Chinese, Japanese), or poor OCR quality (oversplit or undersplit)

HF/tokenziers -> pretrained models for sentence splitting (can use sub-word tokenization)
Cons -> Slower, needs model download
Tokenizer(indirect) -> (tiktoken) -> tokens counts to measure chunk sizes(used for model inputs)


NLTK (sent_tokenize) -> sentence level tokenization (language aware)
High accuracy, ML-based, pretrained models for sentence boundary detection
Slow, but can cached

Tokenization libraries (tiktoken, HF transformers) -> professional 
Can use bpe or wordpiece tokenization
Tokenzier (must be same of generator/embedder model, because token counts differ, there's limitation on max tokens per model input)

Tokenization: (Pro):
 Normalized first (unicode, lowercasing, punctuation handling)-> batch -> cache -> sentence piece(handle multi-lingual)
Custom vocab -> BPE train on corpus, Code -> byte-level(handle variables, special chars)

IF model trained on diff tokenizer -> mismatch (subword, bpe, -> used mostly)

Buffering -> accumulate text under current heading before chunking/flushing
preserves hierarchy -> Long section splits into overlapping chunks with shared heading in metadata
for short chunks -> too small, so combine with heading context 
Figures -> caption + image embed pipeline for pro

Flat Chunking(current) -> convert doc elements -> chunks (each has metadata, like section heading, page number, element id)
# extend above, by storing parent_id for each chunk

Hierarchical -> levels (0 -> full doc meta, 1 -> section, 2 -> subsection, 3 -> intra-element chunks like table parts/figure caption)
# Embed both parent and child chunks for better retrieval context (more compute/storage), Output-> tree (json with parent-child links)
# long docs, manuals, with nested, legal/medical 
# LATER -> ADD Parent metadata to each chunk (section, subsection) (source_doc)
# LATER -> store parent-child relationships in Qdrant payloads for hierarchical retrieval
#  LATER -> Cache chunking, multilingual chunking (language detection per element -> use language specific sentence tokenizer)
# LATER -> tag tables/figures with ele_id, page, & also heading elements with (h1,h2,h3) levels for better context during retrieval
# LATER -> (offline) spacy -> Predownload at build time and include in docker image for faster startup
LATER -> For larger data -> don't save to disk, like raw docling output (only save to memory/db)
LATER -> parallel chunking for large docs
LATER -> add source_doc in all metadata for traceability (filter by document during retrieval)
LATER -> Cache chunks by file hash to avoid re-chunking unchanged documents

LATER -> parllel chunking for large docs (multiprocessing, threading, async)

LATER -> multilingual chunking (language specific tokenizers based on detected language per element)
LATER -> Metadata enrichment (add source_doc, hierarchical context like section/subsection headings to chunk metadata for better retrieval context)
LATER -> LLM will fetch prev_id, or next_id to fetch neighboring chunks for context during generation (if needed)
Later -> Parent heading hierarchy addition (h1,h2,h3) levels for better context during retrieval
later -> add mini_chunk_tokens but merge small chunks with previous, instead of dropping
later -> type hints addition or removal (explore)

'''

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from . import config
#using spacy for sentence splitting (slow but accurate)
import hashlib
import uuid

from .utils import (
    generate_deterministic_id,
    load_json,
    save_json,
    logger,
    count_tokens,
    sliding_window_token_indices,
)
# Use tokenizers custom -> for non-english as well
# Sentence tokenizer (NLTK) -> Text split to sentences,
# Word_tokenize -> word level (not used here) (more granular)

from .utils import split_into_sentences
# sentences = split_into_sentences(text) # can't do in module import

#  LATER -> Structured rows (Lists of lists)-> useful for precise table search or numeric extraction
# MD -> LLM/embedding models understand better for text-based similarity (store both)
# Extract text from element for chunking (output text based on type,or md for table)
def element_text_for_chunk(elem: Dict[str, Any]) -> str:
    etype = elem.get("type", "paragraph")
    if etype == "table":
        # prefer markdown for tables (LATER -> convert rows to markdown if needed)
        md = elem.get("markdown")
        if md:
            return md
        rows = elem.get("rows")
        if rows:
            logger.warning("table_without_markdown", using_raw_rows=True)
            return str(rows)  # Let embedding model handle raw structure
        else:
            return ""   # empty table
    elif etype == "figure":

        caption = elem.get("caption", "").strip()
        page = elem.get("page")
        if caption:
            text = f"Figure: {caption} (Page {page})"
        else:
            text = f"Figure on page {page} — visual diagram or illustration"
        return text
    else:
        return elem.get("text", "")
# Split table element if too large - returns list of table-chunk dicts
# output list of table chunks 
# LATER -> header row extract & include in each chunk for context

def split_table_if_needed(elem: Dict[str, Any], max_tokens: int) -> List[Dict[str, Any]]:
    
    md = element_text_for_chunk(elem)
    tokens = count_tokens(md)

    if tokens <= max_tokens:
        ids = generate_deterministic_id(md, {"page": elem.get("page"), "element_id": elem.get("id")}, "table")
        return [{
            "chunk_id": ids["chunk_id"],
            "qdrant_id": ids["qdrant_id"],
            "text": md,
            "tokens": tokens,
            "source_doc": elem.get("source_doc"),
            "chunk_type": "table",
            "metadata": {
                "page": elem.get("page"),
                "element_id": elem.get("id"),
            },
            "source_elem": elem
        }]
    # Split rows approach
    rows = elem.get("rows") or []
    if not rows:
        # fallback: split markdown by lines
        lines = md.splitlines()
    else:
# LATER -> create MD table with header detection
        # render each row to a line and treat similarly
        lines = [" | ".join(map(str, r)) for r in rows]
    # Get Rows -> per-line token counts -> use sliding window -> build sub-chunks (add split index to metadata) 
    # create sliding windows by approximate token counts
    line_tok = [count_tokens(l) for l in lines]
    # windows ensure each chunk <= max_tokens with overlap
    windows = sliding_window_token_indices(line_tok, max_tokens, config.CHUNK_OVERLAP_TOKENS)
    chunks = []
    for idx, (s, e) in enumerate(windows):
        block_lines = lines[s:e]
        text = "\n".join(block_lines)
        ids = generate_deterministic_id(text, {"page": elem.get("page"), "element_id": elem.get("id"), "split_index": idx}, "table")
        chunks.append({
            "chunk_id": ids["chunk_id"],
            "qdrant_id": ids["qdrant_id"],
            "text": text,
            "tokens": count_tokens(text),
            "source_doc": elem.get("source_doc"),
            "chunk_type": "table",
            "metadata": {"page": elem.get("page"), "element_id": elem.get("id"), "split_index": idx},
            "source_elem": elem
        })
    return chunks

# Chunk paragraph text into smaller chunks based on sentence boundaries
# (sentence splitting instead of arbitrary splits)
def chunk_paragraph_text(text: str, meta: Dict[str, Any], max_tokens: int, overlap: int) -> List[Dict[str, Any]]:
    if not text or not text.strip():
        return []
    sentences = split_into_sentences(text)
    # count tokens per sentence
    sent_token_counts = [count_tokens(s) for s in sentences]
    # If there are no sentences (shouldn't happen), fallback to whole text
    if not sentences:
        ids = generate_deterministic_id(text, meta, "chunk")
        return [{
            "chunk_id": ids["chunk_id"],
            "qdrant_id": ids["qdrant_id"],
            "text": text,
            "source_doc": meta.get("source_doc"),
            "tokens": count_tokens(text),
            "chunk_type": "text",
            "metadata": meta.copy() # avoid mutation
        }]

    windows = sliding_window_token_indices(sent_token_counts, max_tokens, overlap)
    chunks = []
    # Join sentences in each window to form chunk text
    for idx, (s_idx, e_idx) in enumerate(windows):
        chunk_sentences = sentences[s_idx:e_idx]
        chunk_text = " ".join(chunk_sentences).strip()
        if not chunk_text:
            continue
        
        token_count = count_tokens(chunk_text)
        if token_count < config.MIN_CHUNK_TOKENS:
            logger.debug("chunk_too_small", tokens=token_count, min=config.MIN_CHUNK_TOKENS)
            continue 
        ids = generate_deterministic_id(chunk_text, meta, "chunk")
        chunks.append({
            "chunk_id": ids["chunk_id"],
            "qdrant_id": ids["qdrant_id"],
            "text": chunk_text,
            "tokens": count_tokens(chunk_text),
            "chunk_type": "text",
            "source_doc": meta.get("source_doc"),
            "metadata": {
                **meta.copy(),  # avoid mutation (LATER-> ROBUST)
                "split_index": idx,
                "sentence_span": (s_idx, e_idx)
            }
        })
    return chunks
# output: list of chunk dicts with keys: chunk_id, text, tokens, chunk_type, metadata (dict)

# LATER -> Buffer overflow risk (Too many paragraphs under heading-> buffer grows too large & chunk all at once, can be slow)

def chunk_document(parsed: Dict[str, Any], max_tokens: Optional[int] = None, overlap: Optional[int] = None) -> List[Dict[str, Any]]:
    """Simplified chunker with clearer logic."""
    max_tokens = max_tokens or config.MAX_CHUNK_TOKENS
    overlap = overlap or config.CHUNK_OVERLAP_TOKENS
    # Each chunk must know the source document
    source_doc = parsed["doc_meta"].get("source_doc") or parsed["doc_meta"]["source"]
    elements = sorted(parsed.get("elements", []), key=lambda e: (e.get("page") or 0))
    chunks = []
    
    current_heading = "root"
    current_page = None
    text_buffer = []  # Accumulate paragraph texts
    
    def flush_buffer():
        if not text_buffer:
            return
        joined = "\n\n".join(text_buffer)
        # Drop accumulated text if it's purely noise/too small
        if count_tokens(joined) < config.MIN_CHUNK_TOKENS:
            text_buffer.clear()
            return
            
        meta = {"section_heading": current_heading,
                "page": current_page if current_page is not None else "unknown",
                "source_doc": source_doc
                }
        # Create chunks from accumulated paragraph text
        for c in chunk_paragraph_text(joined, meta, max_tokens, overlap):
            chunks.append(c)
        text_buffer.clear()
    # Loop through elements
    for elem in elements:
        etype = elem.get("type", "paragraph")
        elem_page = elem.get("page")

        if elem_page is not None:
            current_page = elem_page

        if etype == "heading":
            flush_buffer()
            current_heading = elem.get("text", "").strip() or "section"
            
        elif etype == "paragraph":
            text = element_text_for_chunk(elem)
            if text.strip():
                text_buffer.append(text)
            # Addition -> prevent huge paragraph 
            # if sum(count_tokens(t) for t in text_buffer) > max_tokens * 1.5:
            #     flush_buffer()

        elif etype == "table":
            flush_buffer()
            for tc in split_table_if_needed(elem, max_tokens):
                tc["metadata"]["section_heading"] = current_heading
                if current_page is not None:
                    tc["metadata"]["page"] = current_page
                chunks.append(tc)
                
        elif etype in ("figure", "image"):
            flush_buffer()
            text = element_text_for_chunk(elem)
            ids = generate_deterministic_id(
                text, 
                {"section_heading": current_heading, "page": current_page if current_page is not None else "unknown"},
                  "figure")
            chunks.append({
                "chunk_id": ids["chunk_id"],
                "qdrant_id": ids["qdrant_id"],
                "text": text,
                "tokens": count_tokens(text),
                "source_doc": source_doc,
                "chunk_type": "figure",
                "metadata": {"section_heading": current_heading,
                             "page": current_page if current_page is not None else "unknown",}
            })
        else:  # Addition: Fallback for unkown types (treat as paragraph)
            text = element_text_for_chunk(elem)
            if text.strip():
                text_buffer.append(text)
    
    flush_buffer()
    
    # Save and log
    save_json(str(Path(config.DATA_DIR) / "chunks.json"), chunks)
    logger.info("chunking_complete", total_chunks=len(chunks))
    return chunks

# CLI run
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/chunker.py /path/to/parsed_doc.json")
        raise SystemExit(1)
    parsed_path = sys.argv[1]
    parsed = load_json(parsed_path)
    chunk_document(parsed)
