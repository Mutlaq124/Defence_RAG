"""
Docling 2.63.0 (Nov 2025) — FINAL FIX
Extracts from 'texts', 'tables', 'pictures' keys (new API structure)
No more 'elements' key in export_to_dict()
LATER -> Removal of Header/footer (docling does automatically)
LATER -> add bobox cordinates (crucial for UI highlighting where PDF answer came from)
LATER ->Sort by page current -> elements in one page might be out of order (Docling cordinate's bbox -> use and sort by x,y axis)
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

from .utils import get_pdf_hash, save_json, logger, generate_deterministic_id
from . import config


def parse_docling(pdf_path: str, out_dir: Optional[str] = None) -> Dict[str, Any]:
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir or config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed_path = out_dir / f"{pdf_path.stem}_parsed.json"
    hash_path = out_dir / f"{pdf_path.stem}_hash.txt"
    # Cache by hash:
    if parsed_path.exists() and hash_path.exists():
        if hash_path.read_text().strip() == get_pdf_hash(str(pdf_path)):
            logger.info("using_cached_parsed_file", path=str(parsed_path))
            with parsed_path.open("r", encoding="utf-8") as fp:
                return json.load(fp)

    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            EasyOcrOptions,
            AcceleratorOptions,
            AcceleratorDevice,
            TableStructureOptions,
            TableFormerMode,
        )

        logger.info("docling_start", pdf=str(pdf_path), vlm_enabled=False)

        # === CONFIG ===
        pipeline_options = PdfPipelineOptions()
        
        pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=config.MAX_CORES, 
        device=AcceleratorDevice.CUDA )

        # OCR for scanned images
        pipeline_options.do_ocr = True # for images or scanned PDFs
        pipeline_options.ocr_options = EasyOcrOptions(lang=["en"])
        
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
            do_cell_matching=True
        )
        # Running on every page -> too slow
        # VLM Enabled Later -> ONLY ON FIGURES or ocred text)
        pipeline_options.do_picture_description = False  # Enable picture description (VLM)

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        result = converter.convert(str(pdf_path))
        doc = result.document

        if not doc:
            raise ValueError("Conversion returned empty document")

# # LATER -> Massive python object (can crash standard container for large PDFs)
# iterate over pages/elements directly from 'doc' object instead of exporting whole dict 
# EXPORT TO DICT ===
        raw_dict = doc.export_to_dict()
        
        # Log what keys we got
        logger.info("export_keys", keys=list(raw_dict.keys()) if isinstance(raw_dict, dict) else "not_dict")
        
        # Save raw output for debugging
        with (out_dir / "RAW_Docling_Output.json").open("w", encoding="utf-8") as fp:
            json.dump(raw_dict, fp, indent=2, ensure_ascii=False, default=str)

        # === EXTRACT ELEMENTS FROM NEW API STRUCTURE ===
        # Docling 2.63.0 structure:
        # {
        #   'texts': [...],      # Paragraphs and headings
        #   'tables': [...],     # Tables
        #   'pictures': [...],   # Figures/images
        #   'pages': [...]       # Page metadata
        # }
        elements: List[Dict[str, Any]] = []
        
        # === 1. EXTRACT TEXTS (Paragraphs and Headings) ===
        texts = raw_dict.get("texts", [])
        logger.info("texts_found", count=len(texts))
        
        for idx, text_item in enumerate(texts):
            try:
                # Text items have: text, label, prov (provenance with page info)
                text_content = text_item.get("text", "")
                if not text_content or not text_content.strip():
                    continue
                
                label = text_item.get("label", "").lower()
                # Skip headers and footers to eliminate micro-chunk noise
                if label in ["page_header", "page_footer"]:
                    continue
                prov = text_item.get("prov", {})
                
                # Extract page number from provenance
                page = None
                if isinstance(prov, dict):
                    page = prov.get("page_no")
                elif isinstance(prov, list) and len(prov) > 0:
                    page = prov[0].get("page_no") if isinstance(prov[0], dict) else None
                
                # Determine if heading or paragraph
                if "title" in label or "heading" in label or "section" in label:
                    elements.append({
                        "type": "heading",
                        "text": text_content.strip(),
                        "level": 1,  # Could parse from label if needed
                        "page": page
                        })
                else:
                    elements.append({
                        "type": "paragraph",
                        "text": text_content.strip(),
                        "page": page
                        })
            except Exception as e:
                logger.warning("text_extraction_failed", idx=idx, error=str(e))
                continue
        
        # === 2. EXTRACT TABLES ===
        tables = raw_dict.get("tables", [])
        logger.info("tables_found", count=len(tables))
        
        for idx, table_item in enumerate(tables):
            # Get markdown representation
            markdown = ""
            rows = []
            
            try:
                # Try to get data - can be dict or list
                data = table_item.get("data", None)
                
                if data:
                    # Check if data is a list of rows
                    if isinstance(data, list) and len(data) > 0:
                        rows = data
                        # Convert to markdown
                        try:
                            # Check if first row exists and is iterable
                            if len(data) > 0 and hasattr(data[0], '__iter__'):
                                header = " | ".join(str(cell) for cell in data[0])
                                separator = " | ".join("---" for _ in data[0])
                                body_rows = [" | ".join(str(cell) for cell in row) for row in data[1:]]
                                markdown = "\n".join([header, separator] + body_rows)
                        except Exception as e:
                            logger.warning("table_markdown_conversion_failed", idx=idx, error=str(e))
                    
                    # Check if data is a dict with 'grid' or 'cells'
                    elif isinstance(data, dict):
                        if "grid" in data and isinstance(data["grid"], list):
                            rows = data["grid"]
                        elif "cells" in data and isinstance(data["cells"], list):
                            rows = data["cells"]
                
                # Try getting text representation
                if not markdown:
                    text = table_item.get("text", "")
                    if text and text.strip():
                        markdown = text.strip()
                
                # Try export_to_markdown if item has the method
                if not markdown:
                    # Note: table_item is a dict, not an object with methods
                    # So this won't work, but keeping for safety
                    pass
                
            except Exception as e:
                logger.warning("table_extraction_failed", idx=idx, error=str(e))
                continue
            
            # Get page info
            prov = table_item.get("prov", {})
            page = None
            if isinstance(prov, dict):
                page = prov.get("page_no")
            elif isinstance(prov, list) and len(prov) > 0:
                page = prov[0].get("page_no") if isinstance(prov[0], dict) else None
            
            # Only add if we have some content
            if markdown or rows:
                content_for_id = markdown if markdown else str(rows)
                elements.append({
                    "type": "table",
                    "markdown": markdown,
                    "rows": rows,
                    "page": page
                    })
            else:
                logger.debug("empty_table_skipped", idx=idx)
        
        # === 3. EXTRACT PICTURES/FIGURES ===
        pictures = raw_dict.get("pictures", [])
        logger.info("pictures_found", count=len(pictures))
        
        for idx, picture_item in enumerate(pictures):
            try:
                # Get caption or description
                caption = picture_item.get("caption", "") or picture_item.get("text", "")
                
                # Get page info
                prov = picture_item.get("prov", {})
                page = None
                if isinstance(prov, dict):
                    page = prov.get("page_no")
                elif isinstance(prov, list) and len(prov) > 0:
                    page = prov[0].get("page_no") if isinstance(prov[0], dict) else None
                
                content_for_id = caption.strip() if caption else "empty_figure"
                elements.append({
                    "type": "figure",
                    "caption": caption,
                    "page": page
                })
            except Exception as e:
                logger.warning("picture_extraction_failed", idx=idx, error=str(e))
                continue
        
        # === FALLBACK: Use body items if no elements extracted ===
        if not elements:
            logger.warning("no_elements_from_export_trying_body")
            
            if hasattr(doc, "body") and hasattr(doc.body, "items"):
                for item in doc.body.items:
                    text = getattr(item, "text", "")
                    if text and text.strip():
                        item_type = str(getattr(item, "label", "paragraph")).lower()
                        page = getattr(item, "page_no", None)
                        
                        if "title" in item_type or "heading" in item_type:
                            elements.append({
                                "type": "heading",
                                "text": text.strip(),
                                "level": 1,
                                "page": page
                                })
                        else:
                            elements.append({
                                "type": "paragraph",
                                "text": text.strip(),
                                "page": page
                                })
        
        # === FINAL CHECK ===
        if not elements:
            raise ValueError(
                f"No elements extracted. "
                f"Found: {len(texts)} texts, {len(tables)} tables, {len(pictures)} pictures. "
                f"Check RAW_Docling_Output.json for details."
            )
        
        # === CANONICAL OUTPUT ===
        # keep id in doc_metadata...
        doc_ids = generate_deterministic_id(str(pdf_path), prefix="doc")
        canonical = {
            "doc_meta": {
                "title": raw_dict.get("name", pdf_path.stem),
                "doc_id": doc_ids["qdrant_id"],
                "source_doc": str(pdf_path.name),
                "pages": len(raw_dict.get("pages", []))
            },
            "elements": elements
        }
        
        out_path = out_dir / f"{pdf_path.stem}_parsed.json"
        save_json(str(out_path), canonical)
        
        # Log statistics
        element_types = {}
        for el in elements:
            el_type = el.get("type", "unknown")
            element_types[el_type] = element_types.get(el_type, 0) + 1
        
        logger.info("docling_success", 
                   total_elements=len(elements),
                   by_type=element_types,
                   output=str(out_path))
        
        hash_path.write_text(get_pdf_hash(str(pdf_path)))
        return canonical

    except Exception as e:
        logger.error("docling_failed", error=str(e))
        import traceback
        logger.error("traceback", trace=traceback.format_exc())
        raise


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/parser_docling.py data/HAF-F16.pdf")
        sys.exit(1)
    parse_docling(sys.argv[1])