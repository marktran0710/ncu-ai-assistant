"""
Build the vector index from combined NCU course data.
Optimized with recursive character splitting for better RAG retrieval.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
# Assuming you have a standard splitter or can add one to core
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from core import LocalEmbedder, VectorIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR    = Path("data")
INDEX_FILE  = "ncu_index.pkl"
# Updated to use the combined file as requested
DEFAULT_FILE = DATA_DIR / "course_combined.jsonl"

EMBEDDER_MODELS = {
    "en": "all-MiniLM-L6-v2",
    "zh": "paraphrase-multilingual-MiniLM-L12-v2",
    "multi": "paraphrase-multilingual-MiniLM-L12-v2",
}

def load_and_split_docs(path: Path, chunk_size: int = 600, chunk_overlap: int = 60):
    """
    Loads JSONL and splits large 'text' fields into manageable chunks.
    'Sweet spot' for these models is usually 500-800 chars.
    """
    raw_docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_docs.append(json.loads(line))

    # Initialize the splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
    )

    final_chunks = []
    for doc in raw_docs:
        base_text = doc.get("text", "")
        if not base_text:
            continue
        
        # Split the text into smaller pieces
        splits = splitter.split_text(base_text)
        
        for i, split in enumerate(splits):
            # Keep original metadata but update text and add a chunk ID
            chunk = doc.copy()
            chunk["text"] = split
            chunk["chunk_id"] = f"{doc.get('course_id', 'unknown')}_{i}"
            final_chunks.append(chunk)
            
    return final_chunks

def build(jsonl_path: Path, lang: str = "multi") -> None:
    if not jsonl_path.exists():
        logger.error(f"File not found: {jsonl_path}")
        return

    logger.info(f"Processing {jsonl_path} with 'Sweet Spot' chunking...")
    
    # We use 'multi' by default for combined files to handle both EN and ZH
    model_name = EMBEDDER_MODELS.get(lang, EMBEDDER_MODELS["multi"])
    
    # Load and Split
    docs = load_and_split_docs(jsonl_path)
    
    if not docs:
        logger.error("No documents loaded.")
        return

    logger.info(f"Generated {len(docs)} chunks from original documents.")
    logger.info(f"Using embedding model: {model_name}")

    embedder = LocalEmbedder(model_name=model_name)
    index    = VectorIndex(embedder)
    
    logger.info("Building index...")
    index.build(docs)
    index.save(INDEX_FILE)
    
    logger.info(f"Index saved → {INDEX_FILE}")
    print(f"\n✓ Index ready: {INDEX_FILE}")
    print(f"  Total Chunks : {len(docs)}")
    print(f"  Model        : {model_name}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build NCU vector index")
    ap.add_argument("--lang", choices=["en", "zh", "multi"], default="multi",
                    help="Model language (default: multi)")
    ap.add_argument("--file", type=Path, default=DEFAULT_FILE,
                    help="Path to course_combined.jsonl")
    args = ap.parse_args()

    build(args.file, lang=args.lang)