#!/usr/bin/env python3
"""
Indexer for the Agentic Deep-Research System.
Extracts text from downloaded PDFs, chunks it, generates embeddings,
and indexes it into a simple local vector database.
"""

import os
import sys
import json
import logging
import argparse
import hashlib
import re
from typing import List, Dict, Any, Tuple
import pypdf
import numpy as np
from tqdm import tqdm

# Ensure parent directory is in system path to avoid ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("indexer")

# Delay imports to allow environment check
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

class RecursiveCharacterTextSplitter:
    """A robust built-in recursive character chunk splitter to avoid external dependencies."""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        return self._split(text, self.separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        # Find best separator
        separator = separators[-1]
        for sep in separators[:-1]:
            if sep in text:
                separator = sep
                break

        chunks = []
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        current_doc = []
        current_len = 0

        for split in splits:
            split_len = len(split)
            if split_len > self.chunk_size:
                if current_doc:
                    chunks.append(separator.join(current_doc))
                    current_doc = []
                    current_len = 0
                sub_chunks = self._split(split, [s for s in separators if s != separator])
                chunks.extend(sub_chunks)
            elif current_len + split_len + (len(separator) if current_doc else 0) > self.chunk_size:
                chunks.append(separator.join(current_doc))
                overlap_doc = []
                overlap_len = 0
                for doc in reversed(current_doc):
                    if overlap_len + len(doc) + (len(separator) if overlap_doc else 0) <= self.chunk_overlap:
                        overlap_doc.insert(0, doc)
                        overlap_len += len(doc) + len(separator)
                    else:
                        break
                current_doc = overlap_doc
                current_doc.append(split)
                current_len = sum(len(d) for d in current_doc) + (len(separator) * (len(current_doc) - 1))
            else:
                current_doc.append(split)
                current_len += split_len + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            chunks.append(separator.join(current_doc))

        return [c.strip() for c in chunks if c.strip()]

class LocalVectorDB:
    """An elegant, robust local vector database implemented using NumPy and JSON serialization."""
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
        self.load()

    def add_documents(self, chunks: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Adds text chunks, metadatas, and their corresponding embedding vectors to the DB."""
        for text, meta, emb in zip(chunks, metadatas, embeddings):
            self.documents.append({
                "text": text,
                "metadata": meta
            })
            self.embeddings.append(emb)

    def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Calculates cosine similarity and returns top k matching chunks."""
        if not self.embeddings:
            return []
        
        q_arr = np.array(query_vector)
        emb_arr = np.array(self.embeddings)
        
        dot_product = np.dot(emb_arr, q_arr)
        norm_emb = np.linalg.norm(emb_arr, axis=1)
        norm_q = np.linalg.norm(q_arr)
        
        similarities = dot_product / (norm_emb * norm_q + 1e-10)
        
        # Guard against index out of range if database has fewer than k elements
        actual_k = min(k, len(self.documents))
        if actual_k <= 0:
            return []
            
        top_indices = np.argsort(similarities)[::-1][:actual_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": self.documents[idx]["text"],
                "metadata": self.documents[idx]["metadata"],
                "score": float(similarities[idx])
            })
        return results

    def save(self):
        """Persists the database to a JSON file."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings
        }
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Database saved successfully to {self.index_path} ({len(self.documents)} chunks indexed).")

    def load(self):
        """Loads database from disk if it exists."""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.documents = data.get("documents", [])
                self.embeddings = data.get("embeddings", [])
                logger.info(f"Loaded {len(self.documents)} indexed chunks from {self.index_path}")
            except Exception as e:
                logger.error(f"Error loading index file: {e}. Starting with an empty database.")

def generate_offline_embedding(text: str, dim: int = 768) -> List[float]:
    """
    Generates a deterministic 768-dimensional token hashing vector.
    Enables surprisingly capable offline keyword similarity matching!
    """
    vector = np.zeros(dim)
    words = re.findall(r'\w+', text.lower())
    if not words:
        return [0.0] * dim
        
    for word in words:
        # Generate stable index in [0, dim-1]
        hash_val = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
        idx = hash_val % dim
        vector[idx] += 1.0
        
    # L2 Normalization
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
        
    return vector.tolist()

def extract_pdf_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """Extracts text page-by-page from a PDF file."""
    pages = []
    try:
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages.append((page_idx + 1, text))
    except Exception as e:
        logger.error(f"Failed to read PDF {pdf_path}: {e}")
    return pages

def populate_vector_db(
    data_dir: str = "data",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    api_key: str = None
):
    """
    Reads local PDFs, chunks their text, calls Gemini API to embed them,
    and updates the local vector store. Falls back to deterministic hash vectors offline.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    is_offline = not api_key
    
    if is_offline:
        logger.warning("======================================================================")
        logger.warning("[WARNING] GEMINI_API_KEY not set! Switching to local Offline Mode.")
        logger.warning("Using high-performance Local Hashing Vectorizer for indexing.")
        logger.warning("======================================================================")
    else:
        if HAS_GENAI:
            genai.configure(api_key=api_key)
        else:
            logger.error("google-generativeai package is not available. Falling back to Offline Mode.")
            is_offline = True

    metadata_path = os.path.join(data_dir, "metadata.json")
    db_path = os.path.join(data_dir, "vector_db", "index.json")
    
    if not os.path.exists(metadata_path):
        logger.error(f"No metadata.json found at {metadata_path}. Please run the scraper first.")
        return
        
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)
        
    db = LocalVectorDB(db_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Track which paper IDs are already in the DB to avoid re-indexing
    indexed_papers = {doc["metadata"]["paper_id"] for doc in db.documents}
    
    for paper in metadata_list:
        paper_id = paper["id"]
        pdf_path = paper["local_pdf_path"]
        
        if paper_id in indexed_papers:
            logger.info(f"Paper '{paper['title']}' ({paper_id}) already indexed. Skipping.")
            continue
            
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF file for '{paper['title']}' not found at {pdf_path}. Skipping.")
            continue
            
        logger.info(f"Extracting text from {pdf_path}...")
        pages = extract_pdf_pages(pdf_path)
        
        chunks = []
        metadatas = []
        
        for page_num, page_text in pages:
            split_chunks = text_splitter.split_text(page_text)
            for chunk_idx, chunk in enumerate(split_chunks):
                chunks.append(chunk)
                metadatas.append({
                    "paper_id": paper_id,
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "page": page_num,
                    "chunk_id": f"{paper_id}_p{page_num}_c{chunk_idx}"
                })
                
        if not chunks:
            logger.warning(f"No text extracted from '{paper['title']}'. Skipping embeddings.")
            continue
            
        logger.info(f"Generating embeddings for {len(chunks)} chunks of '{paper['title']}'...")
        
        embeddings = []
        
        if is_offline:
            # Generate deterministic offline hash embeddings
            for chunk in tqdm(chunks, desc="Generating offline vectors"):
                embeddings.append(generate_offline_embedding(chunk))
        else:
            # Live Gemini API indexing in batches of 50
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                try:
                    result = genai.embed_content(
                        model="models/text-embedding-004",
                        content=batch_chunks,
                        task_type="retrieval_document"
                    )
                    embeddings.extend(result["embedding"])
                except Exception as e:
                    logger.error(f"Error calling live Gemini embedding API: {e}. Falling back to local offline vectors.")
                    for chunk in batch_chunks:
                        embeddings.append(generate_offline_embedding(chunk))
                        
        db.add_documents(chunks, metadatas, embeddings)
        db.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract, chunk, and embed research papers.")
    parser.add_argument("--datadir", type=str, default="data", help="Directory where PDFs and metadata live.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Target chunk size in characters.")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between consecutive chunks.")
    
    args = parser.parse_args()
    
    logger.info("=========================================")
    logger.info("Starting Semantic Indexing Operations")
    logger.info("=========================================")
    
    populate_vector_db(
        data_dir=args.datadir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap
    )
    logger.info("Indexing operations finished successfully.")
