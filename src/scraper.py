#!/usr/bin/env python3
"""
arXiv Scraper for the Agentic Deep-Research System.
Harvests research papers related to LLM agents and saves PDFs and metadata.
"""

import os
import sys
import json
import time
import re
import argparse
import logging
from typing import List, Dict, Any
import requests
from tqdm import tqdm
import arxiv

# Ensure parent directory is in system path to avoid ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging with premium formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("arxiv_scraper")

def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be a safe, clean Windows/Linux filename."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)  # remove illegal chars
    name = re.sub(r'\s+', "_", name)          # spaces to underscores
    return name.strip()[:100]                  # truncate length

def download_pdf_file(url: str, filepath: str) -> bool:
    """Downloads a PDF from a URL using requests with standard User-Agent headers."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            logger.error(f"Failed download with HTTP code: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Exception downloading PDF: {e}")
        return False

def harvest_arxiv_papers(
    query: str = 'ti:"LLM agent" OR ti:"ReAct agent" OR ti:"AI agent" OR abs:"deep research"',
    limit: int = 5,
    output_dir: str = "data"
) -> List[Dict[str, Any]]:
    """
    Queries arXiv and downloads matching papers along with metadata.
    """
    pdf_dir = os.path.join(output_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    
    logger.info(f"Initiating search with query: '{query}' (limit: {limit})")
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    metadata_list = []
    
    try:
        results = list(client.results(search))
        logger.info(f"Found {len(results)} matching papers. Starting download process...")
    except Exception as e:
        logger.error(f"Error querying arXiv: {e}")
        return []
    
    for result in tqdm(results, desc="Downloading papers"):
        paper_id = result.entry_id.split("/abs/")[-1].split("v")[0]
        safe_title = sanitize_filename(result.title)
        filename = f"{paper_id}_{safe_title}.pdf"
        filepath = os.path.join(pdf_dir, filename)
        
        # Prepare metadata structure
        paper_metadata = {
            "id": paper_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "published": result.published.isoformat() if result.published else None,
            "summary": result.summary,
            "pdf_url": result.pdf_url,
            "primary_category": result.primary_category,
            "local_pdf_path": filepath
        }
        
        metadata_list.append(paper_metadata)
        
        if os.path.exists(filepath):
            logger.info(f"Paper '{result.title}' already exists. Skipping download.")
            continue
            
        logger.info(f"Downloading: {result.title} -> {filepath}")
        
        # Call requests-based downloader
        success = download_pdf_file(result.pdf_url, filepath)
        if success:
            logger.info("Download completed successfully.")
            # Standard arXiv polite scraping delay
            time.sleep(2.0)
        else:
            logger.error(f"Failed to download PDF for '{result.title}'")
            
    # Save/append metadata.json
    metadata_path = os.path.join(output_dir, "metadata.json")
    
    # Merge with existing metadata if present
    existing_metadata = []
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                existing_metadata = json.load(f)
        except Exception:
            pass
            
    # Deduplicate metadata entries by paper ID
    meta_dict = {p["id"]: p for p in existing_metadata}
    for p in metadata_list:
        meta_dict[p["id"]] = p
        
    final_metadata = list(meta_dict.values())
    
    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(final_metadata, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully saved metadata for {len(final_metadata)} total papers in {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to write metadata file: {e}")
        
    return metadata_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harvest arXiv research papers.")
    parser.add_argument("--query", type=str, default='ti:"LLM agent" OR ti:"ReAct agent" OR ti:"AI agent"', help="arXiv search query query.")
    parser.add_argument("--limit", type=str, default="5", help="Number of papers to download.")
    parser.add_argument("--outdir", type=str, default="data", help="Output directory for papers and metadata.")
    
    args = parser.parse_args()
    
    limit_val = 1 if args.limit.lower() == "test" else int(args.limit)
    base_dir = os.path.abspath(args.outdir)
    
    logger.info("=========================================")
    logger.info("Starting arXiv Harvesting Operations")
    logger.info("=========================================")
    
    harvest_arxiv_papers(query=args.query, limit=limit_val, output_dir=base_dir)
    logger.info("Harvesting operations finished successfully.")
