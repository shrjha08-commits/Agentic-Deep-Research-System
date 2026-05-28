# Agentic Deep-Research System

An end-to-end Deep-Research System that harvests academic research papers from arXiv, indexes them into a semantic vector store, runs plan-and-solve agent loops to answer complex comparative questions, and evaluates performance against traditional architectures using an automated **LLM-as-a-Judge** ablation harness.

---

## 🏗️ Architecture Overview

The system consists of four primary modular pipelines designed for high autonomy and scientific rigor:

```mermaid
graph TD
    A[arXiv API Search] -->|src/scraper.py| B[(Raw PDFs & Metadata)]
    B -->|src/indexer.py| C[(Local Vector DB)]
    C -->|Semantic Retrieval| D[Deep Research Agent]
    
    subgraph Reasoning Architectures (src/agent.py)
        D1[Naive RAG]
        D2[Plan-and-Solve Agent]
    end
    
    D --> D1
    D --> D2
    
    D1 -->|Ablated Output| E[Evaluator & LLM Judge]
    D2 -->|Agentic Output| E
    E -->|src/evaluator.py| F[Comparative Performance Report]
```

1. **arXiv Harvester (`src/scraper.py`)**: Queries arXiv's scientific database for papers on autonomous agents and planning loops, downloads raw PDFs with polite rate-limiting, and parses indexing metadata.
2. **Semantic Text Indexer (`src/indexer.py`)**: Page-by-page PDF extraction using `pypdf`, overlapping text partition splitting using a custom recursive parser, and batch vector generation utilizing the Gemini Embeddings API (`models/text-embedding-004`). Persists chunks inside a lightweight NumPy + JSON-backed local vector store.
3. **Deep Research Agent (`src/agent.py`)**: Supports:
   - **Naive RAG**: Direct vector search retrieval combined with single-shot synthesis.
   - **Plan-and-Solve Agent**: Formulates a detailed multi-step investigation plan, recursively invokes specialized local metadata/content retrieval tools to accumulate knowledge, and compiles a rigorous peer-review grade report.
4. **LLM-as-Judge Evaluator (`src/evaluator.py`)**: Reads a standard list of evaluation questions from `eval/questions.jsonl`, runs them under both Naive RAG and Plan-and-Solve configurations, outputs the comparative results, and runs a structured grading peer-review loop using Gemini to score performance across Correctness, Completeness, and Faithfulness.

---

## 🚀 Quick Start & Installation

### 1. Set Up Environment
First, ensure you have Python 3.9+ installed. Set your Gemini API Key in your shell environment:

**PowerShell (Windows):**
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

**Command Prompt (Windows):**
```cmd
set GEMINI_API_KEY=your-api-key-here
```

**Bash (Linux/macOS):**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### 2. Install Dependencies
Install all locked dependencies from the requirements file:
```bash
pip install -r requirements.txt
```

---

## ⚡ Single-Reproduction Command

To execute the entire end-to-end deep-research pipeline (Harvest 3 papers -> Index them semantically -> Run Plan-and-Solve vs. Naive RAG ablation -> Generate Peer-Review LLM scores) with a single command, run the following in your terminal:

**In PowerShell / Command Prompt / Bash:**
```bash
python src/scraper.py --limit 3 && python src/indexer.py && python src/evaluator.py
```

---

## 📂 Repository Structure

```
├── data/               # Raw downloaded PDFs and parsed metadata JSON
│   ├── pdfs/           # Saved research PDFs
│   └── vector_db/      # Serialized vector database index
├── src/
│   ├── scraper.py      # arXiv harvesting logic and API client
│   ├── indexer.py      # PDF text extractor, recursive chunker, and embedding generator
│   ├── agent.py        # LLM research loop, plan-and-solve planner, and tools
│   └── evaluator.py    # Ablation runner and LLM-as-judge evaluation logic
├── eval/               # Evaluation dataset folder
│   └── questions.jsonl # Starter research questions and expectation topics
├── predictions/        # Generated ablation predictions and evaluation logs
│   ├── predictions_naive.jsonl
│   └── predictions_agentic.jsonl
├── requirements.txt    # Strict version-locked third-party dependencies
└── README.md           # Getting started guide and pipeline instructions
```
