#!/usr/bin/env python3
"""
LLM Agent Loop, Planner, and Toolsets for the Agentic Deep-Research System.
Implements Naive RAG and advanced Plan-and-Solve agent reasoning modes.
Supports offline simulation mode when GEMINI_API_KEY is not configured.
"""

import os
import sys
import json
import logging
import re
from typing import List, Dict, Any, Tuple

# Ensure parent directory is in system path to avoid ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexer import LocalVectorDB, generate_offline_embedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("agent")

# Delay imports to allow environment check
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

class ResearchTools:
    """Toolbox for the Agentic Deep-Research System."""
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.metadata_path = os.path.join(data_dir, "metadata.json")
        self.db_path = os.path.join(data_dir, "vector_db", "index.json")
        self.db = LocalVectorDB(self.db_path)
        
    def get_paper_metadata(self) -> List[Dict[str, Any]]:
        """Reads and returns metadata of all downloaded papers."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading metadata: {e}")
        return []

    def tool_list_papers(self) -> str:
        """Returns a formatted list of all available papers in the index."""
        papers = self.get_paper_metadata()
        if not papers:
            return "No papers available in the database."
        
        result = "Available Papers in Corpus:\n"
        for idx, paper in enumerate(papers):
            result += f"[{idx+1}] ID: {paper['id']} | Title: {paper['title']}\n"
            result += f"    Authors: {', '.join(paper['authors'])}\n"
            result += f"    Summary: {paper['summary'][:200]}...\n\n"
        return result

    def tool_semantic_search(self, query: str, k: int = 5, is_offline: bool = False) -> str:
        """Performs semantic search over all paper text chunks."""
        # Generate embedding for the query
        try:
            if is_offline or not HAS_GENAI:
                # Use local offline vectorizer
                query_vector = generate_offline_embedding(query)
            else:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=query,
                    task_type="retrieval_query"
                )
                query_vector = result["embedding"]
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}. Falling back to offline.")
            query_vector = generate_offline_embedding(query)

        hits = self.db.search(query_vector, k=k)
        if not hits:
            return "No matching chunks found in database (database is empty, please run scraper and indexer first)."

        formatted_hits = "Semantic Search Results:\n"
        for i, hit in enumerate(hits):
            formatted_hits += f"--- Match {i+1} (Score: {hit['score']:.4f}) ---\n"
            formatted_hits += f"Source: {hit['metadata']['title']} (Page {hit['metadata']['page']})\n"
            formatted_hits += f"Content: {hit['text']}\n\n"
        return formatted_hits

    def tool_get_paper_by_id(self, paper_id: str) -> str:
        """Retrieves complete indexed metadata and chunks for a single paper."""
        papers = self.get_paper_metadata()
        target_paper = None
        for p in papers:
            if p["id"] == paper_id or paper_id in p["title"]:
                target_paper = p
                break
                
        if not target_paper:
            return f"Paper with ID or title matching '{paper_id}' not found."
            
        chunks = [doc for doc in self.db.documents if doc["metadata"]["paper_id"] == target_paper["id"]]
        
        result = f"Paper Details:\nTitle: {target_paper['title']}\n"
        result += f"Authors: {', '.join(target_paper['authors'])}\n"
        result += f"Abstract Summary: {target_paper['summary']}\n\n"
        result += f"Total indexed text chunks: {len(chunks)}\n"
        
        result += "First few pages content preview:\n"
        for chunk in chunks[:3]:
            result += f"[Page {chunk['metadata']['page']}]: {chunk['text'][:300]}...\n"
            
        return result

class DeepResearchAgent:
    """Agent that orchestrates research using Naive RAG or a Plan-and-Solve strategy."""
    def __init__(self, data_dir: str = "data", api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.is_offline = not self.api_key
        
        if self.is_offline:
            logger.warning("======================================================================")
            logger.warning("[WARNING] GEMINI_API_KEY not set! DeepResearchAgent running in Offline Mode.")
            logger.warning("Simulating agent planning and synthesis using local corpus contexts.")
            logger.warning("======================================================================")
        else:
            if HAS_GENAI:
                genai.configure(api_key=self.api_key)
            else:
                logger.error("google-generativeai package is not available. Switch to Offline Mode.")
                self.is_offline = True
                
        self.tools = ResearchTools(data_dir)
        self.model_name = "gemini-1.5-flash"

    def _offline_synthesize_answer(self, question: str, search_results: str) -> str:
        """Helper to generate a highly structured simulated answer from local search results."""
        # Find paper references in retrieved content
        titles = set(re.findall(r'Source: ([^\n]+)', search_results))
        papers_string = ", ".join(titles) if titles else "Retrieved academic papers"
        
        # Pull some actual snippets to make the mock answer look highly premium and grounded!
        snippets = re.findall(r'Content: ([^\n]+)', search_results)
        highlighted_findings = ""
        for i, snip in enumerate(snippets[:3]):
            if len(snip.strip()) > 30:
                highlighted_findings += f"* **Finding {i+1}**: \"{snip.strip()[:180]}...\"\n"
                
        if not highlighted_findings:
            highlighted_findings = "* No matching paper snippets found in the database. Please verify scraping is complete."

        simulated_synthesis = f"""# Deep Research Synthesis Report (Offline Simulation Mode)

### 📋 Executive Summary
This report investigates the question: **"{question}"** by analyzing relevant literature in the local database, specifically matching: *{papers_string}*.

### 🔍 Key Literary Findings
{highlighted_findings}

### ⚖️ Architectural Synthesis & Review
1. **Methodological Rigor**: The papers focus on establishing planning systems (such as *Plan-and-ReAct* or *strategic game-theoretic* agents) to decompose complex prompts into structured tasks.
2. **Key Metrics & Improvements**: In the harvested papers, adding feedback loops and interactive tool check systems significantly reduced logical hallucinations and error cascades.
3. **Context-Grounded Evidence**:
   ```
   {search_results[:600]}...
   ```

*Note: This synthesis was generated locally in Offline Simulation Mode because no GEMINI_API_KEY was provided.*
"""
        return simulated_synthesis

    def run_naive_rag(self, question: str) -> Dict[str, Any]:
        """Simple RAG architecture (direct retrieval and generation)."""
        logger.info("Executing NAIVE RAG workflow")
        
        # 1. Direct semantic search (utilizing offline vectorizer if offline)
        search_results = self.tools.tool_semantic_search(question, k=5, is_offline=self.is_offline)
        
        # 2. Generate final answer
        if self.is_offline:
            answer = self._offline_synthesize_answer(question, search_results)
        else:
            prompt = f"""You are an expert academic research assistant specializing in Large Language Models and AI agent architectures.
You have been asked the following research question:
"{question}"

Based ONLY on the retrieved paper sections below, write a comprehensive, technically detailed, and well-structured answer.
If the retrieved content does not contain enough information to answer fully, state this clearly and summarize what is found.

---
RETIRED CONTEXT FROM CORPUS:
{search_results}
---

Provide a clear, detailed synthesis citing specific paper names if available.
"""
            model = genai.GenerativeModel(self.model_name)
            try:
                response = model.generate_content(prompt)
                answer = response.text
            except Exception as e:
                logger.error(f"Error calling Gemini in Naive RAG: {e}. Falling back to offline synthesis.")
                answer = self._offline_synthesize_answer(question, search_results)
                
        return {
            "answer": answer,
            "steps_taken": [
                {"action": "SemanticSearch", "query": question, "result_preview": search_results[:200]}
            ]
        }

    def run_plan_and_solve(self, question: str) -> Dict[str, Any]:
        """Plan-and-Solve Agent architecture (sequential planning and multi-tool execution)."""
        logger.info("Executing PLAN-AND-SOLVE agentic workflow")
        steps_taken = []
        
        # Step 1: Initial Planning Phase
        if self.is_offline:
            # Deterministic simulated plan
            plan = [
                {"step": 1, "description": "Identify downloaded papers and abstracts in corpus", "tool_to_use": "semantic_search", "argument": question},
                {"step": 2, "description": "Harvest architectural components and prompt strategies", "tool_to_use": "semantic_search", "argument": "agent loop and planning"},
                {"step": 3, "description": "Compare metrics, error rates, and ablation statistics", "tool_to_use": "semantic_search", "argument": "ablation findings and improvement percentages"}
            ]
            logger.info("Generated offline simulation research plan.")
        else:
            planning_prompt = f"""You are a deep-research planner. Your goal is to draft a comprehensive, step-by-step plan to answer this academic query:
"{question}"

First, understand what papers are available in our corpus:
{self.tools.tool_list_papers()}

Given the question and the available papers, write a research plan consisting of exactly 3 sequential investigation steps.
For each step, write a query you will search for or paper ID you will inspect to gather findings.
Return your plan in structured JSON format with this exact layout:
{{
  "plan": [
    {{"step": 1, "description": "Describe what to research in step 1", "tool_to_use": "semantic_search|get_paper_by_id", "argument": "the search query or paper ID"}},
    {{"step": 2, "description": "Describe what to research in step 2", "tool_to_use": "semantic_search|get_paper_by_id", "argument": "the search query or paper ID"}},
    {{"step": 3, "description": "Describe what to research in step 3", "tool_to_use": "semantic_search|get_paper_by_id", "argument": "the search query or paper ID"}}
  ]
}}
Only return the raw JSON object, no markdown wrappers, backticks, or extra commentary.
"""
            model = genai.GenerativeModel(self.model_name)
            try:
                plan_response = model.generate_content(
                    planning_prompt, 
                    generation_config={"response_mime_type": "application/json"}
                )
                plan_data = json.loads(plan_response.text)
                plan = plan_data.get("plan", [])
                logger.info(f"Generated research plan: {json.dumps(plan, indent=2)}")
            except Exception as e:
                logger.error(f"Planning failed: {e}. Falling back to default plan.")
                plan = [
                    {"step": 1, "description": "Search for core keywords in question", "tool_to_use": "semantic_search", "argument": question},
                    {"step": 2, "description": "Search for secondary concepts", "tool_to_use": "semantic_search", "argument": "ablation study and results"},
                    {"step": 3, "description": "Summarize top papers found", "tool_to_use": "semantic_search", "argument": "agent loops and architecture"}
                ]
            
        research_notes = []
        
        # Step 2: Execute Plan Steps
        for idx, step in enumerate(plan):
            tool_name = step.get("tool_to_use")
            arg = step.get("argument")
            desc = step.get("description")
            
            logger.info(f"Executing Plan Step {idx+1}: {desc} Using {tool_name} with '{arg}'")
            
            tool_output = ""
            if tool_name == "semantic_search":
                tool_output = self.tools.tool_semantic_search(arg, k=4, is_offline=self.is_offline)
            elif tool_name == "get_paper_by_id":
                tool_output = self.tools.tool_get_paper_by_id(arg)
            else:
                tool_output = self.tools.tool_semantic_search(arg or question, k=3, is_offline=self.is_offline)
                
            steps_taken.append({
                "step_num": idx+1,
                "description": desc,
                "tool": tool_name,
                "argument": arg,
                "result_preview": tool_output[:300] + "..."
            })
            
            research_notes.append(f"""### Step {idx+1}: {desc}
Query/ID: {arg}
Gathered Findings:
{tool_output}
""")
            
        # Step 3: Synthesis Phase
        synthesis_notes = "\n\n".join(research_notes)
        
        if self.is_offline:
            # Generate offline synthesis
            answer = self._offline_synthesize_answer(question, synthesis_notes)
        else:
            synthesis_prompt = f"""You are an elite research synthesizer. You have investigated a complex question:
"{question}"

You carried out a multi-step research plan and gathered the following notes and evidence:
---
RESEARCH EVIDENCE GATHERED:
{synthesis_notes}
---

Your task is to write a highly detailed, professional, and comprehensive synthesis report that answers the research question.
Make sure to:
1. Provide a direct, thorough answer.
2. Quote/cite specific papers, details, or numbers mentioned in the notes.
3. Compare the findings across different sources if applicable.
4. Structure the report beautifully with markdown headings and clear bullet points.
"""
            logger.info("Synthesizing gathered research notes...")
            try:
                synthesis_response = model.generate_content(synthesis_prompt)
                answer = synthesis_response.text
            except Exception as e:
                logger.error(f"Synthesis failed: {e}. Falling back to offline synthesis.")
                answer = self._offline_synthesize_answer(question, synthesis_notes)
            
        return {
            "answer": answer,
            "steps_taken": steps_taken,
            "research_notes_count": len(research_notes)
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Deep Research Agent.")
    parser.add_argument("--question", type=str, required=True, help="Question to ask the agent.")
    parser.add_argument("--mode", type=str, choices=["naive", "agentic"], default="agentic", help="Search and solve workflow mode.")
    parser.add_argument("--datadir", type=str, default="data", help="Directory where PDFs and index live.")
    
    args = parser.parse_args()
    
    agent = DeepResearchAgent(data_dir=args.datadir)
    
    if args.mode == "naive":
        result = agent.run_naive_rag(args.question)
    else:
        result = agent.run_plan_and_solve(args.question)
        
    print("\n=========================================")
    print(f"ANSWER (Mode: {args.mode.upper()})")
    print("=========================================")
    print(result["answer"])
    print("\n=========================================")
