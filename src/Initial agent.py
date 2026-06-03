from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json

@dataclass
class AgentState:
    user_query: str
    plan: List[str] = field(default_factory=list)
    retrieved_evidence: List[Dict[str, Any]] = field(default_factory=list)
    search_history: List[str] = field(default_factory=list)
    reflection_logs: List[str] = field(default_factory=list)
    final_answer: str = ""
    loop_count: int = 0
    max_loops: int = 4  # Cap to prevent infinite loops and save free-tier tokens

class DeepResearchAgent:
    def __init__(self, llm_client, vector_index, config_flags: Optional[Dict[str, bool]] = None):
        """
        config_flags allows turning components ON/OFF cleanly for ablation tests.
        Example: {"planner": True, "reflector": True, "verifier": True}
        """
        self.llm = llm_client
        self.index = vector_index
        
        # Default to full agentic setup if no flags provided
        self.config = config_flags or {
            "planner": True,
            "reflector": True,
            "verifier": True
        }

    def run(self, query: str) -> AgentState:
        """Executes the dynamic deep research loop end-to-end."""
        state = AgentState(user_query=query)
        
        # 1. Planning Phase
        if self.config["planner"]:
            state.plan = self._generate_plan(state.user_query)
        else:
            state.plan = [state.user_query]  # Ablation baseline fallback
            
        # 2. Dynamic Execution & Reflection Loop
        while state.loop_count < state.max_loops:
            state.loop_count += 1
            
            # Determine the best query for this round
            current_target_query = self._determine_next_query(state)
            state.search_history.append(current_target_query)
            
            # Call retrieval tools (Hybrid Lexical + Semantic Index search)
            new_passages = self.index.search(current_target_query, top_k=5)
            state.retrieved_evidence.extend(new_passages)
            
            # Reflection Phase
            if self.config["reflector"]:
                decision, reasoning = self._reflect(state)
                state.reflection_logs.append(reasoning)
                
                if decision == "STOP":
                    break
            else:
                # Ablation fallback: stop immediately after 1 round (non-agentic baseline)
                break

        # 3. Synthesis Phase
        state.final_answer = self._synthesize_answer(state)
        
        # 4. Citation Verification Phase
        if self.config["verifier"]:
            state.final_answer = self._verify_citations(state)
            
        return state

    def _generate_plan(self, query: str) -> List[str]:
        """Prompts LLM to break the query down into sub-questions."""
        prompt = f"Decompose this complex research question into 2-3 specific sub-questions: {query}"
        # Implement structured LLM calling logic returning a list of strings
        return [query] # Placeholder

    def _determine_next_query(self, state: AgentState) -> str:
        """Decides the query for the current loop iteration based on history."""
        if state.loop_count == 1:
            return state.plan[0]
        
        prompt = f"Original Query: {state.user_query}\nRetrieved info so far: {state.retrieved_evidence}\nWhat missing info should we search for next?"
        # Instruct LLM to emit exactly one concise text search query string
        return state.user_query # Placeholder

    def _reflect(self, state: AgentState) -> tuple[str, str]:
        """Evaluates whether gathered evidence answers the question completely."""
        prompt = (
            f"Question: {state.user_query}\n"
            f"Evidence Gathered: {state.retrieved_evidence}\n"
            f"Respond in JSON format with two keys:\n"
            f"'decision': 'STOP' or 'CONTINUE',\n"
            f"'reasoning': 'Why the data is or isn't sufficient yet.'"
        )
        # Parse JSON response from the LLM
        # return response["decision"], response["reasoning"]
        return "STOP", "Evidence sufficient." # Placeholder

    def _synthesize_answer(self, state: AgentState) -> str:
        """Drafts the final response mapping findings strictly to arXiv citation IDs."""
        prompt = f"Answer the question: {state.user_query} using ONLY this evidence: {state.retrieved_evidence}. Use inline [arXiv:XXXX.XXXX] citations."
        return "Synthesized text..." # Placeholder

    def _verify_citations(self, state: AgentState) -> str:
        """Cross-checks every inline citation against the retrieved source texts."""
        # Parse citations from final_answer, map back to text chunks, and drop unsupported assertions
        return state.final_answer
