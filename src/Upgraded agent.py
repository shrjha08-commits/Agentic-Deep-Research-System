import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# Setup dry, professional engineering logs instead of relying on print statements
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("DeepResearchAgent")

@dataclass
class AgentState:
    user_query: str
    target_objectives: List[str] = field(default_factory=list)
    retrieved_context_pool: List[Dict[str, Any]] = field(default_factory=list)
    execution_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    critic_audit_logs: List[Dict[str, Any]] = field(default_factory=list)
    final_compiled_synthesis: str = ""
    iteration_depth: int = 0
    max_allowable_depth: int = 4

class LeadInvestigatorAgent:
    """Responsible for original problem decomposition and subsequent targeted query generation."""
    def __init__(self, llm_client):
        self.llm = llm_client

    \n    def formulate_initial_objectives(self, query: str) -> List[str]:
        prompt = f"""[INSTRUCTION] Analyze the following research query and break it down into an atomic, ordered list of distinct technical objectives or sub-questions necessary to fulfill the overarching request. Output a clean JSON array of strings only. Do not include markdown code block styling.

[RESEARCH QUERY]
{query}

[JSON OUTPUT FORMAT]
["objective 1", "objective 2"]"""
        try:
            response = self.llm.generate(prompt, temperature=0.1)
            clean_res = response.replace("```json", "").replace("
```", "").strip()
            return json.loads(clean_res)
        except Exception as e:
            logger.warning(f"Planner failed to parse structured objectives. Falling back to raw query. Error: {e}")
            return [query]

    def refine_search_target(self, state: AgentState) -> str:
        """Determines the exact next text search target based on what has been accomplished so far."""
        current_idx = min(state.iteration_depth, len(state.target_objectives) - 1)
        active_objective = state.target_objectives[current_idx] if state.target_objectives else state.user_query
        
        prompt = f"""[CONTEXT] You are executing a research loop. 
Original Query: {state.user_query}
Current Target Sub-Objective: {active_objective}
Search History & Trajectory: {json.dumps(state.execution_trajectory, indent=2)}

[TASK] Generate a highly targeted keyword search string optimized to locate specific literature fragments answering the current objective. Do not wrap in quotes or code formatting. Return only the raw string.

Search String:"""
        return self.llm.generate(prompt, temperature=0.2).strip()


class AdversarialCriticAgent:
    """Acts as a rigorous quality gatekeeper, challenging the validity, completeness, and grounding of current research state."""
    def __init__(self, llm_client):
        self.llm = llm_client

    def audit_evidence(self, state: AgentState) -> Tuple[str, str]:
        """
        Evaluates the existing context pool against the user's target objective.
        Returns a decision ('PASS' or 'REJECT') and a dry critique.
        """
        prompt = f"""[CRITICAL EVALUATION SYSTEM]
You are an unyielding, strict peer reviewer auditing a deep-research trajectory. Your primary goal is to surface omissions, shallow claims, and contextual gaps.

[USER QUERY]
{state.user_query}

[ACCUMULATED RETRIEVED SEGMENTS]
{json.dumps(state.retrieved_context_pool, indent=2)}

[TASK]
Evaluate if the gathered evidence is rigorously sufficient to construct an absolute, comprehensive scientific response to the query without relying on parametric extrapolation or hand-waving.
You must respond with a strict JSON object structure. Do not use markdown blocks.

{{
    "status": "PASS" or "REJECT",
    "critique": "If REJECT, itemize exactly what concrete data points, metrics, or comparisons are missing from the snippets. If PASS, provide an engineering summary of the verified data points."
}}

Output:"""
        try:
            response = self.llm.generate(prompt, temperature=0.0)
            clean_res = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_res)
            return data.get("status", "PASS"), data.get("critique", "")
        except Exception as e:
            logger.error(f"Adversarial Critic failed to emit valid JSON: {e}")
            return "PASS", "Fallback override due to parsing collision."


class DeepResearchAgent:
    """The central state engine orchestrating multi-agent state passings via an event graph pattern."""
    def __init__(self, llm_client, vector_index, config_flags: Optional[Dict[str, bool]] = None):
        self.index = vector_index
        self.config = config_flags or {"planner": True, "critic": True, "verifier": True}
        
        # Initialize internal agent roles
        self.investigator = LeadInvestigatorAgent(llm_client)
        self.critic = AdversarialCriticAgent(llm_client)
        self.llm = llm_client

    def run(self, query: str) -> AgentState:
        state = AgentState(user_query=query)
        logger.info(f"Initiating research protocol for query: '{query}'")

        # Phase 1: Objective Mapping
        if self.config["planner"]:
            state.target_objectives = self.investigator.formulate_initial_objectives(state.user_query)
            logger.info(f"Decomposed query into {len(state.target_objectives)} targets: {state.target_objectives}")
        else:
            state.target_objectives = [state.user_query]

        # Phase 2: Execution Graph Routing Loop
        while state.iteration_depth < state.max_allowable_depth:
            state.iteration_depth += 1
            logger.info(f"Executing Exploration Step {state.iteration_depth}/{state.max_allowable_depth}")

            # Generate target search text string
            search_query = self.investigator.refine_search_target(state)
            
            # Interact with the data retriever engine
            # Uses the dynamic hybrid toggle parameter we implemented inside src/indexer.py
            is_hybrid = getattr(self.index, "search_hybrid_toggle", True)
            chunks = self.index.search(search_query, top_k=5, hybrid=is_hybrid)
            
            # Deduplicate incoming entries on the fly based on unique identifier blocks
            existing_ids = {item["id"] for item in state.retrieved_context_pool}
            new_chunks = [c for c in chunks if c["id"] not in existing_ids]
            state.retrieved_context_pool.extend(new_chunks)
            
            state.execution_trajectory.append({
                "iteration": state.iteration_depth,
                "generated_query": search_query,
                "newly_discovered_chunks": len(new_chunks)
            })

            # Multi-Agent Consensus Verification Step (Dynamic Gating)
            if self.config["critic"]:
                status, critique = self.critic.audit_evidence(state)
                state.critic_audit_logs.append({"iteration": state.iteration_depth, "status": status, "critique": critique})
                logger.info(f"Critic Audit Resolution -> Status: {status} | Focus: {critique[:60]}...")
                
                if status == "PASS" and state.iteration_depth >= len(state.target_objectives):
                    logger.info("Adversarial Critic issued clear pass authorization. Exiting loop.")
                    break
            else:
                # Bypassing the loop reflector completely simulates a classic single-turn baseline architecture
                break

        # Phase 3: Rigid Evidence-Locked Synthesis
        state.final_compiled_synthesis = self._synthesize_grounded_response(state)
        
        # Phase 4: Final Citation Cross-Verification Routing
        if self.config["verifier"]:
            state.final_compiled_synthesis = self._execute_hard_citation_alignment(state)

        return state

    def _synthesize_grounded_response(self, state: AgentState) -> str:
        prompt = f"""[ACADEMIC SYNTHESIS PROTOCOL]
Synthesize a formal response addressing the primary objective: {state.user_query}
You must construct your arguments using strictly verified facts from the context segments below.

[CONTEXT SEGMENTS]
{json.dumps(state.retrieved_context_pool, indent=2)}

[STRICT RESTRAINTS]
- Rely only on facts directly asserted within the segments.
- Append inline citations mapped to the exact origin document source. Use the pattern [arXiv:XXXX.XXXX].
- Do not introduce filler text, grandiose descriptions, or unverified claims.

Synthesis:"""
        return self.llm.generate(prompt, temperature=0.1)

    def _execute_hard_citation_alignment(self, state: AgentState) -> str:
        prompt = f"""[POST-GEN FACT CHECKING INTERFACE]
Verify every inline citation inside the response text block by cross-referencing it with the literal text mappings in the master source logs.

[MASTER SOURCE LOGS]
{json.dumps(state.retrieved_context_pool, indent=2)}

[DRAFT TEXT TO AUDIT]
{state.final_compiled_synthesis}

[STRICT INSTRUCTIONS]
1. Read the text block. Isolate every factual claim tied to an inline [arXiv:XXXX.XXXX] tag.
2. Confirm the exact text segment for that arXiv ID explicitly states the asserted claim.
3. If the claim is verified, retain it. If it cannot be traced or is a generalized statement, delete or rewrite it to fit the literal context.
4. Return only the cleaned, finalized text block. Remove all commentary wrappers.

Final Cleaned Output:"""
        return self.llm.generate(prompt, temperature=0.0).strip()
