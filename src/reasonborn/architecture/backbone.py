import torch
import torch.nn as nn
from typing import Dict, Any

# --- Importing the 11 Modules from the Architecture Diagram ---
from ..data.tokenizer import PerceptionModule                            # Module [1]
from .hybrid_attention import HybridAttentionLayer                       # Module [2]
from .moe import SparseMoELayer                                          # Module [2]
from ..reasoning.engine import ReasoningEngine                           # Module [3]
from ..memory.episodic import EpisodicMemory                             # Module [4]
from ..memory.semantic import SemanticMemory                             # Module [5]
from ..memory.retrieval import RetrievalLayer                            # Module [6]
from ..learning.continual_learner import AdaptiveLearningController      # Module [7]
from ..control.prompt_manager import SystemPromptManager                 # Module [8]
from ..control.safety_filter import OutputFilter                         # Module [9]
from ..audit.proof_extractor import AuditModule                          # Module [10]
from ..learning.alignment import RewardModel                             # Module [11]

class ReasonBornSystem(nn.Module):
    """
    Master Subject-Specific Small Language Model (SS-SLM) connecting the 11 core modules.
    Ref: Section 4.1 High-Level Architecture Overview.
    """
    def __init__(self, config: Any):
        super().__init__()
        if isinstance(config, str):
            import yaml
            from types import SimpleNamespace
            with open(config, 'r') as f:
                config_dict = yaml.safe_load(f)
            config = SimpleNamespace(**config_dict)
        self.config = config
        
        # ┌─────────────────────────────────────────────────────────────────┐
        # │                      INPUT PROCESSING                           │
        # └─────────────────────────────────────────────────────────────────┘
        self.perception = PerceptionModule(config.vocab_size)            # Module [1]

        # ┌─────────────────────────────────────────────────────────────────┐
        # │                    CORE COMPUTATION                             │
        # └─────────────────────────────────────────────────────────────────┘
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            if hasattr(config, 'moe_expert_layers') and i in config.moe_expert_layers:
                self.layers.append(SparseMoELayer(config))               # Module [2]
            else:
                self.layers.append(HybridAttentionLayer(config))         # Module [2]
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # ┌─────────────────────────────────────────────────────────────────┐
        # │                   REASONING & MEMORY                            │
        # └─────────────────────────────────────────────────────────────────┘
        # Passing self as the model to ReasoningEngine
        self.reasoning_engine = ReasoningEngine(self, config.max_depth)  # Module [3]
        self.episodic_memory = EpisodicMemory(capacity=config.e_cap)     # Module [4]
        self.semantic_memory = SemanticMemory(db_size=config.s_cap)      # Module [5]
        self.retrieval_layer = RetrievalLayer(
            self.episodic_memory, self.semantic_memory)                  # Module [6]

        # ┌─────────────────────────────────────────────────────────────────┐
        # │                 ADAPTATION & CONTROL                            │
        # └─────────────────────────────────────────────────────────────────┘
        self.learning_controller = AdaptiveLearningController(self, config) # Module [7]
        self.system_prompt_manager = SystemPromptManager()               # Module [8]

        # ┌─────────────────────────────────────────────────────────────────┐
        # │                   OUTPUT & VERIFICATION                         │
        # └─────────────────────────────────────────────────────────────────┘
        self.output_filter = OutputFilter(config)                        # Module [9]
        self.audit_module = AuditModule(config.policy_hash)              # Module [10]
        self.alignment_model = RewardModel(config)                       # Module [11]

    def generate(self, query: str, system_prompt: dict, user_prompt: dict):
        """Complete Data Flow according to Section 4.1."""
        # Step 1: Control Policy Enforcement (Module 8)
        policy = self.system_prompt_manager.load_and_merge_configs(system_prompt, user_prompt)
        pre_decision = self.system_prompt_manager.enforce_input_policy(query, policy)
        if pre_decision.action != 'ALLOW':
            return pre_decision.safe_alternative
            
        # Step 2: Perception & Retrieval (Modules 1, 4, 5, 6)
        context = self.retrieval_layer.hybrid_retrieve(query)
        
        # Step 3: Core Computation & Reasoning (Modules 2, 3)
        # Nested CoT handles decomposition and verification
        raw_answer, reasoning_tree = self.reasoning_engine.run(query, context, policy)
        
        # Step 4: Output Filtering & Provenance (Modules 9, 10)
        filtered_answer = self.output_filter.filter_hallucinations(raw_answer, self)
        proof_object = self.audit_module.extract_proof_object(reasoning_tree)
        
        # Step 5: Format and Return Output
        return self.output_filter.format_final_output(filtered_answer, proof_object, policy)

    # --- Helper methods for ReasoningEngine ---
    def generate_atomic_solution(self, goal, context):
        return "Atomic solution for " + goal

    def synthesize(self, goal, child_solutions):
        return "Synthesized solution for " + goal

    def verify_solution(self, node, solution):
        from dataclasses import dataclass
        @dataclass
        class Verification:
            passed: bool
            confidence: float
            feedback: str = None
        return Verification(passed=True, confidence=0.95)

    def repair_solution(self, node, solution, feedback):
        return solution

    def model_confidence(self, claim):
        return 0.85

ReasonBornModel = ReasonBornSystem
