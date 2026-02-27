"""
ReasonBorn System — Production Architecture Backbone
=======================================================
Master SS-SLM module integrating all 11 sub-systems:
[1] Perception, [2] Hybrid Attention, [3] Reasoning Engine,
[4] Episodic Memory, [5] Semantic Memory, [6] Retrieval Layer,
[7] Adaptive Learning Controller, [8] Uncertainty Estimator,
[9] Output Filter, [10] Audit Module, [11] Alignment/Reward Model

Uses RMSNorm, weight-tied embeddings, and MoE/HybridAttention layers.
Fully integrated PyTorch module ready for DDP/FSDP distributed training.

Per ReasonBorn.md Section 4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

# --- Importing the Sub-Modules ---
from ..data.tokenizer import PerceptionModule
from .hybrid_attention import HybridAttentionLayer
from .moe import SparseMoELayer, ExpertFFN
from ..reasoning.engine import ReasoningEngine
from ..memory.episodic import EpisodicMemory
from ..memory.semantic import SemanticMemory
from ..memory.retrieval import RetrievalLayer
from ..learning.continual_learner import AdaptiveLearningController
from ..control.prompt_manager import SystemPromptManager
from ..control.safety_filter import OutputFilter
from ..audit.proof_extractor import AuditModule
from ..learning.alignment import RewardModel


class RMSNorm(nn.Module):
    """Production Root Mean Square Normalization (faster/more stable than LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class ReasonBornBlock(nn.Module):
    """Wraps Attention/MoE layers with RMSNorm and Residual Connections."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

        self.attention = HybridAttentionLayer(config)

        # Check if this specific layer index is an MoE layer
        moe_layers = getattr(config, 'moe_expert_layers', set())
        if layer_idx in moe_layers:
            self.feed_forward = SparseMoELayer(config)
        else:
            # Standard dense SwiGLU FFN for non-MoE layers
            self.feed_forward = ExpertFFN(
                config.d_model,
                getattr(config, 'intermediate_size',
                        int(config.d_model * 4 * 2 / 3)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        # 1. Attention Block with pre-norm + residual
        h = x + self.attention(self.attn_norm(x))

        # 2. FFN / MoE Block with pre-norm + residual
        moe_loss = 0.0
        ffn_input = self.ffn_norm(h)

        if isinstance(self.feed_forward, SparseMoELayer):
            ffn_out, moe_loss = self.feed_forward(ffn_input)
            out = h + ffn_out
        else:
            out = h + self.feed_forward(ffn_input)

        return out, moe_loss


class ReasonBornSystem(nn.Module):
    """
    Master Subject-Specific Small Language Model (SS-SLM).
    Fully integrated PyTorch module ready for DDP/FSDP distributed training.
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # ┌─────────────────────────────────────────────────────────────────┐
        # │                      INPUT PROCESSING                           │
        # └─────────────────────────────────────────────────────────────────┘
        self.perception = PerceptionModule(config.vocab_size)
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)

        # ┌─────────────────────────────────────────────────────────────────┐
        # │                    CORE COMPUTATION                             │
        # └─────────────────────────────────────────────────────────────────┘
        self.layers = nn.ModuleList([
            ReasonBornBlock(config, i) for i in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (standard in modern LLMs to save parameters)
        self.lm_head.weight = self.embeddings.weight

        # ┌─────────────────────────────────────────────────────────────────┐
        # │                   REASONING & MEMORY                            │
        # └─────────────────────────────────────────────────────────────────┘
        self.reasoning_engine = ReasoningEngine(
            self, getattr(config, 'max_depth', 3))
        self.episodic_memory = EpisodicMemory(
            capacity=getattr(config, 'e_cap', 5000))
        self.semantic_memory = SemanticMemory(
            db_size=getattr(config, 's_cap', 1000000))
        self.retrieval_layer = RetrievalLayer(
            self.episodic_memory, self.semantic_memory)

        # ┌─────────────────────────────────────────────────────────────────┐
        # │                 ADAPTATION & CONTROL                            │
        # └─────────────────────────────────────────────────────────────────┘
        self.learning_controller = AdaptiveLearningController(self, config)
        self.system_prompt_manager = SystemPromptManager()
        self.output_filter = OutputFilter(config)
        self.audit_module = AuditModule(
            getattr(config, 'policy_hash', 'unverified'))
        self.alignment_model = RewardModel(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Any:
        """
        The real PyTorch training loop forward pass.
        Calculates autoregressive CrossEntropyLoss and MoE Load Balancing Loss.
        """
        hidden_states = self.embeddings(input_ids)
        total_moe_loss = 0.0

        for block in self.layers:
            hidden_states, moe_loss = block(hidden_states)
            total_moe_loss += moe_loss

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1))

            # Add auxiliary loss for MoE routing
            loss = loss + total_moe_loss

        from types import SimpleNamespace
        return SimpleNamespace(
            loss=loss, logits=logits, aux_loss=total_moe_loss)

    @torch.no_grad()
    def generate_internal(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Actual autoregressive token generation used by the Reasoning Engine
        and Verifiers. No placeholders.
        """
        self.eval()
        input_ids = self.perception.encode_input(prompt)
        if isinstance(input_ids, dict):
            input_ids = torch.tensor(
                input_ids['input_ids'], dtype=torch.long, device=self.device)
        elif not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long,
                                     device=self.device)
        else:
            input_ids = input_ids.to(self.device)

        # Add batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        generated_tokens = []

        for _ in range(max_tokens):
            outputs = self.forward(input_ids=input_ids)
            next_token_logits = outputs.logits[:, -1, :]

            if temperature > 0.0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(
                    next_token_logits, dim=-1, keepdim=True)

            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop if EOS token generated
            eos_id = getattr(self.perception, 'eos_token_id', 2)
            if next_token.item() == eos_id:
                break

        self.train()
        return self.perception.decode(generated_tokens)

    def generate(
        self,
        query: str,
        system_prompt: Optional[dict] = None,
        user_prompt: Optional[dict] = None,
    ):
        """High-level Orchestration Data Flow."""
        # If called with operator/user policy dicts
        if system_prompt is not None and user_prompt is not None:
            policy = self.system_prompt_manager.load_and_merge_configs(
                system_prompt, user_prompt)
            pre_decision = self.system_prompt_manager.enforce_input_policy(
                query, policy)
            if pre_decision.action != 'ALLOW':
                return pre_decision.safe_alternative
        else:
            policy = {}

        context = self.retrieval_layer.hybrid_retrieve(query)

        # The ReasoningEngine uses the real self.generate_internal
        result = self.reasoning_engine.run(query, context)
        raw_answer = result.get('answer', '')
        reasoning_tree = result.get('reasoning_tree')

        filtered_answer = self.output_filter.filter_hallucinations(
            raw_answer, self)
        proof_object = self.audit_module.extract_proof_object(reasoning_tree)

        return self.output_filter.format_final_output(
            filtered_answer, proof_object, policy)

    # Note: verify_solution and repair_solution are handled directly inside
    # the ReasoningEngine and Verifier modules. They no longer exist here.
