# ReasonBorn Architecture Overview

ReasonBorn consists of 11 integrated modules arranged in a hierarchical processing pipeline:

1. **Perception/Input Module:** Domain-specialized BPE tokenizer.
2. **SLM Transformer Backbone:** Hybrid local-global attention with Sparse MoE.
3. **Reasoning Engine:** Nested CoT Controller for verifiable decomposition.
4. **Episodic Memory:** Short-term, fast read/write buffer.
5. **Semantic Memory:** Long-term hybrid vector database.
6. **Retrieval Layer:** Dense, Sparse, and Graph traversal RAG.
7. **Adaptive Learning Controller:** EWC + Replay online learning.
8. **System-Prompt Manager:** Strict operator policy enforcement.
9. **Output Filter:** Hallucination mitigation and safety.
10. **Audit Module:** JSON-LD formal proof objects and provenance.
11. **Alignment Model:** SFT and RLHF alignment mechanisms.
