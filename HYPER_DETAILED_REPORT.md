# ReasonBorn Codebase & Model Architecture Analysis

## 1. Codebase Composition & Density Analysis

A deep static analysis of all `.py` files under the `src/` directory reveals the following code breakdown:

- **Total Lines Analyzed**: 6,637
- **Real Action Code Lines**: 4,046
- **Non-Code Lines (Fluff)**: 2,591
  - **Empty Lines**: 1,105
  - **Comment Lines**: 333
  - **Docstring Lines**: 1,153

### Percentage Breakdown
- **Real Action Code**: **60.96%**
- **Fluff / Clown Code / Documentation / Whitespace**: **39.04%**

### Structural Analysis
- **Total Classes**: 70
- **Total Functions/Methods**: 267
- **Empty Functions (just pass/docstring)**: 0
- **Functions with Mock/Simple Returns**: 30 (Many of these are in the symbolic verifier, mock objects, or testing stubs)
- **Functions that look like placeholders/mocks/empty**: 11.24%

*Conclusion on Code Quality:* The repository contains a moderately high proportion of structural documentation and whitespace (nearly 40%). While this is helpful for understanding the ReasonBorn theory and architecture, a sizable fraction of functions (~11%) rely on simplistic or mock returns (e.g. returning static confidence scores or boolean values in verification modules).

---

## 2. Model Parameter Count Estimation

Based on the model definition scaling in `configs/training/pretraining_mi300x.yaml`, we performed a parameter calculation to verify the exact parameter count.

**Architectural Parameters (AMD Instella 3B Target Scale)**:
- **d_model**: 1536
- **num_layers**: 48
- **vocab_size**: 50000
- **intermediate_size**: 6144
- **num_heads**: 24
- **num_experts**: 8
- **MoE Expert Layers**: 5 layers (indices 8, 16, 24, 32, 40)

**Parameter Calculation Breakdown**:
- **Non-MoE Layers**: 43 layers
- **MoE Layers**: 5 layers
- **Attention parameters per layer**: \(4 \times d\_model^2\) = ~9.4M
- **FFN parameters per layer**: \(3 \times d\_model \times intermediate\_size\) = ~28.3M
- **MoE FFN parameters per layer**: \(num\_experts \times FFN\_params\) = ~226M
- **MoE Gate parameters per layer**: \(d\_model \times num\_experts\) = ~12.2K

### Final Parameter Count
**Total Estimated Parameters**: **2.957 Billion (~3B)**

This perfectly aligns with the documentation claiming a 3B parameter model, optimized for the AMD MI300X configuration while maintaining the paper's base architectural ratios.

---

## 3. Summary
- **Code Actionability**: 61% real, executable Python logic. 39% whitespace/documentation/fluff.
- **Model Size**: ~2.96B parameters, confirming the model acts as a "3B" parameter class SLM.
