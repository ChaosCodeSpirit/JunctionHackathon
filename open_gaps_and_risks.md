# Open Gaps and Risks

## Previously Identified Gaps (Resolved)

### NG-1: Date Anomaly
- **Status**: ACCEPTABLE
- **Notes**: Minor date discrepancy noted in documentation; reviewer found acceptable.

### NG-2: S7 Quote Truncation
- **Status**: RESOLVED
- **Action Taken**: S7 quote updated to include full abstract text mentioning O(1 μs) per round decoding latency.
- **Source**: Chamberland et al. (2026) - Full abstract now included in source_audit_table.md

### NG-3: S7 Quote Missing "AI-based pre-decoder" Explicit Mention
- **Status**: RESOLVED
- **Action Taken**: S7 quote now explicitly includes "AI-based pre-decoder" in the quote text.
- **Source**: Chamberland et al. (2026) - Quote updated in source_audit_table.md

## Blocking Gaps (Resolved)

### CG-1: Invalid Support Citation for C9
- **Status**: RESOLVED
- **Action Taken**: C9 removed entirely from claim inventory (was incorrectly marked "Support: inference").
- **Impact**: Claim inventory now contains 10 claims (C1-C8, C10-C11)

### CG-2: S9 Cited for DEMs Claim Without DEMs Discussion
- **Status**: RESOLVED
- **Action Taken**: S9 removed from C2's support (S9 discusses CUDA-Q QEC 0.6 with NVQLink, not DEMs).
- **Impact**: C2 now supported by S7 (DIRECT) and S2 (INDIRECT) only.

## Remaining Open Questions

### OQ-1: Scalability Beyond GPU Clusters
- **Question**: Can single-GPU systems achieve required latencies for production-scale QEC?
- **Risk Level**: MEDIUM
- **Mitigation**: Block-wise parallel decoding with multi-GPU access addresses this per S7.

### OQ-2: Noise Model Generalization
- **Question**: Does data-driven weight inference generalize across different quantum hardware platforms?
- **Risk Level**: MEDIUM
- **Mitigation**: S7 demonstrates approach without explicit circuit-level noise model requirement.

### OQ-3: Integration Complexity
- **Question**: What is the engineering overhead for integrating modular pre-decoders with existing quantum hardware control systems?
- **Risk Level**: LOW
- **Mitigation**: S7 states architecture is backend-agnostic and completely open source.

---

## Risk Assessment Summary
- **Total Open Gaps**: 3 (all resolved)
- **Total Remaining Risks**: 3 (all LOW/MEDIUM)
- **Blocking Issues**:0 (all resolved)
