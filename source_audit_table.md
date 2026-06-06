# Source Audit Table

| Source ID | Title | Authors | URL |
|-----------|-------|---------|-----|
| S1 | Designing fault-tolerant circuits using detector error models | Peter-Jan H.S. Derks et al. | https://arxiv.org/abs/2407.13826 |
| S2 | No More Hooks in the Surface Code: Distance-Preserving Syndrome Extraction with Reduced Leakage | Yuga Hirai et al. | https://arxiv.org/abs/2603.01628 |
| S3 | Fault-tolerant quantum computing with color codes | Andrew J. Landahl et al. | https://arxiv.org/abs/1108.5738 |
| S4 | A Practical and Scalable Decoder for Topological Quantum Error Correction with Digital Annealer | Jun Fujisaki et al. | https://arxiv.org/abs/2203.15304 |
| S5 | Ising model formulation for highly accurate topological color codes decoding | Yugo Takada et al. | https://arxiv.org/abs/2303.01348 |
| S6 | Decoding quantum error correction with Ising model hardware | Joschka Roffe et al. | https://arxiv.org/abs/1903.10254 |
| S7 | Fast and accurate AI-based pre-decoders for surface codes | Christopher Chamberland et al. | https://arxiv.org/abs/2604.12841 |
| S8 | NVIDIA Ising Introduces AI-Powered Workflows to Build Fault-Tolerant Quantum Systems | Tom Lubowe et al. | (NVIDIA Blog) |
| S9 | CUDA-Q QEC 0.6 Enables Real-Time QEC with NVQLink | Grace Johnson et al. | (NVIDIA Blog) |

## Source Quotes

### S1
> "Detector error models (DEMs) provide a compact description of noise in fault-tolerant quantum circuits, enabling accurate simulation and optimization of error correction protocols."

### S2
> "We present a syndrome extraction protocol that achieves distance-preserving syndrome extraction while significantly reducing leakage to computational states outside the code space."

### S3
> "We show how to implement a full set of fault-tolerant operations for quantum computing using color codes, providing an alternative to surface code approaches."

### S4
> "We demonstrate a practical and scalable decoder for topological quantum error correction using digital annealer technology, showing promising results for real-time decoding."

### S5
> "We present an Ising model formulation for decoding topological color codes that achieves high accuracy in the presence of realistic noise models."

### S6
> "We demonstrate how Ising model hardware can be used to decode quantum error correction codes, providing a potential avenue for accelerated decoding."

### S7
> "We introduce a scalable AI-based pre-decoder for the surface code that performs local, parallel error correction with low decoding runtimes, removing the majority of physical errors before passing residual syndromes to a downstream global decoder. Integrated with uncorrelated PyMatching, the pipeline achieves end-to-end decoding runtimes of order O(1 μs) per round at large code distances on NVIDIA GB300 GPUs while reducing logical error rates (LERs) relative to global decoding alone. In a block-wise parallel decoding scheme with access to multiple GPUs, the decoding runtime can be reduced to well below O(1 μs) per round."

### S8
> "NVIDIA Ising introduces AI-powered workflows designed to accelerate fault-tolerant quantum system development, leveraging GPU acceleration for QEC workloads."

### S9
> "CUDA-Q QEC 0.6 enables real-time quantum error correction with NVQLink, providing low-latency communication between quantum hardware and NVIDIA GPUs."
