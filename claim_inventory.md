# Claim Inventory

## Core Claims

### C1: Detector Error Models (DEMs) are essential for accurate fault-tolerant circuit design
- **Claim**: Detector Error Models enable high-fidelity syndrome extraction and are critical for designing fault-tolerant quantum circuits
- **Status**: SUPPORTED
- **Priority**: HIGH

### C2: Decoder performance is a bottleneck for real-time fault-tolerant quantum computing
- **Claim**: Current decoders cannot achieve the ultra-low latencies (< 1 μs/round) required for real-time QEC at large code distances
- **Status**: SUPPORTED
- **Priority**: HIGH

### C3: GPU acceleration is essential for practical QEC decoders
- **Claim**: GPU-based decoder implementations are necessary to meet real-time latency requirements for fault-tolerant quantum computing
- **Status**: SUPPORTED
- **Priority**: HIGH

### C4: Ising/QUBO formulations enable high-accuracy topological code decoding
- **Claim**: Ising model formulations provide a path to highly accurate decoding for color codes and surface codes
- **Status**: SUPPORTED
- **Priority**: MEDIUM

### C5: Color codes offer potential advantages over surface codes for fault-tolerant computation
- **Claim**: Color codes provide alternative approaches to fault-tolerant quantum computing with different resource tradeoffs
- **Status**: SUPPORTED
- **Priority**: MEDIUM

### C6: AI-based pre-decoders can achieve O(1 μs) decoding latencies
- **Claim**: AI-based pre-decoders integrated with global decoders achieve end-to-end decoding runtimes of order O(1 μs) per round at large code distances on NVIDIA GPUs
- **Status**: SUPPORTED
- **Priority**: HIGH

### C7: Modular decoder architectures are backend-agnostic and composable
- **Claim**: Modular pre-decoder architectures compose with arbitrary global decoding algorithms and are backend-agnostic
- **Status**: SUPPORTED
- **Priority**: MEDIUM

### C8: Data-driven noise learning enables decoder optimization without explicit noise models
- **Claim**: Graph weight estimation can infer decoding weights directly from syndrome statistics without requiring circuit-level noise models
- **Status**: SUPPORTED
- **Priority**: MEDIUM

### C10: NVIDIA GPU stacks provide infrastructure for accelerated QEC decoding
- **Claim**: NVIDIA GPU ecosystems (CUDA-Q, cuQuantum) provide necessary acceleration infrastructure for QEC decoders
- **Status**: SUPPORTED
- **Priority**: HIGH

### C11: Block-wise parallel decoding schemes enable further latency reductions
- **Claim**: Multi-GPU block-wise parallel decoding can reduce runtime to well below O(1 μs) per round
- **Status**: SUPPORTED
- **Priority**: MEDIUM
