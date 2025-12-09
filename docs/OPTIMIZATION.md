# GPU Optimization Journey

This document records the optimization history of mocnpub's CUDA kernel for secp256k1 public key generation.

---

## Final Performance

| Stage | Throughput | vs CPU |
|-------|------------|--------|
| CPU (16 threads) | ~70,000 keys/sec | 1x |
| **GPU (final)** | **3.26B keys/sec** | **46,571x** |

**8-character prefix found in ~6 minutes!**

---

## Optimization Timeline

### Phase 1: Core Algorithms

| Optimization | Effect | Description |
|--------------|--------|-------------|
| Consecutive Secret Keys + PointAdd | ~300x | Instead of random keys, use k, k+1, k+2... and compute P+G |
| Montgomery's Trick | ~85x | Batch modular inverse: N inversions → 1 inversion + 3(N-1) multiplications |
| Mixed Addition | ~30% | Exploit Z=1 for generator point G |
| GPU-side Prefix Matching | Skip bech32 | Match directly on X-coordinate bytes |

### Phase 2: secp256k1 Specific

| Optimization | Effect | Description |
|--------------|--------|-------------|
| **Endomorphism (β, β²)** | **2.9x** | Check 3 X-coordinates per scalar multiplication (Nostr uses X-only) |
| `_ModSquare` Optimization | +3.5% | Exploit symmetry: 16 → 10 multiplications |

### Phase 3: GPU Tuning

| Optimization | Effect | Description |
|--------------|--------|-------------|
| `keys_per_thread` (1408) | **+120%** | Maximize work per thread within VRAM limits |
| `threads_per_block` (128) | +6.2% | 4 warps = sweet spot |
| `batch_size` (3584000) | +10.4% | Reduce kernel launch overhead, GPU utilization 70% → 95% |
| Tail Effect Mitigation | +1.4% | Auto-adjust batch_size to SM × threads_per_block multiple |
| `_ModSub`/`_ModAdd` Branchless | +2.3% | Eliminate 50% warp divergence from 256-bit comparisons |
| Shared Memory for patterns/masks | +3.4% | Reduce global memory access for multi-prefix matching |
| `__launch_bounds__(128, 4)` | **+5%** | Limit registers 130→128, Occupancy 25%→33% |

---

## Experiments That Didn't Work

### SoA (Structure of Arrays) - Abandoned

**Hypothesis**: Coalesced memory access should improve performance.

**Result**: 3.09B → 2.70B keys/sec (-13%)

| Metric | AoS (current) | SoA |
|--------|---------------|-----|
| Compute Throughput | 83% | 77% |
| Waves Per SM | 42.67 | 0.75 |
| Global Coalescing | 25% | **97%** |
| keys/sec | **3.09B** | 2.70B |

**Why SoA failed**:
- SoA consumes 4x VRAM for work buffers → smaller batch_size
- Parallelism matters more than coalescing for compute-heavy kernels
- AoS hides memory latency through massive parallelism (28000 blocks)

**Learning**: Coalescing isn't everything. For compute-heavy kernels, parallelism wins.

---

### CPU Public Key Precomputation - Abandoned

**Hypothesis**: Compute initial public keys on CPU to reduce GPU register pressure.

**Result**: 3.09B → 844M keys/sec (-73%)

| Version | keys/sec | Registers | Occupancy |
|---------|----------|-----------|-----------|
| GPU `_PointMult` | **3.09B** | 130 | 33% |
| CPU Precompute (rayon) | 844M | 98 | 33% |

**Why it failed**:
- Registers dropped 130→98, but Occupancy stayed at 33% (not enough reduction)
- CPU became the bottleneck: 1.1M key generations slower than GPU
- "CPU is idle" assumption was wrong

**Learning**: GPU is so fast that even parallel CPU can't keep up.

---

### Register Reduction (64 registers) - Abandoned

**Hypothesis**: Fewer registers → higher Occupancy → better performance.

**Result**: 1.14B → 1.03B keys/sec (-10%)

- Occupancy: 33% → 67% ✅
- Spilling Overhead: 96% 😱

**Learning**: secp256k1 needs 256-bit variables. 128 registers is the sweet spot. Let the compiler do its job.

---

### `_Reduce512` Branchless - Abandoned

**Hypothesis**: Eliminate 99.16% divergent branch in reduction.

**Result**: 3.20B → 3.19B keys/sec (-0.3%)

**Why it failed**: Branch prediction was working well. Branchless version has more instructions.

---

## Profiling Results (ncu)

### Final Configuration (with `__launch_bounds__(128, 4)`)

| Metric | Value | Notes |
|--------|-------|-------|
| Compute Throughput | 81% | **Compute bound** |
| Memory Throughput | 20% | Plenty of headroom |
| Occupancy | 33% | Limited by registers |
| Registers/Thread | 128 | `__launch_bounds__` enforced |
| L1/TEX Hit Rate | 0.05% | Low, but hidden by parallelism |
| L2 Hit Rate | 43% | |

### Key Insights

1. **Parallelism is king** - 28000 blocks hide memory latency
2. **Compute-heavy kernels are forgiving** - Low cache hit rate doesn't matter
3. **VRAM efficiency enables parallelism** - AoS uses less VRAM → larger batch_size
4. **128 = 2^7 aligns with GPU architecture** - Register allocation and block sizes

---

## Tools Used

**nsys (Nsight Systems)**: Overall timeline analysis
```bash
nsys profile ./mocnpub-main --gpu --prefix 0000
```

**ncu (Nsight Compute)**: Kernel-level profiling
```bash
ncu --set full -o profile ./mocnpub-main --gpu --prefix 0000 --batch-size 1024
```

**ncu-ui Source Analysis**: Requires `-lineinfo` in nvcc
- Open `.ncu-rep` file → Source page → Divergent Branches column

---

## References

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- secp256k1 endomorphism: GLV method (Gallant-Lambert-Vanstone)
