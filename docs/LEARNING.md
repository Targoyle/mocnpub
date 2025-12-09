# Learning Journey: From Zero to 46,000x Speedup

This document records the learning process of building mocnpub - a journey from knowing nothing about CUDA, Rust, or secp256k1 to achieving a 46,000x speedup over CPU.

**This project was built through pair programming with AI (Claude).**

---

## Starting Point

- **CUDA/GPGPU**: Never touched before
- **Rust**: Almost beginner
- **secp256k1**: First time
- **Cryptographic computation**: No experience

**Goal**: Build a Nostr npub miner that finds vanity prefixes fast.

---

## Philosophy: Learn by Building

Instead of trying to learn everything upfront, we took a step-by-step approach:

1. **Small wins first** - Get something working before optimizing
2. **Verify at each step** - Make sure it actually works
3. **Don't be afraid to fail** - Wrong approaches teach valuable lessons
4. **Document everything** - Future self will thank you

---

## Step 0: Hello World (CUDA + Rust)

**Goal**: Confirm the development environment works.

- Install CUDA Toolkit 13.0 (Windows + WSL)
- Set up Rust with `cudarc` crate
- Verify RTX 5070 Ti is detected

**Why this matters**: "It doesn't work on my machine" is the worst debugging experience. Verify the basics first.

---

## Step 1: Mandelbrot Set (GPU Basics)

**Goal**: Learn CUDA fundamentals with a visual, easy-to-verify problem.

![Mandelbrot Set](../mandelbrot.png)

- Implement CPU version first
- Port to GPU
- Compare performance: **3.5x speedup** (not amazing, but it works!)

**What we learned**:
- CUDA kernel structure
- Thread/block organization
- Memory transfer (host ↔ device)
- PTX compilation with `build.rs`

**Why Mandelbrot?** It's embarrassingly parallel, easy to visualize correctness, and a classic GPU exercise.

---

## Step 2: CPU npub Miner

**Goal**: Understand the actual problem before GPU-ifying it.

- Learn secp256k1 elliptic curve cryptography
- Understand Nostr key generation (nsec → npub)
- Implement bech32 encoding
- Build CLI with `clap`

**Performance**: ~70,000 keys/sec (single thread)

**What we learned**:
- Public key = Private key × Generator point
- npub is bech32-encoded X-coordinate
- Most time spent in scalar multiplication (93%)

---

## Step 2.5: CPU Optimization

**Goal**: Make CPU version production-ready and identify bottlenecks.

- Multi-threading with `rayon`: **12-20x speedup**
- Input validation (bech32 invalid characters: 1, b, i, o)
- Multiple prefix support (comma-separated)
- Benchmarking with `criterion`

**Key insight**: 93% of time is in secp256k1 operations. This is what GPU needs to accelerate.

---

## Step 3: GPU Port

**Goal**: Move the bottleneck (secp256k1) to GPU.

This was the hardest step. We had to:

1. **Implement 256-bit arithmetic in CUDA** - No BigInt in CUDA!
2. **Port modular arithmetic** - Addition, subtraction, multiplication mod p
3. **Port elliptic curve operations** - Point addition, doubling, scalar multiplication
4. **Handle endianness** - GPU uses little-endian limbs, Nostr uses big-endian bytes

**First working GPU version**: 16x faster than CPU

Not amazing, but **it worked**! The foundation was solid.

---

## Step 4: GPU Optimization

With a working GPU version, we could now optimize:

1. **Consecutive keys + PointAdd** (~300x)
   - Instead of random keys, use k, k+1, k+2...
   - P_{k+1} = P_k + G (addition is cheaper than multiplication)

2. **Montgomery's Trick** (~85x)
   - Batch modular inverse for affine coordinates
   - N inversions → 1 inversion + 3(N-1) multiplications

3. **Endomorphism** (2.9x)
   - secp256k1 has special structure (j-invariant = 0)
   - One scalar multiplication gives 3 X-coordinates to check

4. **Many small optimizations** (see [OPTIMIZATION.md](./OPTIMIZATION.md))

**Final result**: **46,571x faster than CPU** 🔥

---

## Failed Experiments (Equally Valuable!)

Not everything worked. These failures taught us important lessons:

| Experiment | Expected | Actual | Lesson |
|------------|----------|--------|--------|
| SoA memory layout | +20% | -13% | Parallelism > coalescing |
| CPU precomputation | +30% | -73% | GPU is faster than CPU |
| Aggressive register reduction | +50% | -10% | Spilling kills performance |

**Learning**: Always measure. Intuition about GPU performance is often wrong.

---

## Tools That Helped

| Tool | Purpose |
|------|---------|
| `cargo build` | Rust compilation (auto-runs `build.rs` for PTX) |
| `nsys` | GPU timeline profiling |
| `ncu` | Kernel-level analysis |
| `ncu-ui` | Source-level divergence analysis |
| `criterion` | Rust benchmarking |

---

## Reflections

### What Made This Project Successful

1. **Step-by-step approach** - Never tried to do everything at once
2. **Working code first** - Optimize only after correctness is verified
3. **Measure, don't guess** - `ncu` saved us from many wrong turns
4. **Document the journey** - These notes helped us stay on track

### AI Pair Programming

This project demonstrates that AI can be an effective pair programming partner for:

- **Learning new domains** - CUDA, cryptography, GPU optimization
- **Debugging complex issues** - 256-bit arithmetic, endianness
- **Exploring optimization strategies** - Try ideas quickly, measure results
- **Documentation** - Keep track of what worked and what didn't

The human brings domain knowledge (Nostr), intuition, and final judgment.
The AI brings broad technical knowledge, patience, and systematic exploration.

Together: **46,571x speedup** and a working vanity miner! 🎉

---

## Timeline

| Date | Milestone |
|------|-----------|
| 2025-11-14 | Project started |
| 2025-11-22 | CPU version complete |
| 2025-11-23 | CPU optimization complete |
| 2025-11-26 | GPU version working (16x) |
| 2025-11-29 | Major optimizations (38,000x) |
| 2025-12-09 | Final tuning (46,571x) |

**Total development time**: ~4 weeks of evening/weekend sessions
