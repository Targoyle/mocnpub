# mocnpub - Development Guide

**Last Updated**: 2025-12-10

This is the development guide for Claude Code. For user documentation, see [README.md](./README.md).

---

## Project Overview

**mocnpub** is a Nostr npub vanity miner using CUDA.

- Find npub with desired prefix (e.g., `npub1m0ctane...`)
- GPU-accelerated secp256k1 computation
- **3.26B keys/sec** on RTX 5070 Ti (46,571x faster than CPU)

---

## Development Environment

### Build

```bash
cargo build --release
```

PTX is automatically compiled by `build.rs`. No manual `nvcc` required.

### Build Options

```bash
# Custom MAX_KEYS_PER_THREAD (default: 1408)
MAX_KEYS_PER_THREAD=2048 cargo build --release
```

### Run

```bash
# CPU mode
cargo run --release -- --prefix m0ctane

# GPU mode
cargo run --release -- --gpu --prefix m0ctane
```

### Test

```bash
cargo test
```

### Windows Development

- Develop in WSL, commit and push
- Pull on Windows, run for best performance
- Windows native is ~20x faster than WSL for CUDA

---

## Architecture

### Key Files

| File | Description |
|------|-------------|
| `src/main.rs` | CLI entry point |
| `src/gpu.rs` | CUDA kernel loading and execution |
| `cuda/secp256k1.cu` | CUDA kernel (secp256k1 + prefix matching) |
| `build.rs` | PTX compilation |

### Current Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `threads_per_block` | 128 | 4 warps, sweet spot |
| `keys_per_thread` | 1408 | Compile-time constant |
| `batch_size` | 3,584,000 | Auto-adjusted for SM count |
| Registers/Thread | 128 | `__launch_bounds__(128, 4)` |

---

## Documentation

| Document | Content |
|----------|---------|
| [docs/OPTIMIZATION.md](./docs/OPTIMIZATION.md) | Optimization history and profiling results |
| [docs/LEARNING.md](./docs/LEARNING.md) | Learning journey (Step 0-4) |

---

## TODOs for Public Release

### Code Quality

- [ ] Unify comment language to English
- [ ] Review and clean up tests
- [ ] Resolve `build.rs` TODO: dynamic GPU architecture detection

### CI/CD

- [ ] GitHub Actions for build verification
- [ ] dependabot for dependency updates
- [ ] lefthook for pre-commit formatting

### Documentation

- [ ] README.ja.md (Japanese version)
- [ ] MAX_KEYS_PER_THREAD and VRAM relationship explanation

### Build Portability

- [ ] Dynamic GPU architecture detection (`sm_120` is currently hardcoded)
- [ ] Document minimum CUDA version requirements
- [ ] Test on different GPU generations

---

## Profiling

### nsys (Timeline)

```bash
nsys profile ./target/release/mocnpub-main --gpu --prefix 0000
```

### ncu (Kernel Analysis)

```bash
ncu --set full -o profile ./target/release/mocnpub-main --gpu --prefix 0000 --batch-size 1024
```

Open `.ncu-rep` in ncu-ui for source-level analysis (requires `-lineinfo` in nvcc).
