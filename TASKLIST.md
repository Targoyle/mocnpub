# mocnpub - Public Release Tasklist

**Created**: 2025-12-10
**Status**: Preparing for public release

---

## Current Performance

| Metric | Value |
|--------|-------|
| Throughput | **3.26B keys/sec** |
| vs CPU | **46,571x** |
| 8-char prefix | ~6 minutes |

---

## Completed Milestones

| Phase | Description | Date |
|-------|-------------|------|
| Step 0-1 | CUDA + Rust setup, Mandelbrot | 2025-11 |
| Step 2-2.5 | CPU miner implementation | 2025-11-22 |
| Step 3 | GPU port (16x speedup) | 2025-11-26 |
| Step 4 | GPU optimization (46,571x) | 2025-12-09 |

For detailed optimization history, see [docs/OPTIMIZATION.md](./docs/OPTIMIZATION.md).

---

## Public Release Checklist

### Code Quality

- [ ] Unify comment language to English
- [ ] Review and clean up tests (remove obsolete ones)
- [ ] Resolve TODO: dynamic GPU architecture detection in `build.rs:62`

### CI/CD

- [ ] GitHub Actions for build verification (Windows + Linux)
- [ ] dependabot configuration
- [ ] lefthook for pre-commit formatting (rustfmt, nvcc format?)

### Documentation

- [ ] README.ja.md (Japanese version for friends)
- [ ] Document MAX_KEYS_PER_THREAD and VRAM relationship
- [ ] Add Mandelbrot image to README (learning journey showcase)

### Build Portability

- [ ] Dynamic GPU architecture detection
  - Currently hardcoded: `sm_120` (RTX 50 series / Blackwell)
  - PTX has forward compatibility, but explicit arch is faster
  - Options: runtime detection with `cuDeviceGetAttribute` or build-time with `nvidia-smi`
- [ ] Document minimum requirements
  - CUDA Toolkit version (13.0? or lower?)
  - Rust version
  - Supported GPU generations (Compute Capability)
- [ ] Test on different GPUs (if possible)

### License

- [x] MIT License file exists
- [ ] Review license choice (MIT vs Apache 2.0 vs dual?)

### Release

- [ ] Version number decision (0.1.0? 1.0.0?)
- [ ] CHANGELOG.md (optional)
- [ ] crates.io publication (TBD - maintenance commitment?)
- [ ] GitHub Release with binary

---

## Notes

- Commit messages are in Japanese (that's OK)
- Code comments will be English
- README will have both English and Japanese versions
