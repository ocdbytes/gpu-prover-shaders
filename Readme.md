# GPU Prover Shaders

GPU-accelerated BN254 field arithmetic on Apple Silicon using Metal compute shaders. Benchmarks core operations used in zero-knowledge proof systems (specifically [whir](https://github.com/WizardOfMenlo/whir)) against multi-threaded CPU baselines.

## Benchmarks (Apple M4 Pro)

All operations on BN254 scalar field elements in Montgomery form. CPU uses rayon parallelism. GPU times include dispatch overhead; `compute` is the Metal-reported kernel execution time.

### Element-wise Multiply: `result[i] = a[i] * b[i]`

```
  2^11 =     2048 | CPU:   102.62µs | GPU:   253.92µs (compute:    96.33µs) | 0.40x
  2^16 =    65536 | CPU:   323.04µs | GPU:   243.25µs (compute:   124.88µs) | 1.33x
  2^20 =  1048576 | CPU:     1.80ms | GPU:     1.26ms (compute:     1.12ms) | 1.42x
  2^22 =  4194304 | CPU:     5.70ms | GPU:     2.79ms (compute:     2.32ms) | 2.04x
```

### Scalar Mul-Add: `acc[i] += weight * vec[i]`

```
  2^11 =     2048 | CPU:    73.17µs | GPU:   119.12µs (compute:     9.38µs) | 0.61x
  2^16 =    65536 | CPU:   246.50µs | GPU:   139.33µs (compute:    28.21µs) | 1.77x
  2^20 =  1048576 | CPU:     2.07ms | GPU:     1.25ms (compute:     1.01ms) | 1.65x
  2^22 =  4194304 | CPU:     7.51ms | GPU:     2.91ms (compute:     2.44ms) | 2.58x
```

### Eval Eq (Tree): `acc[i] += scalar * prod_k(eq_factor_k(i))`

Uses a hybrid CPU-seed + GPU tree expansion algorithm. The first 10 levels are computed on CPU, then expanded level-by-level on GPU via double-buffered ping-pong.

```
  2^11 =     2048 | CPU:    29.04µs | GPU:   134.38µs (compute:    16.42µs) | 0.22x
  2^16 =    65536 | CPU:   212.38µs | GPU:   244.88µs (compute:   107.79µs) | 0.87x
  2^20 =  1048576 | CPU:     2.26ms | GPU:     1.14ms (compute:   907.75µs) | 1.98x
  2^22 =  4194304 | CPU:     7.84ms | GPU:     2.54ms (compute:     2.38ms) | 3.09x
```

### Fold: `values[i] += (values[half+i] - values[i]) * weight`

```
  2^11 =     2048 | CPU:    11.08µs | GPU:   122.96µs (compute:     9.50µs) | 0.09x
  2^16 =    65536 | CPU:   189.75µs | GPU:   126.71µs (compute:    17.00µs) | 1.50x
  2^20 =  1048576 | CPU:     1.15ms | GPU:   454.12µs (compute:   320.50µs) | 2.54x
  2^22 =  4194304 | CPU:     3.92ms | GPU:     1.84ms (compute:     1.35ms) | 2.13x
```

### Key Observations

- **~110-120µs fixed dispatch overhead** per command buffer (encode + commit + wait). This dominates at small sizes.
- **GPU wins at 2^16+** for most kernels when dispatch overhead is amortized.
- **Eval eq tree** reduced GPU work from O(n * 2^n) to O(2^n), turning a 10x slowdown into a 3x speedup at 2^22.

## Architecture

```
src/
├── main.rs                 # Entry point — runs all benchmarks
├── lib.rs                  # Crate root
├── gpu/
│   ├── context.rs          # Metal device, pipelines, dispatch methods, GPU timing
│   └── buffer.rs           # Page-aligned memory allocation (posix_memalign)
├── field/
│   ├── constants.rs        # BN254 modulus, R², P_INV
│   └── conversion.rs       # Fr ↔ [u32; 8] Montgomery limb conversion
└── bench/
    ├── mod.rs              # verify_results utility
    ├── mul.rs              # Element-wise multiply + scalar mul-add benchmarks
    ├── eval_eq.rs          # Eval eq with tree-based GPU algorithm
    └── fold.rs             # Fold/linear-interpolation benchmark

shaders/
├── field.h                 # BN254 field arithmetic (add, sub, mul, select, load/store)
├── dotprod_field.metal     # dot_product_field, scalar_mul_add_field kernels
├── eval_eq.metal           # eval_eq_field kernel (brute-force, kept for reference)
├── eval_eq_tree.metal      # eval_eq_tree_expand kernel (tree-based)
└── fold.metal              # fold_field kernel
```

## GPU Kernels

All kernels operate on BN254 scalar field elements (256-bit) in Montgomery form, stored as 8 x `uint32` limbs.

| Kernel | Operation | Threads |
|--------|-----------|---------|
| `dot_product_field` | `result[i] = a[i] * b[i]` | n |
| `scalar_mul_add_field` | `acc[i] += weight * vec[i]` | n |
| `eval_eq_tree_expand` | Tree level: `dst[2i] = src[i]*(1-pk)`, `dst[2i+1] = src[i]*pk` | 2^level |
| `fold_field` | `values[i] += (values[half+i] - values[i]) * weight` | half_size |

### Field Arithmetic (`field.h`)

- **Montgomery CIOS multiplication** — 8x8 limb multiply with interleaved reduction
- **Pure 32-bit add/sub** — Carry/borrow detection via comparison (no 64-bit promotion)
- **Branchless select** — Constant-time conditional using bitmask
- All loops annotated with `#pragma unroll`

### Eval Eq Tree Algorithm

The brute-force approach dispatches 2^n threads each doing n multiplications (O(n * 2^n) total work). The tree approach:

1. **CPU seed** — Compute first 10 levels recursively (1024 elements, ~10µs)
2. **GPU expand** — For each remaining level k, dispatch 2^k threads:
   - `hi = val * point[k]` (1 field mul)
   - `lo = val - hi` (1 field sub, avoids second multiply)
   - Write to alternate buffer (ping-pong)
3. All passes encoded in a **single command buffer** with implicit memory barriers

Total GPU work: O(2^n) — same as the CPU recursive algorithm.

## Building

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Rust toolchain (edition 2024)
- Xcode Command Line Tools (for `xcrun`, `metal`, `metallib`)

### Compile Shaders

```sh
cd shaders

# Compile each shader: .metal → .air → .metallib
for f in dotprod_field eval_eq eval_eq_tree fold; do
  xcrun -sdk macosx metal -c ${f}.metal -o ${f}.air
  xcrun -sdk macosx metallib ${f}.air -o ${f}.metallib
done
```

The `.metallib` files are embedded into the Rust binary via `include_bytes!()` at compile time. Recompile shaders whenever `field.h` or any `.metal` file changes.

### Build & Run

```sh
cargo run --release
```

## GPU Timing

Each benchmark reports two GPU times:

- **GPU** — Wall-clock time including command buffer encode, commit, and wait
- **compute** — GPU-reported kernel execution time (via Metal's `GPUStartTime`/`GPUEndTime`)

The difference is dispatch overhead (~110-120µs per command buffer on M4 Pro). At small sizes (2^11-2^14), overhead dominates. At large sizes (2^20+), compute dominates.

## Dependencies

| Crate | Purpose |
|-------|---------|
| `metal` | Apple Metal GPU framework |
| `objc` | ObjC runtime for GPU timestamp queries |
| `ark-bn254` | BN254 scalar field (Fr) |
| `ark-ff`, `ark-std` | Field traits and utilities |
| `rayon` | CPU parallelism for benchmarks |
| `libc` | Page-aligned memory allocation |
