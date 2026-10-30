# Tensor roadmap: implementation and evidence

The uploaded roadmap is the feature benchmark for version 0.3.0. Its suggested
Nim-first/Rust-port micro-loop is a workflow recommendation; the owner explicitly
allowed a different implementation order. Both implementations were developed,
then checked against each other and independently against NumPy.

| Roadmap requirement | Implementation | Verification |
| --- | --- | --- |
| Correct matrix arithmetic, product, transpose, slicing | Generic Matrix APIs and repaired Nim core | Exact, boundary and generated tests in both languages |
| Shape-aware standardized docs | Rust /// and Nim ## API contracts | cargo doc (warnings denied), nim doc |
| Specialized 1-D vectors | Contiguous Vector types, dot, cross, L1/L2 norms and conversions | Geometry, dimensional errors, stable normalization tests |
| General n-D Tensor | Shape/strides/offset, scalars, empty axes, broadcasting, views, contiguity and reshape | Generated layouts, invalid shapes, cross-language/NumPy cases |
| Vector/Matrix/Tensor integration | Checked rank conversions preserving logical element order | Owned and strided conversion tests |
| CLI REPL and variable state | Real expression parsers, assignment, arrays, constructors and library calls | Piped sessions, recovery, state preservation and resource guards |
| n-D visualizers | Nested pretty printing with bounded previews | Large/deep/empty layout tests |
| Criterion / benchy | Dedicated roadmap runners and pinned benchmark-only benchy setup | Repeated runs plus per-size correctness checks |
| 128², 512², 1024² baselines | Naive, blocked, parallel and SIMD products using identical inputs | Complete output comparisons and weighted checksums |
| Profiling / flamegraphs | Real GDB stack sampling, Callgrind instructions and cache simulation | Committed folded samples, SVGs and summary data |
| Loop tiling | Explicit validated tiled matrix kernels | Rectangular sizes, tile tails, empty axes, invalid tile sizes |
| Parallel execution | Rust Rayon; Nim std/tasks with bounded threads and owned channel results | Multiple partitions, debug/release and sanitizer checks |
| AVX / NEON intrinsics | Runtime-dispatched f64 kernels with unaligned tails and scalar fallbacks | AVX exercised; Rust AArch64 compile-check; fallback checks |
| Feedback into slower implementation | Nim checked accessors inlined after instruction profiling | Post-inlining correctness/timing pass retained separately |

## Measured baselines

The following are single timed verification passes printed before the repeated
Criterion/benchy runs (milliseconds). They are observations, not confidence
intervals or universal language speed claims. Parallel runs use four workers.
Repeated-tool output is retained in `results/repeated-benchmark.log`; the complete
comparison is `results/cross-language.json`. Allocation is included consistently.
Inputs are exactly representable binary fractions, so these fixture outputs
matched elementwise with max error zero and equal cross-language checksums.

| Square size | Kernel | Rust ms | Nim ms |
| --- | --- | ---: | ---: |
| 128 | blocked | 0.542 | 2.366 |
| 128 | naive | 6.566 | 9.298 |
| 128 | parallel | 0.459 | 1.460 |
| 128 | simd | 0.378 | 0.389 |
| 512 | blocked | 32.546 | 160.912 |
| 512 | naive | 533.776 | 636.010 |
| 512 | parallel | 13.051 | 74.414 |
| 512 | simd | 19.768 | 25.127 |
| 1024 | blocked | 271.704 | 1222.317 |
| 1024 | naive | 4310.166 | 5233.936 |
| 1024 | parallel | 84.272 | 507.785 |
| 1024 | simd | 174.189 | 175.726 |

Nim's post-inlining pass is retained separately in its `inlined-verify.csv`.
Inlining leaves explicit shape/bounds guards intact. It reduces checked accessor
call overhead; it does not make a naive strided loop cache-friendly. Small or
single-shot measurements vary with scheduling/cache state. SIMD dot benchmarks
also show a speedup over scalar accumulation on this AVX host, justifying the
narrow architecture-specific implementation; optimized paths remain explicit.

See [profiling methodology](PROFILING.md) and [actual flamegraph](profiling/flamegraph.svg).
The instruction chart is not a CPU-time flamegraph. ARM execution is not claimed:
NEON source is present and Rust AArch64 compiles, but this machine is x86-64.

## Deliberate API differences

Rust Tensor handles use copy-on-write; Nim retained Tensor views share element
writes until explicit clone/contiguous copying. Rust reshaping strided views
requires a contiguous copy; Nim does that copy automatically. Nim additionally
supports negative-step slices. Matrix arithmetic stays strict in both languages,
while Tensor arithmetic broadcasts trailing axes. Nim integer `/` promotes to
floating point; Rust integer division follows its scalar type. Random algorithms
are deterministic within each language but differ across languages. None of these
differences changes the common mathematical cases used for parity verification.

Existing real-valued factorization functionality remains available. Sparse,
complex, GPU, autograd, and general nonsymmetric eigen capabilities are not part
of the supplied roadmap and are not represented as implemented.
