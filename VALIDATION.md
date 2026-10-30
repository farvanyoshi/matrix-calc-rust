# Roadmap verification, version 0.3.0

Tested on Linux x86-64 with Rust 1.85.1 and Nim 2.2.12 (C backend). This records
executed checks; hosted CI and ARM runtime execution are not claimed.

| Check | Result |
| --- | --- |
| Rust debug, all features | 57 tests + 6 documentation examples passed |
| Rust release, all features | 57 tests + 6 documentation examples passed |
| Rust formatting / Clippy | Passed, warnings denied |
| Rust documentation | Passed, warnings denied; explicit public shape contracts |
| Nim debug / ORC | 33 named tests passed |
| Nim release / ORC | 33 named tests passed |
| Nim release / ARC | 33 named tests passed |
| Nim release with compiler bounds/overflow checks disabled | 33 named tests passed; explicit guards remain effective |
| Cross-language / independent NumPy | 114 cases passed, NumPy 2.3.5 |
| Rust Miri | 10 Tensor/Vector tests + 1 explicit scalar-dispatch test passed |
| Rust AddressSanitizer | Accelerated, Tensor and Vector suites passed |
| Nim AddressSanitizer + UndefinedBehaviorSanitizer | Accelerated task/FFI kernel checks passed |
| Scalar fallback | Rust Miri dispatch and Nim forced scalar/threads-off execution passed |
| Architecture checks | AVX executed on host; Rust NEON/AArch64 compile-check passed |
| REPL integration | Piped matrix, variable, syntax-error recovery and subsequent sum executed in both languages |
| Benchmarks | Criterion and benchy repeated all 128²/512²/1024² kernels; elementwise errors zero on binary-fraction fixtures |
| Profiling | Real GDB stack samples and Callgrind instruction/cache simulation retained |

Rust's pre-existing generated test categories contain 416 cases inside their test
functions. New Tensor tests additionally generate layouts/broadcasts. Nim includes
128 seeded float32/64 factorization cases and 49 tensor shape combinations. These
are internal generated cases, not additional named test-function counts. Tests
cover invalid coordinates, rank-zero and empty tensors, strided/noncontiguous
layouts, right broadcasting, conversion, ownership/alias behavior, geometry,
normalization, factor reconstruction, pseudoinverse identities, nonconvergence,
integer overflow, and REPL malformed input/state preservation/output limits.

Both READMEs document tensor ownership differences, division behavior and supported
commands. Shape checks are not disabled to obtain faster benchmark results. Nim
accessors were inlined based on profiling; their explicit guards are retained.
Hardware intrinsics are narrowly scoped and runtime-dispatched; no NEON execution
on this x86 worker is claimed. Miri tests force a scalar dispatch because Miri
does not execute the architecture kernels; ASan exercises the actual AVX code.

See [roadmap coverage and timings](docs/ROADMAP.md),
[profiling methodology](docs/PROFILING.md), and the committed profile/timing data.
No package registry publication is part of these checks.

## Rust packaging

`cargo package --list` and extracted-archive build verification are rerun after
the final commit. Runtime dependencies are Rust-only; Criterion, the independent
numerical oracle and property generator are development dependencies.
