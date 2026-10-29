# Implementation verification

Completed on `dev` with Rust 1.85.1 on Linux x86-64. CI additionally defines
checks for current stable Rust; hosted CI has not been executed in this local
session. All implementation commits use author and committer
`farvanyoshi <nvnjoshi@outlook.com>`. Only the Rust checkout was changed.

## Executed checks

| Check | Observed result |
| --- | --- |
| `cargo fmt --check` | Passed |
| `cargo test --locked --no-default-features` | 40 tests and 5 documentation examples passed |
| `cargo test --locked --all-features` | 41 tests and 5 documentation examples passed |
| `cargo test --locked --release --all-features` | 41 tests and 5 documentation examples passed |
| `cargo clippy --locked --all-targets --all-features -- -D warnings` | Passed with no warnings |
| `RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps --all-features` | Passed, including the enabled missing-public-documentation lint |
| `cargo run --locked --example solve` | Passed solution/residual assertions; ran square and minimum-norm solves |
| `cargo run --locked --example views` | Passed mutation assertions and ran transpose-view extraction |
| `cargo bench --locked --bench kernels` | Completed; steady-state measurements include dense sizes through 256×256 |
| `cargo +nightly miri test --locked --test structural --test core_api --test regressions` | All 6 targeted tests passed |
| `cargo package --list --locked --all-features` | Reviewed package contents; ignored local tooling is excluded |
| `cargo package --locked --all-features` | Packaged and rebuilt the extracted archive successfully |

No executed suite has failed, ignored, disabled, or expected-failure tests in the
final results. The one-test difference in minimal builds is the intentionally
feature-gated serde suite, which is compiled and executed in all-feature builds.
The empty default feature set is equivalent to the tested minimal configuration.

Miri ran on the isolated 2026-10-29 nightly, rustc 1.101.0-nightly
`92ffebece` (compiler commit dated 2026-10-28). It exercised element access,
mutable rows/columns, empty slices, chained mutable/transpose views, disjoint
mutation, and structural regressions. It did not interpret the full numerical
suite; that suite ran under the normal debug and release test runners. The crate
also forbids unsafe code. The separate Miri toolchain and caches live only in the
Rust repository's ignored `.tools` directory.

## Numerical evidence

Four fixed-seed property tests execute **416 generated cases**: 128 structural,
128 SVD/QR/least-squares, 96 LU/Cholesky/symmetric-eigen, and 64 single-precision
factor/residual cases. These cases are internal to the 41-test function count;
they are not 416 additional test functions. The double-precision numerical tests
compare with an independent development-only `nalgebra` oracle.

Additional fixtures cover exact expected arithmetic, invalid coordinates/shapes,
empty shapes, integer limits, atomic checked assignment, mapping callback errors,
nonfinite inputs/results, large/small scales, cancellation, scale-ratio overflow,
rectangular/rank-deficient systems, repeated spectra, and nonconvergence.
Reconstruction and orthogonality checks, all four Moore–Penrose identities,
minimum-norm solutions, and scaled backward residuals validate algorithm behavior.
Single-precision factor tests use norm-scaled residuals so entries near zero are
not incorrectly judged against an unrelated local absolute scale.

## Package and dependency review

Default runtime dependencies are `num-traits` and `thiserror`, plus thiserror's
Rust procedural-macro dependencies. Optional serde adds Rust-only serialization.
The numerical oracle and property generator are development-only. The direct
runtime dependencies declare `MIT OR Apache-2.0`. Following the owner's decision,
the repository is also licensed under `MIT OR Apache-2.0`, at the recipient's
option. Both license texts are included, with copyright attributed to
farvanyoshi. Cargo package verification was repeated after adding the license
metadata and no longer reports a missing-license warning. No crate publication
was performed.

`NUMERICS.md` records algorithm choices and floating-point limitations.
`BENCHMARKS.md` records the exploratory kernel comparison and final longer-sample
measurements. These checks establish the tested scope, not a guarantee of exact
arithmetic, arbitrary ill-conditioned accuracy, or universal performance.
