# Rust matrix library completion plan

## Scope, ownership, and working agreement

Complete the Rust crate as a reliable, documented dense matrix library. The Nim
implementation can follow the resulting API later; this work changes only
`matrix-calc-rust`. Calculator parsing and a REPL are separate application work.

- Integration branch: `dev`, created from `841e782`.
- All future commits must use author and committer
  `farvanyoshi <nvnjoshi@outlook.com>`, without additional attribution trailers.
  Repository-local Git identity is configured and verified. Check the effective
  identity before committing, including any environment overrides.
- Keep changes in small, coherent commits on `dev`. If an algorithm needs an
  isolated branch, use `feature/<topic>` based on `dev` and integrate it only after its
  acceptance checks pass. Do not rewrite existing history or publish/push as part
  of this plan.
- This document is a plan, not a claim that its implementation is complete.
  The original planning pass did not modify implementation files. Subsequent
  implementation changes are tracked on `dev`; the Nim checkout is unchanged.

The completion target includes generic matrix storage and arithmetic, real-valued
numerics for `f32` and `f64`, reusable decompositions, square and rectangular
solvers, rank, pseudoinverse, condition number, and real symmetric eigendecomposition.
Sparse storage, GPU/distributed execution, complex-valued decompositions,
general nonsymmetric eigendecomposition, arbitrary precision, and native BLAS
integration are follow-on projects rather than release prerequisites.

## Baseline and implementation order

The current crate has a private row-major `Vec<T>`, basic constructors, element
access, iteration, mapping, arithmetic, and matrix multiplication. Its CLI prints
`Hello, world!`. A fresh `cargo test --frozen` reports **13 passed, 2 failed**;
both failures are defective test fixtures. Additional confirmed implementation
bugs are listed in milestone 1.

Milestones are sequential. Structural API contracts are settled before building
numerical algorithms on them. Each milestone must have working code, public API
tests, examples where appropriate, and documentation before it is complete.

| Milestone | Deliverable | Depends on |
| --- | --- | --- |
| 1 | Correct existing core and meaningful regressions | Baseline |
| 2 | Consistent storage, errors, construction, access, and arithmetic API | 1 |
| 3 | Matrix manipulation, views, vector operations, and reductions | 2 |
| 4 | Checked arithmetic and stable real-valued numerical primitives | 3 |
| 5 | Triangular solvers, pivoted LU, determinant, inverse, and Cholesky | 4 |
| 6 | Householder QR and full-rank least squares | 5 |
| 7 | SVD, rank-deficient solving, pseudoinverse, condition number, symmetric eigen | 6 |
| 8 | Performance validation, documentation, packaging, and release checks | 1–7 |

## Contracts shared by every milestone

### Representation and empty matrices

- Keep `Matrix<T>` generic, owned, dense, row-major, and dynamically sized.
- Maintain `data.len() == rows * cols`, with checked dimension arithmetic before
  allocating, indexing, or constructing results. Keep fields private; expose a
  mutable slice, never a mutable `Vec` that could change the element count.
- Preserve distinct shapes `0×0`, `m×0`, and `0×n`. `len` is storage length and
  `is_empty` means zero stored elements. Both coordinates must be independently
  valid before flattening an index.
- A row iterator yields exactly `rows` rows, including empty rows for `m×0`.
  Column iteration similarly respects `cols`. Use an iterator representation
  that can express zero-width rows instead of `chunks(0)`.
- Define operations on empty shapes explicitly: transpose swaps dimensions;
  `(m×0) * (0×n)` produces an `m×n` zero matrix. Empty sums and products use their
  additive and multiplicative identities. Empty mean/min/max are errors.
- For square `0×0` matrices, trace is zero, determinant is one, and inverse is
  `0×0`. Solving a `0×0` system with a `0×k` RHS returns `0×k`.

### Errors, panics, and naming

- Fallible structural operations use `try_*` and return structured errors.
  Panicking convenience constructors/operators delegate to the validated path.
  `get`/`get_mut` retain idiomatic `Option`; explicit fallible accessors return
  the useful row/column errors. `Index`/`IndexMut` panic on invalid coordinates.
- Shape-preserving operations that cannot structurally fail should not return a
  meaningless `Result`. Make `map` accept `FnMut`; give `try_map` meaningful
  callback fallibility. Keep `try_zip_map` responsible for shape validation;
  avoid mixing arbitrary callback errors into every numerical error type.
- Use clear canonical names, such as `element_div` instead of `el_div`, and one
  consistent constructor argument order. Preserve the existing order where
  practical; document unavoidable changes and temporarily retain aliases when
  inexpensive. Record API migration notes before the next version release.
- Expand errors for dimension overflow, allocation/capacity failure where
  recoverable, invalid bounds/ranges, nonsquare input, singular or rank-deficient
  systems, nonfinite input, invalid tolerance, failed positive definiteness, and
  iteration nonconvergence. Error formatting must not overflow itself.
- `try_*` is not a blanket guarantee against panicking user callbacks, custom
  scalar implementations, or arbitrary element allocations. Document those
  boundaries. Use `try_reserve` where fallible allocation is part of the API.
- Mark discarded matrix results with `#[must_use]`. Document allocation,
  mutation, errors, and panics for each public operation.

### Scalars and arithmetic

- Apply minimal per-operation trait bounds. Storage, indexing, transpose, and
  moving elements must not require numeric types or unnecessary `Copy` bounds.
- Preserve exact generic arithmetic where the corresponding Rust traits exist;
  do not silently convert integers to floats or introduce implicit broadcasting.
- `*` between matrices means matrix multiplication; Hadamard multiplication is
  explicit. `/` is scalar division, with elementwise matrix division named.
  Add unary `Neg` and scalar assignment operators where Rust coherence permits.
- Borrowed, owned, and assignment variants should share kernels. Avoid blanket
  scalar implementations that overlap matrix implementations. Named scalar
  methods remain available if an operator form would require restrictive bounds.
- Ordinary integer arithmetic follows the scalar's Rust semantics, including
  truncating division and possible overflow. Separate `checked_*` methods for
  supported primitive integers return errors for overflow, zero divisors, and
  signed minimum divided by minus one. Never catch panics to emulate checking.
- Validated scalar and elementwise division consistently reject zero divisors,
  including floating signed zero. Validate shapes first. Handle a zero scalar
  consistently even when the input has no elements. Nonzero floating arithmetic
  follows IEEE behavior; stricter finite checks apply to numerical solvers.
- Exact equality remains `PartialEq`. Approximate equality is explicit and uses
  absolute and relative tolerances; it must never replace equality semantics.

### Numerical policy

- Implement decomposition APIs first for real `f32` and `f64`. Reuse generic
  kernels where justified, without claiming integers form a numerical field.
- Introduce a small real-scalar abstraction only for capabilities the algorithms
  need. Keep trait bounds and public types comprehensible.
- Expose validated nonnegative finite absolute/relative tolerances. Specify the
  scale for each comparison: singular-value thresholds, for example, use
  `atol + rtol * sigma_max`, with a dimension-aware default based on machine
  epsilon. Avoid one hardcoded epsilon for every algorithm and scalar type.
- Reject NaN/infinity at solver/decomposition entry points. Report invalid input,
  numerical rank deficiency, and nonconvergence distinctly. Document conditioning
  and tolerance dependence rather than promising exact floating-point rank.
- Use pivoting, Householder transformations, scaled norms, and bounded iterative
  algorithms. Solve through decompositions; do not compute an inverse to solve a
  system and do not form normal equations for least squares.
- Decomposition objects retain factors and permutations for repeated solves with
  vector-shaped or multiple-column RHS matrices. Convenience methods delegate to
  them. Prefer economy-size factors, with full forms available explicitly where
  useful. Specify orientation and permutation conventions in documentation.

## Milestone 1 — Repair the existing core

1. Correct inverted `is_empty`.
2. Validate row and column separately in `get` and `get_mut`; make tuple indexing
   use those checks. Test that an invalid column cannot alias the next row.
3. Require identical shapes in `try_zip_map` before invoking the callback.
   Verify all dependent operations reject equal-length/different-shape operands,
   shorter RHS storage, and longer RHS storage without silently truncating.
4. Check shape products for constructors, special matrices, and multiplication
   output. Return an error from fallible APIs; do not allow debug/release behavior
   to diverge in structural validation.
5. Make scalar and elementwise division use the same zero-divisor policy and
   shape-first validation. Add dedicated numeric-edge regressions.
6. Implement the agreed behavior for zero-width row iteration and empty shapes.
7. Repair `test_element_div_mul` to inspect returned matrices and reach division
   checks. Replace the empty `-1..-16` fixture with the intended negative values.
8. Replace weak `is_ok()`/panic-only tests with exact values, shapes, and error
   variants. Check full row/column traversal and output independence from inputs.

Acceptance: existing tests and regression tests pass in debug and release. Test
nonempty and empty shapes, out-of-range coordinates, shape overflow without huge
allocations, invalid operations, exact integer arithmetic, and rectangular
multiplication with independently computed expected results.

## Milestone 2 — Make the core API consistent

1. Organize implementation into focused modules for storage/access, arithmetic,
   structural operations, numerical operations, and later decompositions. Retain
   a small, documented public surface through crate-root re-exports.
2. Add validated vector-backed construction without unnecessary copying,
   `from_fn`, filled matrices, special matrices, nested array/row conversion,
   and consuming `into_vec`. Reject ragged nested inputs. Define an empty nested
   input as `0×0`; explicit dimensions express other empty shapes.
3. Add `shape`, `is_square`, immutable/mutable slices, consuming storage access,
   checked setters, row/column access, and consistent iterator names. Retain
   existing aliases where useful during migration.
4. Finish immutable, mutable, and consuming element iteration; add safe row and
   column traversal. Verify iterator lengths, exhaustion, reverse traversal where
   supported, and mutation without aliasing.
5. Standardize mapping and zipped mapping with `FnMut`, meaningful fallibility,
   and minimal bounds. Add conversion helpers with explicit failure for lossy or
   out-of-range numeric conversions. Document the transition from the current
   structurally infallible `try_scalar_mul` and similar wrappers: ordinary scalar
   transforms use ordinary methods; numeric overflow checks belong to the
   separately named checked methods.
6. Finish operator ergonomics, scalar add/subtract/multiply/divide methods,
   negation, and useful assignment variants. Specify whether each operation
   allocates. Structural errors must be detected before in-place mutation.
7. Publish error semantics, migration notes, and a compact API usage guide.

Acceptance: integration tests use only public APIs and cover integer, floating,
and nonnumeric storage/transformation cases. Demonstrate owned/borrowed operations
and compile-time rejection of unsupported scalar combinations. Preserve shape
invariants through every mutable public entry point.

## Milestone 3 — Complete matrix manipulation and basic operations

1. Add owned transpose, in-place transpose where supported, checked reshape,
   flattening, and row-major conversions. Reshape preserves element order and
   validates the new element count before mutation.
2. Add borrowed immutable/mutable matrix views with shape and strides, checked
   submatrix ranges, and a transpose view. Keep the borrowing model safe and avoid
   introducing `unsafe` merely for iterator convenience. Reuse operations through
   an internal shared access abstraction instead of copying every view.
3. Add owned slicing/extraction, row/column replacement and swaps, diagonal
   extraction/construction, horizontal/vertical concatenation, and checked block
   insertion. Define behavior for empty ranges and degenerate shapes.
4. Add explicit row/column-vector operations: dot and outer product, matrix-vector
   multiplication, and length validation. Reuse vector-shaped matrices rather
   than introducing an unrelated vector storage abstraction. No implicit
   broadcasting or complex conjugation is assumed.
5. Add trace, sum/product, min/max and their indices, row-wise and column-wise
   reductions, and numeric predicates. Row reductions produce `m×1`; column
   reductions produce `1×n`. Document tie-breaking and empty-reduction behavior.
6. Add square matrix powers by exponentiation by squaring. Exponent zero returns
   identity; negative powers are a real-valued extension after inverse exists.

Acceptance: transpose twice restores the matrix; reshape preserves flattened
order; views agree with owned extraction; submatrix mutations affect only the
selected region; concatenation, reductions, dot/outer products, and powers match
small independent fixtures. Exercise rectangular and zero-sized cases.

## Milestone 4 — Add reliable numerical primitives

1. Add checked integer elementwise/scalar operations, negation, division, and
   matrix multiplication. Define overflow errors without partial in-place writes
   for validation failures; document callback/custom-type panic behavior.
2. Add explicit casting, approximate comparisons, finite-value checks, means,
   floating elementwise helpers, and numeric predicate helpers.
3. Add stable vector 1/2/infinity norms, matrix Frobenius/1/infinity norms, and
   normalization with zero-vector handling. Reserve matrix spectral norm for SVD;
   avoid confusing it with the Frobenius norm.
4. Use scaled sum-of-squares for Euclidean/Frobenius norms and a documented
   accumulation strategy for floating sums, means, and dot products. Test
   cancellation and very large/small magnitudes.
5. Implement tolerance options and internal numerical helpers shared by later
   decompositions, including safe reflector and rotation construction.

Acceptance: exercise `f32` and `f64`, signed zero, nonfinite values, integer limits,
mixed magnitudes, zero norms, and approximate comparisons. Checked integer results
are profile-independent. Approximate comparisons do not treat NaN as equal.

## Milestone 5 — Square systems and reusable factorizations

1. Add lower/upper triangular solves, transpose solves, diagonal validation, and
   multiple RHS support; never divide through a rejected pivot.
2. Implement reusable LU with partial row pivoting. Define `P*A = L*U`, retain
   permutation parity, and document singular/numerically singular handling. Make
   singular status usable by determinant without forcing an invalid solve.
3. Implement `solve`, determinant, sign/log-absolute-determinant, and inverse
   through LU. Determinant must account for row swaps; inverse solves against
   identity. Real-valued factorization APIs must not silently truncate integers.
4. Implement reusable Cholesky for symmetric positive-definite matrices, including
   symmetry validation, solve, and log determinant. Report nonsquare,
   nonsymmetric, and non-positive-definite input clearly.
5. Document that determinant magnitude is not a reliable singularity test and
   expose diagnostics appropriate to the decomposition.

Acceptance: test `P*A ≈ L*U`, `A ≈ L*Lᵀ`, known determinants, pivot-requiring
systems, singular and nearly singular matrices, multiple RHS, and residual-based
solve validation. Verify inverse from both multiplication orders on suitable
well-conditioned fixtures. Cover triangular, `1×1`, and `0×0` cases.

## Milestone 6 — Rectangular systems and QR

1. Implement Householder QR, including column-pivoted QR for rank awareness.
   Document `A*P = Q*R` for pivoted QR. Store reflectors compactly and support
   applying `Q`/`Qᵀ` without materializing a large square matrix.
2. Expose economy/full factors where appropriate and decomposition reuse for
   multiple RHS. Handle tall and wide shapes deliberately.
3. Add overdetermined, full-column-rank least squares through QR and triangular
   solves. Return solution, residual information, rank diagnostics, and the
   algorithm/tolerance used. Do not return an unexplained arbitrary solution for
   rank-deficient or underdetermined input before SVD support is ready.

Acceptance: reconstruct pivoted and unpivoted inputs; verify orthogonality,
residual minimization, permutation recovery, and reflector application. Exercise
nearly dependent columns, rank-deficient inputs, tall/wide/empty shapes, and
multiple RHS with condition-aware tolerances.

## Milestone 7 — Advanced real-valued linear algebra

1. Implement SVD with Householder bidiagonalization and a bounded, scaled
   bidiagonal iteration such as implicit-shift QR. Handle wide inputs via a
   documented transpose strategy. Use sorted nonnegative singular values and
   economy shapes `U: m×r`, `s: r`, `Vᵀ: r×n`, where `r = min(m,n)`.
2. Expose iteration limits, convergence status/errors, and tolerance controls;
   do not silently return unconverged factors. Avoid deriving SVD from `Aᵀ*A`,
   which squares the condition number.
3. Add numerical rank, spectral norm, pseudoinverse, and 2-norm condition number.
   Define singular/zero/empty cases explicitly; do not silently divide by a zero
   singular value. Rank and truncation use documented thresholds.
4. Extend least squares with SVD for rank-deficient and underdetermined systems,
   returning the minimum-norm solution. Expose explicit algorithm selection and
   a documented automatic policy; keep square `solve` semantics distinct.
5. Add real symmetric eigendecomposition using symmetric tridiagonal reduction
   and a bounded orthogonal iteration. Return sorted eigenvalues and orthonormal
   eigenvectors; validate symmetry and support repeated eigenvalues.
6. Add negative real matrix powers using the already validated inverse path.

Acceptance: validate SVD reconstruction and orthogonality, all four
Moore–Penrose identities for pseudoinverses, numerical rank and conditioning,
minimum-norm solutions, and `A*V ≈ V*Λ` for symmetric eigenproblems. Cover repeated
singular/eigenvalues, zero/rank-deficient matrices, wide/tall shapes, and explicit
nonconvergence. Compare subspaces/reconstructions rather than raw signed vectors
when factors are nonunique.

## Milestone 8 — Performance, usability, and release readiness

1. Benchmark allocation-heavy operations, owned versus view access, rectangular
   multiplication, decompositions, and repeated solves. Include tiny and medium
   inputs, plus larger representative matrices within available resources.
2. Optimize only measured bottlenecks: reuse buffers, avoid accidental copies,
   add output-buffer operations with validated shape/aliasing rules, and consider
   cache blocking for multiplication. Preserve numerical and correctness checks.
   Native backends and parallelism require a later measured case for inclusion.
3. Add matrix `Display`, examples for every main workflow, module/crate docs, a
   README, API migration notes, and a numerical limitations guide. Document
   complexity, factor conventions, tolerance defaults, and error recovery.
4. Add optional serialization only with validated deserialization that preserves
   shape invariants. Define a stable row-major representation; do not derive an
   unchecked deserializer for private fields.
5. Establish an explicit supported Rust version, edition compatibility, CI, and
   package metadata. Audit dependency licenses and required license metadata;
   do not invent a repository license. Confirm licensing before an external crate
   publication if it remains unspecified.
6. Decide separately what to do with the placeholder binary; it must not be
   documented as a working calculator. Building a REPL is outside this library
   release. Run package verification locally; publishing requires a separate
   explicit instruction.

Acceptance: documented examples compile, library-only consumers can use the crate,
optional features pass their own tests, package verification succeeds, and
benchmark results contain no unexplained major regressions. Do not claim a
published crate or a finished calculator from these checks.

## Validation and completion gates

Tests accompany every implementation commit instead of being deferred until the
end. Use unit tests for kernels/errors and integration tests for the public API.
Add property-based generation for valid and invalid shapes, transposition,
arithmetic identities without integer overflow, and decomposition invariants.
Use deterministic seeds and keep generated matrix sizes practical.

For floating-point algorithms, assert scaled residuals and backward error on
well-conditioned fixtures rather than expecting exact equality or forward
accuracy on ill-conditioned systems. Test both `f32` and `f64`, extreme scales,
singular cases, and failure paths. Keep tolerances justified by epsilon, size,
scale, and conditioning. Never weaken tolerances merely to make failures pass.

Use a mature independent implementation, such as `nalgebra`, only as a development
oracle, supplemented by analytic and fixed fixtures. Keep production algorithms
in this library. Property testing, comparison, approximation, and benchmarking
dependencies should be development-only unless the public API truly needs them.

The intended milestone checks, once tooling is available, are:

```sh
cargo fmt --check
cargo test --locked
cargo test --locked --release
cargo test --locked --all-features
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo doc --locked --no-deps --all-features
```

Also test supported minimal feature combinations and documentation examples;
after optional features exist, test their combinations and release behavior as
appropriate. Verify the declared minimum Rust version in CI. Use Miri for any
nontrivial borrowing/unsafe code if introduced. Check package contents with
`cargo package --list` and use verified `cargo package` before release. New
dependencies intentionally update the lockfile; existing installs use it.

A milestone is complete only when its agreed API, meaningful checks, docs, and
error paths work. Maintain a green `dev` after the initial repair. The library is
complete for this scope when milestones 1–8 pass their gates, numerical
limitations are documented, and remaining follow-on work is explicitly recorded.

## Execution checklist

- [x] Inspect the current Rust checkout and baseline.
- [x] Create `dev` and verify the requested local author/committer identity.
- [x] Record this plan without changing implementation files.
- [x] Milestone 1: repair current behavior and tests.
- [x] Milestone 2: settle and implement the consistent core API.
- [x] Milestone 3: structural operations, views, vectors, and reductions.
- [x] Milestone 4: checked arithmetic and numerical primitives.
- [x] Milestone 5: LU, Cholesky, square solves, determinant, and inverse.
- [x] Milestone 6: QR and full-rank least squares.
- [x] Milestone 7: SVD, pseudoinverse, rank, conditioning, and symmetric eigen.
- [x] Milestone 8: measured optimization, documentation, packaging, and release gates.


## Implementation record

All eight milestones are implemented for the release scope above. Source is
organized into storage/access, arithmetic, structural operations, views,
numerics, validated serialization, and decomposition modules. Migration notes,
examples, CI, and numerical limitations are included. The placeholder greeting
binary was removed to make the release library-only. Runtime dependencies remain
pure Rust; the independent numerical oracle and property generator are dev-only.

Algorithm adjustment: SVD uses scaled one-sided Jacobi on the input directly,
with a transpose strategy for wide matrices. Symmetric eigen uses scaled cyclic
Jacobi. These replace the planned bidiagonal/tridiagonal iterations while retaining
bounded convergence, orthogonality, and residual validation; neither forms normal
equations. The choice, convergence policy, performance tradeoffs, and numerical
limits are documented in NUMERICS.md. The direct algorithms passed independent
oracle comparisons and tests of nonunique/rank-deficient cases.

Fixed-seed generated tests cover 128 structural cases, 128 SVD/QR/least-squares
cases, 96 LU/Cholesky/symmetric-eigen cases, and 64 single-precision cases, in
addition to analytic,
boundary, error-path, serialization, and documentation examples. BENCHMARKS.md
records measured improvements from the shared contiguous multiplication kernel
and allocation reuse, plus longer steady-state samples through 256×256; the
strided path keeps the general safe implementation. Six targeted borrowing and
structural tests also passed under Miri on the isolated 2026-10-29 nightly.

Release validation and exact results are recorded in VALIDATION.md. The owner
subsequently selected dual `MIT OR Apache-2.0` licensing and authorized pushing
`dev` to the remote. Both license texts and Cargo's SPDX declaration are included.
Crate publication remains a separate action.
