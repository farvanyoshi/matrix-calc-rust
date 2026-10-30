# matrix-calc

A Rust library and interactive calculator for generic dense matrices, vectors,
strided n-dimensional tensors, and real-valued numerical linear algebra. Matrices own a flat row-major buffer; dimensions and storage length stay
consistent through the safe public API. The numerical layer supports `f32` and
`f64`. Rust 1.85 or newer is required.

The companion [Nim implementation](https://github.com/farvanyoshi/matrix-calc-nim)
provides similar functionality. See [roadmap coverage](docs/ROADMAP.md) and
[verification results and limitations](VALIDATION.md) for this version.

## Vectors and n-dimensional tensors

```rust
use matrix_calc::{Tensor, Vector};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Vector::from([1.0, 0.0, 0.0]);
    let y = Vector::from([0.0, 1.0, 0.0]);
    assert_eq!(x.cross(&y)?.to_vec(), vec![0.0, 0.0, 1.0]);
    let a = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3])?;
    let row = Tensor::new(vec![10, 20, 30], vec![3])?;
    assert_eq!(a.try_add(&row)?.to_vec(), vec![11, 22, 33, 14, 25, 36]);
    let view = a.transpose(&[1, 0])?;
    assert_eq!(view.shape(), &[3, 2]);
    assert!(!view.is_contiguous());
    assert_eq!(view.make_contiguous().to_vec(), vec![1, 4, 2, 5, 3, 6]);
    Ok(())
}
```

`Vector<T>` uses direct contiguous 1-D kernels for dot and length-three cross
products; float vectors provide L1/L2/infinity norms and scaled normalization.
Convert using row/column matrices or rank-one tensors. `Tensor<T>` stores shape,
strides and an offset. Rank zero holds one scalar; zero axes hold no elements.
Arithmetic broadcasts trailing dimensions when they are equal or one is 1.
Matrix arithmetic remains strict about dimensions. Tensor `*` is elementwise;
Matrix `*` is matrix multiplication.

Tensor transpose/permutation and positive-step half-open slices retain backing
storage without copying. Reshape requires contiguity; use `make_contiguous` first
for strided layouts. Tensor handles use copy-on-write: `set` detaches shared
storage and does **not** mutate the original tensor. Mutable Matrix views instead
borrow their parent exclusively. Conversion between tensors, matrices and vectors
uses logical element order and validates rank. Small tensor displays are nested;
large/deep displays are bounded previews.

## Interactive calculator

Run `cargo run --release --bin matrix-calc`, or pipe a script into it:

```text
A = [[1,2],[3,4]]
B = A * transpose(A)
C = A + [10,20]
r = Matrix.random(3,3,7)
v = cross([1,0,0],[0,1,0])
reshape(B,[4])
vars
quit
```

The parser supports numbers, nested rectangular arrays, variables, assignment,
parentheses and `+ - * /`. All runtime values are float64 tensors. `*` multiplies
two rank-two values as matrices and otherwise broadcasts elementwise; use
`hadamard(A,B)` for explicit elementwise multiplication. Functions include
`matrix`, `vector`, `tensor(data,shape)`, seeded `random`, `zeros`, `ones`, `eye`,
`shape`, `transpose`, `reshape`, `slice`, `dot`, `cross`, `norm1`, `norm2`, and
`sum`. `help` lists signatures. `vars`, `clear`, `quit`/`exit` manage the session.
Errors preserve existing variables and the next line can run normally. This is a
small expression grammar, not host-language evaluation. Input/depth/output limits
prevent accidental giant interactive expressions; library APIs remain unrestricted
by those interactive limits. Random sequences are reproducible within a language;
Nim and Rust use different PRNGs.

## Execution backends and performance

For `Matrix<f64>`, `matmul_blocked(rhs, tile)`, `matmul_parallel(rhs, tile)` and
`matmul_simd(rhs)` are explicit execution choices. Tile sizes must be positive and
all products require A(M,K), B(K,N). Rayon partitions disjoint row tiles; set
`RAYON_NUM_THREADS` to control its pool. Runtime-dispatched AVX (x86) and NEON
(AArch64) intrinsics handle full lanes and scalar tails, with a portable fallback.
The normal generic matrix API remains available. SIMD dot reductions can differ
slightly in floating-point order.

`cargo bench --locked --bench roadmap` uses Criterion at 128², 512² and 1024².
`ROADMAP_MODE=verify` performs one timed correctness comparison; `quick` selects
128², and `ROADMAP_SIZES` overrides sizes. See [roadmap results](docs/ROADMAP.md),
[profiling instructions](docs/PROFILING.md), and the actual
[sampled flamegraph](docs/profiling/flamegraph.svg). AArch64 is compile-checked;
AVX and scalar fallback are exercised on the current x86 host, not on ARM hardware.

## Construct, transform, and multiply

```rust
use matrix_calc::{Matrix, Result};
fn main() -> Result<()> {
    let a = Matrix::from([[1, 2, 3], [4, 5, 6]]);
    let gram = a.transpose().try_matmul(&a)?;
    assert_eq!(gram, Matrix::from([[17, 22, 27], [22, 29, 36], [27, 36, 45]]));
    assert_eq!(a.row_sums(), Matrix::from([[6], [15]]));
    assert!(a.get(0, 3).is_none());
    Ok(())
}
```

`Matrix::try_new(data, rows, cols)` validates an iterator; `try_from_vec` takes a
buffer without copying; `try_from_rows` rejects ragged input. `try_from_fn`,
`try_filled`, `try_zeros`, `try_ones`, and `try_identity` check dimensions and
reserve their storage fallibly. Their ordinary convenience forms panic on these
errors. `From<[[T; C]; R]>` supports nonnumeric and non-`Copy` elements too.

## Solve systems through reusable factorizations

```rust
use matrix_calc::{Matrix, Result, Tolerance};
fn main() -> Result<()> {
    let a = Matrix::from([[3.0_f64, 1.0], [1.0, 2.0]]);
    let b = Matrix::from([[9.0], [8.0]]);
    let lu = a.lu()?;
    let x = lu.solve(&b)?;
    assert!(x.approx_eq(&Matrix::from([[2.0], [3.0]]), Tolerance::new(1e-12, 1e-12)?));
    assert!(a.matmul(&x).approx_eq(&b, Tolerance::new(1e-12, 1e-12)?));
    let inverse = lu.inverse()?;
    assert_eq!(inverse.shape(), (2, 2));
    Ok(())
}
```

LU uses partial row pivoting. Cholesky handles symmetric positive-definite
systems. Both retain factors for repeated solves and multiple RHS columns.
`solve_lower`/`solve_upper` handle triangular systems. `Lu::solve_transpose`
solves the transposed system without refactorization.

## Least squares, SVD, and eigenproblems

```rust
use matrix_calc::{Matrix, Result, Tolerance, LeastSquaresMethod};
fn main() -> Result<()> {
    let a = Matrix::from([[1.0_f64, 1.0]]);
    let fit = a.least_squares(&Matrix::from([[2.0]]))?;
    assert_eq!(fit.method, LeastSquaresMethod::Svd);
    assert!(fit.solution.approx_eq(&Matrix::from([[1.0], [1.0]]), Tolerance::new(1e-12, 1e-12)?));
    assert_eq!(fit.rank, 1);
    let pinv = a.pseudoinverse()?;
    assert_eq!(pinv.shape(), (2, 1));
    let eigen = Matrix::from([[2.0_f64, 1.0], [1.0, 2.0]]).symmetric_eigen()?;
    assert_eq!(eigen.eigenvalues().len(), 2);
    Ok(())
}
```

Full-column-rank tall least squares uses column-pivoted Householder QR. Wide and
rank-deficient systems use SVD and return a minimum-norm solution. Explicit
`LeastSquaresMethod` selection is available. `Svd` also provides rank, spectral
norm, condition number, and pseudoinverse. Symmetric eigenvalues are ascending;
singular values are descending. See [NUMERICS.md](NUMERICS.md) for conventions,
tolerances, algorithms, and limitations.

## Borrow views and mutate safely

```rust
use matrix_calc::{Matrix, Result};
fn main() -> Result<()> {
    let mut a = Matrix::from([[1, 2, 3], [4, 5, 6]]);
    {
        let mut corner = a.slice_mut(0..2, 1..3)?.transpose();
        corner[(0, 1)] = 9;
    }
    assert_eq!(a[(1, 1)], 9);
    let view = a.view().transpose();
    assert_eq!(view.shape(), (3, 2));
    let gram = view.try_matmul(&a)?;
    assert_eq!(gram.shape(), (3, 3));
    Ok(())
}
```

Views are borrowed strided rectangles; slicing and transposition avoid copying.
Mutable matrix views consume or reborrow exclusive access. Unsafe code is denied
by default; only the audited architecture intrinsics in `accelerated` are allowed. `to_owned` copies a view. Owned `into_transpose` moves elements, while
`transpose_in_place` currently requires a square shape. Reshape preserves flat
row-major order. Row/column replacement, swaps, diagonal construction/extraction,
concatenation, and block insertion validate their boundaries.

## Arithmetic and errors

- Matrix `*` means matrix multiplication; `hadamard` is elementwise.
- Matrix `+` and `-` require identical shapes; there is no implicit broadcasting.
- Scalar multiplication/division have operators and named methods. Named scalar
  addition/subtraction are available. Borrowed and owned matrix operands work.
- `get`/`get_mut` return `Option`; `try_get`/`try_get_mut` return detailed index
  errors. Tuple indexing panics on invalid coordinates.
- `try_*` arithmetic returns structural errors. Ordinary element arithmetic
  follows the scalar's own semantics. Integer division truncates. Validated
  scalar/element division rejects zero divisors, including signed floating zero.
- `checked_*` methods for primitive integers return overflow/zero-divisor errors,
  including signed minimum divided by minus one. Checked assignment computes
  before replacing its destination. Ordinary assignment can partially modify a
  destination if custom scalar code panics.
- `map` accepts `FnMut`; `try_map` accepts a fallible callback and stops at the
  first error. `try_zip_map` checks shape before invoking its ordinary callback.
- Fallible allocation uses `try_reserve` on explicitly fallible constructors.
  User callbacks, element cloning/arithmetic, and ordinary allocating convenience
  methods retain Rust's usual panic/allocation behavior. There is no promise of
  catching arbitrary panics or globally recovering from out-of-memory aborts.

```rust
use matrix_calc::{Matrix, Result};
fn main() -> Result<()> {
    let a = Matrix::from([[i8::MAX, 1]]);
    assert!(a.checked_scalar_add(1).is_err());
    let mapped = a.try_map(|x| x.checked_mul(2).ok_or("overflow"));
    assert_eq!(mapped.unwrap_err(), "overflow");
    Ok(())
}
```

## Empty shapes and reductions

`0×0`, `m×0`, and `0×n` are distinct shapes. Row iteration yields exactly `m`
rows even when they are empty. `(m×0)*(0×n)` produces `m×n` zeros. Empty sums are
zero and products one; empty mean/min/max are errors when a reduction group
exists. Row reductions produce `m×1`, column reductions `1×n`. `argmin`/`argmax`
choose the first row-major occurrence; NaN is rejected in ordered reductions.
Trace and powers require square matrices. For `0×0`, determinant is one, trace
zero, and inverse `0×0`. Empty SVD has rank/norm zero and condition number one.

## Optional serialization

Enable `features = ["serde"]` for `{ "rows": ..., "cols": ..., "data": [...] }`
serialization. Data is row-major. Deserialization rejects unknown fields,
dimension overflow, and element-count mismatches. Default builds do not depend
on serde. There are no native BLAS, service, or application runtime dependencies.

## Development

Run `cargo test --locked`, `cargo test --locked --release --all-features`,
`cargo clippy --locked --all-targets --all-features -- -D warnings`, and
`cargo fmt --check`. Documentation examples are executable tests. Generated tests
use fixed seeds and compare algorithms with development-only `nalgebra`.
Run `cargo bench --locked --bench kernels` for reproducible kernel measurements.
See [BENCHMARKS.md](BENCHMARKS.md) and [MIGRATION.md](MIGRATION.md).

The completion checklist and algorithm decisions are recorded in
[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md). The crate includes both the library and the `matrix-calc` calculator binary.

## License

Copyright (c) 2026 farvanyoshi <nvnjoshi@outlook.com>.

Licensed under either the [MIT License](LICENSE-MIT) or the
[Apache License, Version 2.0](LICENSE-APACHE), at your option.
