# matrix-calc

A Rust library for generic dense matrices and real-valued numerical linear
algebra. Matrices own a flat row-major buffer; dimensions and storage length stay
consistent through the safe public API. The numerical layer supports `f32` and
`f64`. Rust 1.85 or newer is required.

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
Mutable views consume or reborrow exclusive access, and the crate forbids unsafe
code. `to_owned` copies a view. Owned `into_transpose` moves elements, while
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
[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md). This crate is a library; calculator
parsing and a REPL are separate work. The repository currently has no declared
license; a license decision is required before external publication.
