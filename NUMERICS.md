# Numerical conventions and limits

## Scalar and error policy

Generic storage and structural operations support nonnumeric elements. Ordinary
arithmetic uses the element's Rust operators, with no implicit promotion.
Primitive integer overflow checking is explicit through `checked_*` methods.
Numerical decompositions are implemented for real `f32` and `f64`; integers are
never silently treated as fields. Complex decompositions are outside this release.

Decomposition/solver inputs and right-hand sides must be finite. Invalid shapes,
nonfinite values, singularity, rank deficiency, failed symmetry/positive
definiteness, invalid options, and nonconvergence have distinct errors.
Overflowed computed solutions/factors return `NonFinite`. General elementwise
floating operations retain IEEE semantics; validated division additionally
rejects zero. A mathematically valid but unrepresentable result can fail.

## Tolerances

`Tolerance::new(atol, rtol)` requires finite nonnegative values. Approximate
comparison uses `|a-b| <= atol + rtol*max(|a|,|b|)` with scaling to avoid overflow.
NaN/infinity never compare approximately equal. Default decomposition relative
tolerance is `8*max(dimension,1)*epsilon`, with absolute tolerance zero.

LU tests scaled pivots against `atol/scale + rtol`; Cholesky applies that threshold
to its scaled diagonal Schur complements. Symmetry compares corresponding entries
using their local scale. QR diagnoses rank using diagonal magnitudes; pivoted QR
is needed for a useful rank estimate. Unpivoted QR's diagonal count is not a
rank-revealing diagnostic. `Matrix::rank` uses SVD instead. SVD rank/pseudoinverse
truncation uses `atol + rtol*sigma_max`. Rank is numerical and tolerance-dependent.

SVD convergence uses a separate machine-precision column correlation threshold
`4*max(m,1)*epsilon`; changing rank tolerance does not lower factor accuracy.
Symmetric eigen convergence uses the requested absolute/relative matrix-scale
threshold, with a machine-epsilon floor. Iterative options bound complete sweeps.
Zero iteration limits are invalid; exhausted limits return `NonConvergence`.

## Algorithms and factor orientation

- **LU:** global input scaling and partial row pivoting; `P*A = L*U`.
  `permutation[i]` is the original row at factor row `i`. The stored lower factor
  has unit diagonal. `solve_transpose` applies inverse permutations correctly.
- **Determinant:** obtained from LU signs, permutation parity, and logarithms of
  pivots. Exact zero pivots give determinant zero; the numerical solve threshold
  does not forcibly round every small determinant to zero. `sign_log_determinant`
  avoids overflow/underflow of determinant magnitude. Magnitude alone is not a
  singularity or conditioning test.
- **Cholesky:** symmetric positive-definite input, `A=L*Lᵀ`; internal scaling is
  undone when factors are exposed. Tiny positive pivots can be rejected at the
  selected tolerance. Near-symmetric inputs use the lower triangle after symmetry
  validation. Solves use both triangular factors.
- **QR:** unit Householder reflectors, optionally with column pivoting;
  `A*P=Q*R`. `permutation[j]` names the original column in position `j`.
  Reflectors apply Q/Qᵀ directly. `q`/`r` are economy factors; full factors have
  separate methods. QR least squares requires full column rank and m>=n.
- **SVD:** scaled, cyclic one-sided Jacobi rotations directly on A. Wide input is
  transposed. This deliberately replaces the original planned bidiagonal QR
  algorithm with a smaller direct orthogonalization implementation, validated
  independently. No Gram matrix is formed. With r=min(m,n), U is m×r, singular
  values have length r, and Vᵀ is r×n. Exact null columns get an orthonormal basis
  completion. Sorted singular values are nonnegative.
- **Symmetric eigen:** scaled cyclic two-sided Jacobi rotations, replacing the
  planned tridiagonal iteration. `A*V=V*diag(values)`; eigenvalues ascend and
  eigenvectors are columns. Within-tolerance asymmetry is symmetrized. Iteration
  is bounded and independent residual tests verify the result.
- **Least squares:** Auto uses column-pivoted QR for full-column-rank tall/square
  input and SVD otherwise. SVD returns the truncated minimum-norm solution. The
  result includes the selected method, numerical rank, tolerance, and combined
  Frobenius residual across all RHS. Explicit QR refuses deficient/wide input.
  Normal equations and inverse-based solves are never used.

The 2-norm condition number is largest/smallest retained full-rank singular value;
it is infinity for numerical deficiency and one for an empty spectrum. A zero
nonempty matrix has infinite condition number. Spectral and Frobenius norms are
separate methods. Solve square systems with `solve`; use `least_squares` when
rectangular or rank-deficient semantics are intended.

## Accuracy and validation

Norms use hypot-based scaling. Compensated sums/dots reduce cancellation; means
avoid overflowing the sum before division, and normalization scales before taking
a norm. Ordinary matrix multiplication retains the scalar's accumulation rules.
None of these techniques can recover information already lost in the input.
Global scaling cannot preserve ratios beyond the scalar's dynamic range; extremely
small singular values can underflow. A zero rank tolerance does not turn floating
arithmetic into exact arithmetic. Ill-conditioned systems can have substantial
forward error despite a small backward residual. `f32` is less accurate than `f64`.

Tests check reconstructability, orthogonality, solve residuals, Moore–Penrose
identities, repeated spectra, shape/empty semantics, large/small scales, invalid
input, and bounded-iteration failures. Fixed-seed generated cases compare singular
values, eigenvalues, least-squares solutions, and determinants against an independent
development-only implementation. Factors are nonunique: signs and bases within
repeated-value subspaces can differ. Tests compare identities/subspaces, not raw
vectors with arbitrarily chosen signs.

Jacobi is intentionally a correctness-oriented baseline for small and medium dense
matrices. Its cost can exceed bidiagonal/tridiagonal or native implementations at
large sizes. Performance figures are local measurements, not cross-machine
performance guarantees. General nonsymmetric eigenproblems, complex factors,
sparse matrices, and native/GPU backends remain follow-on projects.
