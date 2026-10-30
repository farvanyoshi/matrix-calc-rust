//! Reusable real-valued decompositions. Inputs must be finite; no normal equations
//! or inverse-based solves are used. Factor conventions are documented on types.
mod cholesky;
mod eigen;
mod lu;
mod qr;
mod svd;
use crate::{
    DecompositionOptions, Matrix, Real,
    error::{NumericError, Result},
};
pub use cholesky::Cholesky;
pub use eigen::SymmetricEigen;
pub use lu::Lu;
pub use qr::Qr;
pub use svd::Svd;

pub(crate) fn normalized<T: Real>(a: &Matrix<T>) -> Result<(Matrix<T>, T)> {
    a.finite()?;
    let scale = if a.max_abs() == T::zero() {
        T::one()
    } else {
        a.max_abs()
    };
    Ok((a.scalar_div(scale), scale))
}
pub(crate) fn rhs<T: Real>(rows: usize, b: &Matrix<T>) -> Result<()> {
    b.finite()?;
    if b.rows() != rows {
        return Err(crate::error::ShapeError::Dimension {
            op: "solve",
            l_rows: rows,
            l_cols: rows,
            r_rows: b.rows(),
            r_cols: b.cols(),
        }
        .into());
    }
    Ok(())
}
impl<T: Real> Matrix<T> {
    /// Compute scaled partial-row-pivoted LU of finite square input.
    /// Shape requirements: input is finite N×N.
    pub fn lu(&self) -> Result<Lu<T>> {
        Lu::decompose(self, DecompositionOptions::default())
    }
    /// Compute LU with an explicit numerical singularity threshold.
    /// Shape requirements: input is finite N×N.
    pub fn lu_with_options(&self, options: DecompositionOptions<T>) -> Result<Lu<T>> {
        Lu::decompose(self, options)
    }
    /// Solve a finite square system with one or more right-hand-side columns using LU.
    /// Shape requirements: retained A=N×N and B=N×K; A is numerically nonsingular.
    pub fn solve(&self, b: &Self) -> Result<Self> {
        self.lu()?.solve(b)
    }
    /// Compute a square determinant; its magnitude can underflow or overflow.
    /// Shape requirements: input is finite N×N.
    pub fn determinant(&self) -> Result<T> {
        Ok(self.lu()?.determinant())
    }
    /// Returns (sign, log(abs(det))). Exact singularity gives (0, -infinity).
    /// Shape requirements: input is finite N×N.
    pub fn sign_log_determinant(&self) -> Result<(T, T)> {
        Ok(self.lu()?.sign_log_determinant())
    }
    /// Solve against identity to invert finite nonsingular square input.
    /// Shape requirements: input is finite invertible N×N.
    pub fn inverse(&self) -> Result<Self> {
        self.lu()?.inverse()
    }
    /// Factor finite symmetric positive-definite input with the default tolerance.
    /// Shape requirements: input is finite symmetric positive-definite N×N.
    pub fn cholesky(&self) -> Result<Cholesky<T>> {
        Cholesky::decompose(self, DecompositionOptions::default())
    }
    /// Compute Cholesky with explicit symmetry and pivot tolerances.
    /// Shape requirements: input is finite symmetric positive-definite N×N.
    pub fn cholesky_with_options(&self, options: DecompositionOptions<T>) -> Result<Cholesky<T>> {
        Cholesky::decompose(self, options)
    }
    /// Compute an unpivoted compact Householder QR of finite input.
    /// Shape requirements: input is finite M×N; economy factors use min(M,N).
    pub fn qr(&self) -> Result<Qr<T>> {
        Qr::decompose(self, false, DecompositionOptions::default())
    }
    /// Compute compact column-pivoted Householder QR of finite input.
    /// Shape requirements: input is finite M×N; economy factors use min(M,N).
    pub fn pivoted_qr(&self) -> Result<Qr<T>> {
        Qr::decompose(self, true, DecompositionOptions::default())
    }
    /// Compute QR with explicit column-pivoting and rank-threshold settings.
    /// Shape requirements: input is finite M×N; economy factors use min(M,N).
    pub fn qr_with_options(&self, pivot: bool, options: DecompositionOptions<T>) -> Result<Qr<T>> {
        Qr::decompose(self, pivot, options)
    }
    /// Compute the economy SVD of finite input with bounded default iteration.
    /// Shape requirements: input is finite M×N, including empty axes.
    pub fn svd(&self) -> Result<Svd<T>> {
        Svd::decompose(self, DecompositionOptions::default())
    }
    /// Compute economy SVD with explicit rank tolerance and sweep limit.
    /// Shape requirements: input is finite M×N, including empty axes.
    pub fn svd_with_options(&self, options: DecompositionOptions<T>) -> Result<Svd<T>> {
        Svd::decompose(self, options)
    }
    /// Compute a tolerance-truncated Moore–Penrose pseudoinverse using SVD.
    /// Shape requirements: finite M×N input produces an N×M output.
    pub fn pseudoinverse(&self) -> Result<Self> {
        self.svd()?.pseudoinverse()
    }
    /// Numerical rank from singular values above the selected threshold.
    /// Shape requirements: any finite M×N input or its retained factorization.
    pub fn rank(&self) -> Result<usize> {
        Ok(self.svd()?.rank())
    }
    /// Matrix spectral norm (largest singular value), distinct from the Frobenius norm.
    /// Shape requirements: any finite M×N input.
    pub fn norm_2(&self) -> Result<T> {
        Ok(self.svd()?.spectral_norm())
    }
    /// SVD-based 2-norm condition number; numerical deficiency gives infinity, empty input gives one.
    /// Shape requirements: any finite M×N input or retained SVD.
    pub fn condition_number(&self) -> Result<T> {
        Ok(self.svd()?.condition_number())
    }
    /// Compute ascending real eigenvalues and orthonormal column eigenvectors of symmetric input.
    /// Shape requirements: input is finite symmetric N×N.
    pub fn symmetric_eigen(&self) -> Result<SymmetricEigen<T>> {
        SymmetricEigen::decompose(self, DecompositionOptions::default())
    }
    /// Compute symmetric eigenpairs with explicit symmetry/convergence tolerance and sweep limit.
    /// Shape requirements: input is finite symmetric N×N.
    pub fn symmetric_eigen_with_options(
        &self,
        options: DecompositionOptions<T>,
    ) -> Result<SymmetricEigen<T>> {
        SymmetricEigen::decompose(self, options)
    }
    /// Signed integer power of square input; negative exponents first invert the matrix.
    /// Shape requirements: input is square N×N and invertible when exponent is negative.
    pub fn powi(&self, power: i32) -> Result<Self> {
        if power < 0 {
            self.inverse()?.pow(power.unsigned_abs())
        } else {
            self.pow(power as u32)
        }
    }
    /// Solve with the lower triangle; `unit_diagonal` treats diagonal entries as one.
    /// Shape requirements: A=N×N, B=N×K; the lower triangle and selected diagonal convention are used.
    pub fn solve_lower(&self, b: &Self, unit_diagonal: bool) -> Result<Self> {
        triangular(
            self,
            b,
            true,
            unit_diagonal,
            DecompositionOptions::default(),
        )
    }
    /// Solve with the upper triangle; `unit_diagonal` treats diagonal entries as one.
    /// Shape requirements: A=N×N, B=N×K; the upper triangle and selected diagonal convention are used.
    pub fn solve_upper(&self, b: &Self, unit_diagonal: bool) -> Result<Self> {
        triangular(
            self,
            b,
            false,
            unit_diagonal,
            DecompositionOptions::default(),
        )
    }
    /// Solve using a selected triangle and diagonal convention; the unused triangle is ignored.
    /// Shape requirements: A=N×N, B=N×K; the selected triangle determines the solve.
    pub fn solve_triangular_with_options(
        &self,
        b: &Self,
        lower: bool,
        unit_diagonal: bool,
        options: DecompositionOptions<T>,
    ) -> Result<Self> {
        triangular(self, b, lower, unit_diagonal, options)
    }
}
fn triangular<T: Real>(
    a: &Matrix<T>,
    b: &Matrix<T>,
    lower: bool,
    unit: bool,
    options: DecompositionOptions<T>,
) -> Result<Matrix<T>> {
    let n = a.square()?;
    a.finite()?;
    rhs(n, b)?;
    let tol = options.resolve(n)?.threshold(if unit {
        a.max_abs().max(T::one())
    } else {
        a.max_abs()
    });
    if !unit && (0..n).any(|i| a[(i, i)].abs() <= tol) {
        return Err(NumericError::Singular.into());
    }
    let mut x = b.clone();
    for step in 0..n {
        let i = if lower { step } else { n - 1 - step };
        for c in 0..b.cols() {
            let mut value = x[(i, c)];
            if lower {
                for k in 0..i {
                    value -= a[(i, k)] * x[(k, c)];
                }
            } else {
                for k in i + 1..n {
                    value -= a[(i, k)] * x[(k, c)];
                }
            }
            x[(i, c)] = if unit { value } else { value / a[(i, i)] };
        }
    }
    x.finite()?;
    Ok(x)
}

/// Explicit least-squares method; Auto uses pivoted QR for full-rank tall input,
/// and SVD for wide or numerically deficient input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeastSquaresMethod {
    /// Prefer pivoted QR for full-column-rank tall input; otherwise use SVD.
    Auto,
    /// Require full-column-rank tall input and use pivoted Householder QR.
    Qr,
    /// Use SVD and return the truncated minimum-norm solution.
    Svd,
}
/// Solution and diagnostics. Residual is the Frobenius norm across all RHS.
#[must_use]
#[derive(Clone, Debug)]
pub struct LeastSquares<T> {
    /// Computed solution matrix with one column per RHS.
    pub solution: Matrix<T>,
    /// Combined Frobenius norm of A*solution - B.
    pub residual_norm: T,
    /// Numerical rank reported by the selected decomposition.
    pub rank: usize,
    /// Algorithm actually used; never Auto in a successful result.
    pub method: LeastSquaresMethod,
    /// Resolved tolerance used for the solve.
    pub tolerance: crate::Tolerance<T>,
}
impl<T: Real> Matrix<T> {
    /// Auto-select pivoted QR or SVD and return solution, rank, method, and residual diagnostics.
    /// Shape requirements: A=M×N and B=M×K; returns an N×K solution.
    pub fn least_squares(&self, b: &Self) -> Result<LeastSquares<T>> {
        self.least_squares_with_options(
            b,
            LeastSquaresMethod::Auto,
            DecompositionOptions::default(),
        )
    }
    /// Least squares with explicit algorithm, tolerance, and iteration limit.
    /// Shape requirements: A=M×N and B=M×K; explicit QR requires M≥N and full column rank.
    pub fn least_squares_with_options(
        &self,
        b: &Self,
        method: LeastSquaresMethod,
        options: DecompositionOptions<T>,
    ) -> Result<LeastSquares<T>> {
        rhs(self.rows(), b)?;
        let tolerance = options.resolve(self.rows().max(self.cols()))?;
        let (solution, rank, used) =
            if method == LeastSquaresMethod::Svd || self.rows() < self.cols() {
                if method == LeastSquaresMethod::Qr {
                    return Err(NumericError::RankDeficient.into());
                }
                let svd = self.svd_with_options(options)?;
                (svd.solve(b)?, svd.rank(), LeastSquaresMethod::Svd)
            } else {
                let qr = self.qr_with_options(true, options)?;
                if qr.rank() == self.cols() {
                    (qr.solve(b)?, qr.rank(), LeastSquaresMethod::Qr)
                } else {
                    if method == LeastSquaresMethod::Qr {
                        return Err(NumericError::RankDeficient.into());
                    }
                    let svd = self.svd_with_options(options)?;
                    (svd.solve(b)?, svd.rank(), LeastSquaresMethod::Svd)
                }
            };
        let residual = &self.try_matmul(&solution)? - b;
        residual.finite()?;
        Ok(LeastSquares {
            residual_norm: residual.norm_frobenius(),
            solution,
            rank,
            method: used,
            tolerance,
        })
    }
}
