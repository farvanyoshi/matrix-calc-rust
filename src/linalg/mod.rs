//! Reusable real-valued decompositions. Inputs must be finite; no normal equations
//! or inverse-based solves are used. Factor conventions are documented on types.
mod cholesky;
mod lu;
mod qr;
use crate::{
    DecompositionOptions, Matrix, Real,
    error::{NumericError, Result},
};
pub use cholesky::Cholesky;
pub use lu::Lu;
pub use qr::Qr;

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
    pub fn lu(&self) -> Result<Lu<T>> {
        Lu::decompose(self, DecompositionOptions::default())
    }
    pub fn lu_with_options(&self, options: DecompositionOptions<T>) -> Result<Lu<T>> {
        Lu::decompose(self, options)
    }
    pub fn solve(&self, b: &Self) -> Result<Self> {
        self.lu()?.solve(b)
    }
    pub fn determinant(&self) -> Result<T> {
        Ok(self.lu()?.determinant())
    }
    /// Returns (sign, log(abs(det))). Exact singularity gives (0, -infinity).
    pub fn sign_log_determinant(&self) -> Result<(T, T)> {
        Ok(self.lu()?.sign_log_determinant())
    }
    pub fn inverse(&self) -> Result<Self> {
        self.lu()?.inverse()
    }
    pub fn cholesky(&self) -> Result<Cholesky<T>> {
        Cholesky::decompose(self, DecompositionOptions::default())
    }
    pub fn cholesky_with_options(&self, options: DecompositionOptions<T>) -> Result<Cholesky<T>> {
        Cholesky::decompose(self, options)
    }
    pub fn qr(&self) -> Result<Qr<T>> {
        Qr::decompose(self, false, DecompositionOptions::default())
    }
    pub fn pivoted_qr(&self) -> Result<Qr<T>> {
        Qr::decompose(self, true, DecompositionOptions::default())
    }
    pub fn qr_with_options(&self, pivot: bool, options: DecompositionOptions<T>) -> Result<Qr<T>> {
        Qr::decompose(self, pivot, options)
    }
    pub fn powi(&self, power: i32) -> Result<Self> {
        if power < 0 {
            self.inverse()?.pow(power.unsigned_abs())
        } else {
            self.pow(power as u32)
        }
    }
    pub fn solve_lower(&self, b: &Self, unit_diagonal: bool) -> Result<Self> {
        triangular(
            self,
            b,
            true,
            unit_diagonal,
            DecompositionOptions::default(),
        )
    }
    pub fn solve_upper(&self, b: &Self, unit_diagonal: bool) -> Result<Self> {
        triangular(
            self,
            b,
            false,
            unit_diagonal,
            DecompositionOptions::default(),
        )
    }
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
