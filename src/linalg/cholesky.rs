use super::{normalized, rhs};
use crate::{
    DecompositionOptions, Matrix, Real,
    error::{NumericError, Result},
    numeric::number,
};
/// Cholesky decomposition of a symmetric positive-definite matrix: `A = L*Lᵀ`.
#[derive(Clone, Debug)]
pub struct Cholesky<T> {
    lower: Matrix<T>,
    scale: T,
}
impl<T: Real> Cholesky<T> {
    /// Compute this factorization with explicit numerical options; reject invalid input and exhausted iteration.
    /// Shape requirements: finite symmetric positive-definite N×N input.
    pub fn decompose(a: &Matrix<T>, options: DecompositionOptions<T>) -> Result<Self> {
        let n = a.square()?;
        let tol = options.resolve(n)?;
        a.finite()?;
        if !a.is_symmetric(tol) {
            return Err(NumericError::NotSymmetric.into());
        }
        let (mut l, scale) = normalized(a)?;
        let cutoff = tol.scaled(scale);
        for i in 0..n {
            for j in 0..=i {
                let mut value = l[(i, j)];
                for k in 0..j {
                    value -= l[(i, k)] * l[(j, k)];
                }
                if i == j {
                    if value <= cutoff {
                        return Err(NumericError::NotPositiveDefinite.into());
                    }
                    l[(i, j)] = value.sqrt();
                } else {
                    l[(i, j)] = value / l[(j, j)];
                }
            }
            for j in i + 1..n {
                l[(i, j)] = T::zero();
            }
        }
        l.finite()?;
        Ok(Self { lower: l, scale })
    }
    /// Allocate the lower factor in the original input units/convention.
    /// Shape requirements: any valid stored layout; additional operand restrictions are documented above.
    pub fn lower(&self) -> Matrix<T> {
        self.lower.scalar_mul(self.scale.sqrt())
    }
    /// Solve using the retained factors; finite RHS dimensions must match the original input.
    /// Shape requirements: retained A=N×N and B=N×K; A is numerically nonsingular.
    pub fn solve(&self, b: &Matrix<T>) -> Result<Matrix<T>> {
        let n = self.lower.rows();
        rhs(n, b)?;
        let mut x = b.scalar_div(self.scale);
        for i in 0..n {
            for c in 0..b.cols() {
                let mut value = x[(i, c)];
                for k in 0..i {
                    value -= self.lower[(i, k)] * x[(k, c)];
                }
                x[(i, c)] = value / self.lower[(i, i)];
            }
        }
        for i in (0..n).rev() {
            for c in 0..b.cols() {
                let mut value = x[(i, c)];
                for k in i + 1..n {
                    value -= self.lower[(k, i)] * x[(k, c)];
                }
                x[(i, c)] = value / self.lower[(i, i)];
            }
        }
        x.finite()?;
        Ok(x)
    }
    /// Log determinant of the retained positive-definite Cholesky factors.
    /// Shape requirements: any valid stored layout; additional operand restrictions are documented above.
    pub fn log_determinant(&self) -> T {
        number::<T>(self.lower.rows()) * self.scale.ln()
            + number::<T>(2)
                * (0..self.lower.rows()).fold(T::zero(), |s, i| s + self.lower[(i, i)].ln())
    }
    /// Compute a square determinant; its magnitude can underflow or overflow.
    /// Shape requirements: input is finite N×N.
    pub fn determinant(&self) -> T {
        self.log_determinant().exp()
    }
}
