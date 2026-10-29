use super::{normalized, rhs};
use crate::{
    DecompositionOptions, Matrix, Real,
    error::{NumericError, Result},
    numeric::number,
};
/// Partial-row-pivoted LU: `P*A = L*U`. Factors are scaled internally.
#[derive(Clone, Debug)]
pub struct Lu<T> {
    packed: Matrix<T>,
    permutation: Vec<usize>,
    odd: bool,
    scale: T,
    cutoff: T,
}
impl<T: Real> Lu<T> {
    pub fn decompose(a: &Matrix<T>, options: DecompositionOptions<T>) -> Result<Self> {
        let n = a.square()?;
        let tol = options.resolve(n)?;
        let (mut packed, scale) = normalized(a)?;
        let mut permutation: Vec<_> = (0..n).collect();
        let mut odd = false;
        for k in 0..n {
            let mut pivot = k;
            for i in k + 1..n {
                if packed[(i, k)].abs() > packed[(pivot, k)].abs() {
                    pivot = i;
                }
            }
            if pivot != k {
                packed.swap_rows(k, pivot)?;
                permutation.swap(k, pivot);
                odd = !odd;
            }
            let d = packed[(k, k)];
            if d == T::zero() {
                continue;
            }
            for i in k + 1..n {
                packed[(i, k)] /= d;
                for j in k + 1..n {
                    let term = packed[(i, k)] * packed[(k, j)];
                    packed[(i, j)] -= term;
                }
            }
        }
        packed.finite()?;
        Ok(Self {
            packed,
            permutation,
            odd,
            scale,
            cutoff: tol.scaled(scale),
        })
    }
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }
    pub fn permutation_matrix(&self) -> Matrix<T> {
        Matrix::from_fn(self.packed.rows(), self.packed.rows(), |r, c| {
            if self.permutation[r] == c {
                T::one()
            } else {
                T::zero()
            }
        })
    }
    pub fn lower(&self) -> Matrix<T> {
        let n = self.packed.rows();
        Matrix::from_fn(n, n, |r, c| {
            if r == c {
                T::one()
            } else if r > c {
                self.packed[(r, c)]
            } else {
                T::zero()
            }
        })
    }
    pub fn upper(&self) -> Matrix<T> {
        let n = self.packed.rows();
        Matrix::from_fn(n, n, |r, c| {
            if r <= c {
                self.packed[(r, c)] * self.scale
            } else {
                T::zero()
            }
        })
    }
    pub fn is_singular(&self) -> bool {
        (0..self.packed.rows()).any(|i| self.packed[(i, i)].abs() <= self.cutoff)
    }
    pub fn solve(&self, b: &Matrix<T>) -> Result<Matrix<T>> {
        let n = self.packed.rows();
        rhs(n, b)?;
        if self.is_singular() {
            return Err(NumericError::Singular.into());
        }
        let mut x =
            Matrix::try_from_fn(n, b.cols(), |r, c| b[(self.permutation[r], c)] / self.scale)?;
        for i in 0..n {
            for c in 0..b.cols() {
                let mut value = x[(i, c)];
                for k in 0..i {
                    value -= self.packed[(i, k)] * x[(k, c)];
                }
                x[(i, c)] = value;
            }
        }
        for i in (0..n).rev() {
            for c in 0..b.cols() {
                let mut value = x[(i, c)];
                for k in i + 1..n {
                    value -= self.packed[(i, k)] * x[(k, c)];
                }
                x[(i, c)] = value / self.packed[(i, i)];
            }
        }
        x.finite()?;
        Ok(x)
    }
    pub fn inverse(&self) -> Result<Matrix<T>> {
        self.solve(&Matrix::try_identity(self.packed.rows())?)
    }
    pub fn sign_log_determinant(&self) -> (T, T) {
        let mut sign = if self.odd { -T::one() } else { T::one() };
        let mut log = number::<T>(self.packed.rows()) * self.scale.ln();
        for i in 0..self.packed.rows() {
            let x = self.packed[(i, i)];
            if x == T::zero() {
                return (T::zero(), T::neg_infinity());
            }
            if x < T::zero() {
                sign = -sign;
            }
            log += x.abs().ln();
        }
        (sign, log)
    }
    /// Uses exact zero pivots for determinant; solve singularity is tolerance-based.
    pub fn determinant(&self) -> T {
        let (sign, log) = self.sign_log_determinant();
        if sign == T::zero() {
            T::zero()
        } else {
            sign * log.exp()
        }
    }
}
