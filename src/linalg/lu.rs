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
    /// Compute this factorization with explicit numerical options; reject invalid input and exhausted iteration.
    /// Shape requirements: finite N×N input.
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
        if (0..n).any(|r| (r..n).any(|c| !(packed[(r, c)] * scale).is_finite())) {
            return Err(NumericError::NonFinite.into());
        }
        Ok(Self {
            packed,
            permutation,
            odd,
            scale,
            cutoff: tol.scaled(scale),
        })
    }
    /// Original row/column indices in the order described by this factorization.
    /// Shape requirements: any valid stored layout; additional operand restrictions are documented above.
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }
    /// Allocate the permutation matrix matching this factorization's documented convention.
    /// Shape requirements: any valid stored layout; additional operand restrictions are documented above.
    pub fn permutation_matrix(&self) -> Matrix<T> {
        Matrix::from_fn(self.packed.rows(), self.packed.rows(), |r, c| {
            if self.permutation[r] == c {
                T::one()
            } else {
                T::zero()
            }
        })
    }
    /// Allocate the lower factor in the original input units/convention.
    /// Shape requirements: any valid stored layout; additional operand restrictions are documented above.
    pub fn lower(&self) -> Matrix<T> {
        let n = self.packed.rows();
        Matrix::from_fn(n, n, |r, c| match r.cmp(&c) {
            std::cmp::Ordering::Equal => T::one(),
            std::cmp::Ordering::Greater => self.packed[(r, c)],
            std::cmp::Ordering::Less => T::zero(),
        })
    }
    /// Allocate the upper factor in original input units.
    /// Shape requirements: any valid stored layout; additional operand restrictions are documented above.
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
    /// Whether any normalized LU pivot is at or below the selected threshold.
    /// Shape requirements: any valid stored layout; additional operand restrictions are documented above.
    pub fn is_singular(&self) -> bool {
        (0..self.packed.rows()).any(|i| self.packed[(i, i)].abs() <= self.cutoff)
    }
    /// Solve using the retained factors; finite RHS dimensions must match the original input.
    /// Shape requirements: retained A=N×N and B=N×K; A is numerically nonsingular.
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
    /// Solve against identity to invert finite nonsingular square input.
    /// Shape requirements: input is finite invertible N×N.
    pub fn inverse(&self) -> Result<Matrix<T>> {
        self.solve(&Matrix::try_identity(self.packed.rows())?)
    }
    /// Return determinant sign and log absolute magnitude; exact zero pivots give `(0, -infinity)`.
    /// Shape requirements: input is finite N×N.
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
    /// Shape requirements: input is finite N×N.
    pub fn determinant(&self) -> T {
        let (sign, log) = self.sign_log_determinant();
        if sign == T::zero() {
            T::zero()
        } else {
            sign * log.exp()
        }
    }
}
impl<T: Real> Lu<T> {
    /// Solve Aᵀ X = B using the retained factors, without refactorizing.
    /// Shape requirements: retained A=N×N and B=N×K.
    pub fn solve_transpose(&self, b: &Matrix<T>) -> Result<Matrix<T>> {
        let n = self.packed.rows();
        rhs(n, b)?;
        if self.is_singular() {
            return Err(NumericError::Singular.into());
        }
        let mut z = b.scalar_div(self.scale);
        for i in 0..n {
            for c in 0..b.cols() {
                let mut value = z[(i, c)];
                for k in 0..i {
                    value -= self.packed[(k, i)] * z[(k, c)];
                }
                z[(i, c)] = value / self.packed[(i, i)];
            }
        }
        for i in (0..n).rev() {
            for c in 0..b.cols() {
                let mut value = z[(i, c)];
                for k in i + 1..n {
                    value -= self.packed[(k, i)] * z[(k, c)];
                }
                z[(i, c)] = value;
            }
        }
        let mut out = Matrix::try_zeros(n, b.cols())?;
        for i in 0..n {
            for c in 0..b.cols() {
                out[(self.permutation[i], c)] = z[(i, c)];
            }
        }
        out.finite()?;
        Ok(out)
    }
}
