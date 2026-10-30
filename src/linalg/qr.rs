use super::{normalized, rhs};
use crate::{
    DecompositionOptions, Matrix, Real,
    error::{NumericError, Result},
    numeric::{number, scaled_ratio},
};
/// Compact Householder QR. With pivoting, `A*P = Q*R`, and `permutation[j]`
/// identifies the original column now occupying position j.
#[derive(Clone, Debug)]
pub struct Qr<T> {
    r: Matrix<T>,
    reflectors: Vec<Vec<T>>,
    permutation: Vec<usize>,
    scale: T,
    cutoff: T,
}
impl<T: Real> Qr<T> {
    /// Compute this factorization with explicit numerical options; reject invalid input and exhausted iteration.
    /// Shape requirements: finite M×N input.
    pub fn decompose(a: &Matrix<T>, pivot: bool, options: DecompositionOptions<T>) -> Result<Self> {
        let tol = options.resolve(a.rows().max(a.cols()))?;
        let (mut r, scale) = normalized(a)?;
        let m = a.rows();
        let n = a.cols();
        let mut permutation: Vec<_> = (0..n).collect();
        let mut reflectors = Vec::new();
        for k in 0..m.min(n) {
            if pivot {
                let mut best = k;
                let mut max = T::zero();
                for c in k..n {
                    let norm = (k..m).fold(T::zero(), |s, i| s.hypot(r[(i, c)]));
                    if norm > max {
                        max = norm;
                        best = c;
                    }
                }
                r.swap_cols(k, best)?;
                permutation.swap(k, best);
            }
            let mut v: Vec<_> = (k..m).map(|i| r[(i, k)]).collect();
            let norm = v.iter().fold(T::zero(), |s, x| s.hypot(*x));
            if norm == T::zero() {
                reflectors.push(Vec::new());
                continue;
            }
            let alpha = if v[0] >= T::zero() { -norm } else { norm };
            v[0] -= alpha;
            let vn = v.iter().fold(T::zero(), |s, x| s.hypot(*x));
            for x in &mut v {
                *x /= vn;
            }
            for c in k..n {
                let dot = number::<T>(2) * (k..m).fold(T::zero(), |s, i| s + v[i - k] * r[(i, c)]);
                for i in k..m {
                    r[(i, c)] -= dot * v[i - k];
                }
            }
            r[(k, k)] = alpha;
            for i in k + 1..m {
                r[(i, k)] = T::zero();
            }
            reflectors.push(v);
        }
        r.finite()?;
        if r.iter().any(|x| !(*x * scale).is_finite()) {
            return Err(NumericError::NonFinite.into());
        }
        Ok(Self {
            r,
            reflectors,
            permutation,
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
        Matrix::from_fn(self.r.cols(), self.r.cols(), |r, c| {
            if self.permutation[c] == r {
                T::one()
            } else {
                T::zero()
            }
        })
    }
    /// Count R diagonal entries above the threshold; use pivoted QR for a rank-revealing estimate.
    /// Shape requirements: any finite M×N input or its retained factorization.
    pub fn rank(&self) -> usize {
        (0..self.r.rows().min(self.r.cols()))
            .filter(|&i| self.r[(i, i)].abs() > self.cutoff)
            .count()
    }
    /// Economy R has min(m,n) rows; use `r_full` for the zero-padded form.
    /// Shape requirements: any valid stored layout; additional operand restrictions are documented above.
    pub fn r(&self) -> Matrix<T> {
        Matrix::from_fn(self.r.rows().min(self.r.cols()), self.r.cols(), |r, c| {
            self.r[(r, c)] * self.scale
        })
    }
    /// Allocate the full zero-padded R in the original input units.
    /// Shape requirements: any valid stored layout; additional operand restrictions are documented above.
    pub fn r_full(&self) -> Matrix<T> {
        self.r.scalar_mul(self.scale)
    }
    /// Allocate economy Q by applying the stored reflectors to a rectangular identity.
    /// Shape requirements: any valid stored layout; additional operand restrictions are documented above.
    pub fn q(&self) -> Matrix<T> {
        self.apply_q(&Matrix::from_fn(
            self.r.rows(),
            self.r.rows().min(self.r.cols()),
            |r, c| if r == c { T::one() } else { T::zero() },
        ))
        .expect("valid identity")
    }
    /// Allocate full square Q by applying the stored reflectors to identity.
    /// Shape requirements: any valid stored layout; additional operand restrictions are documented above.
    pub fn q_full(&self) -> Matrix<T> {
        self.apply_q(&Matrix::identity(self.r.rows()))
            .expect("valid identity")
    }
    /// Apply Q to finite RHS columns without materializing Q; row counts must match.
    /// Shape requirements: Q is M×M and B has M rows.
    pub fn apply_q(&self, b: &Matrix<T>) -> Result<Matrix<T>> {
        self.apply(b, false)
    }
    /// Apply Qᵀ to finite RHS columns without materializing Q; row counts must match.
    /// Shape requirements: Q is M×M and B has M rows.
    pub fn apply_q_transpose(&self, b: &Matrix<T>) -> Result<Matrix<T>> {
        self.apply(b, true)
    }
    fn apply(&self, b: &Matrix<T>, transpose: bool) -> Result<Matrix<T>> {
        rhs(self.r.rows(), b)?;
        let (mut out, scale) = normalized(b)?;
        let count = self.reflectors.len();
        for step in 0..count {
            let k = if transpose { step } else { count - 1 - step };
            let v = &self.reflectors[k];
            if v.is_empty() {
                continue;
            }
            for c in 0..out.cols() {
                let dot = number::<T>(2)
                    * v.iter()
                        .enumerate()
                        .fold(T::zero(), |s, (i, x)| s + *x * out[(k + i, c)]);
                for (i, x) in v.iter().enumerate() {
                    out[(k + i, c)] -= dot * (*x);
                }
            }
        }
        let out = out.scalar_mul(scale);
        out.finite()?;
        Ok(out)
    }
    /// Full-column-rank least squares. Wide or deficient systems require SVD.
    /// Shape requirements: retained A=M×N with M≥N and full column rank; B=M×K.
    pub fn solve(&self, b: &Matrix<T>) -> Result<Matrix<T>> {
        rhs(self.r.rows(), b)?;
        let n = self.r.cols();
        if self.r.rows() < n || self.rank() < n {
            return Err(NumericError::RankDeficient.into());
        }
        let (scaled_b, bscale) = normalized(b)?;
        let y = self.apply_q_transpose(&scaled_b)?;
        let mut xp = Matrix::<T>::try_from_fn(n, b.cols(), |r, c| y[(r, c)])?;
        for i in (0..n).rev() {
            for c in 0..b.cols() {
                let mut value = xp[(i, c)];
                for j in i + 1..n {
                    value -= self.r[(i, j)] * xp[(j, c)];
                }
                xp[(i, c)] = value / self.r[(i, i)];
            }
        }
        let out = Matrix::try_from_fn(n, b.cols(), |r, c| {
            let i = self.permutation.iter().position(|&x| x == r).unwrap();
            scaled_ratio(xp[(i, c)], bscale, self.scale, T::one())
        })?;
        out.finite()?;
        Ok(out)
    }
}
