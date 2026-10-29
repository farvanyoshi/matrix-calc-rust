use super::{normalized, rhs};
use crate::{
    DecompositionOptions, Matrix, Real,
    error::{NumericError, Result},
    numeric::number,
};
/// Compact Householder QR. With pivoting, `A*P = Q*R`, and permutation[j]
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
        Ok(Self {
            r,
            reflectors,
            permutation,
            scale,
            cutoff: tol.scaled(scale),
        })
    }
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }
    pub fn permutation_matrix(&self) -> Matrix<T> {
        Matrix::from_fn(self.r.cols(), self.r.cols(), |r, c| {
            if self.permutation[c] == r {
                T::one()
            } else {
                T::zero()
            }
        })
    }
    pub fn rank(&self) -> usize {
        (0..self.r.rows().min(self.r.cols()))
            .filter(|&i| self.r[(i, i)].abs() > self.cutoff)
            .count()
    }
    /// Economy R has min(m,n) rows; use `r_full` for the zero-padded form.
    pub fn r(&self) -> Matrix<T> {
        Matrix::from_fn(self.r.rows().min(self.r.cols()), self.r.cols(), |r, c| {
            self.r[(r, c)] * self.scale
        })
    }
    pub fn r_full(&self) -> Matrix<T> {
        self.r.scalar_mul(self.scale)
    }
    pub fn q(&self) -> Matrix<T> {
        self.apply_q(&Matrix::from_fn(
            self.r.rows(),
            self.r.rows().min(self.r.cols()),
            |r, c| if r == c { T::one() } else { T::zero() },
        ))
        .expect("valid identity")
    }
    pub fn q_full(&self) -> Matrix<T> {
        self.apply_q(&Matrix::identity(self.r.rows()))
            .expect("valid identity")
    }
    pub fn apply_q(&self, b: &Matrix<T>) -> Result<Matrix<T>> {
        self.apply(b, false)
    }
    pub fn apply_q_transpose(&self, b: &Matrix<T>) -> Result<Matrix<T>> {
        self.apply(b, true)
    }
    fn apply(&self, b: &Matrix<T>, transpose: bool) -> Result<Matrix<T>> {
        rhs(self.r.rows(), b)?;
        let mut out = b.clone();
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
        out.finite()?;
        Ok(out)
    }
    /// Full-column-rank least squares. Wide or deficient systems require SVD.
    pub fn solve(&self, b: &Matrix<T>) -> Result<Matrix<T>> {
        rhs(self.r.rows(), b)?;
        let n = self.r.cols();
        if self.r.rows() < n || self.rank() < n {
            return Err(NumericError::RankDeficient.into());
        }
        let y = self.apply_q_transpose(b)?;
        let mut xp = Matrix::<T>::try_from_fn(n, b.cols(), |r, c| y[(r, c)] / self.scale)?;
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
            xp[(i, c)]
        })?;
        out.finite()?;
        Ok(out)
    }
}
