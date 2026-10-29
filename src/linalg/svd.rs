use super::{normalized, rhs};
use crate::{
    DecompositionOptions, Matrix, Real,
    error::{NumericError, Result},
    numeric::{compensated, number, scaled_ratio},
};
/// Economy SVD: `A = U*diag(s)*Vᵀ`, singular values descending and nonnegative.
/// Uses scaled one-sided Jacobi rotations directly on A, never AᵀA. Wide inputs
/// are handled through transposition. Iteration limits count full pair sweeps.
#[derive(Clone, Debug)]
pub struct Svd<T> {
    u: Matrix<T>,
    v_t: Matrix<T>,
    values: Vec<T>,
    scaled_values: Vec<T>,
    scale: T,
    cutoff: T,
    sweeps: usize,
}
impl<T: Real> Svd<T> {
    /// Compute this factorization with explicit numerical options; reject invalid input and exhausted iteration.
    pub fn decompose(a: &Matrix<T>, options: DecompositionOptions<T>) -> Result<Self> {
        let tol = options.resolve(a.rows().max(a.cols()))?;
        if a.rows() < a.cols() {
            let transposed = Self::decompose(&a.transpose(), options)?;
            return Ok(Self {
                u: transposed.v_t.transpose(),
                v_t: transposed.u.transpose(),
                ..transposed
            });
        }
        let (mut b, scale) = normalized(a)?;
        let m = b.rows();
        let n = b.cols();
        let mut v = Matrix::try_identity(n)?;
        let convergence = T::epsilon() * number::<T>(m.max(1)) * number::<T>(4);
        let mut sweeps = 0;
        let mut converged = n < 2;
        for sweep in 0..options.max_iterations {
            if converged {
                break;
            }
            let mut changed = false;
            for p in 0..n {
                for q in p + 1..n {
                    let alpha = (0..m).fold(T::zero(), |s, i| s.hypot(b[(i, p)]));
                    let beta = (0..m).fold(T::zero(), |s, i| s.hypot(b[(i, q)]));
                    if alpha == T::zero() || beta == T::zero() {
                        continue;
                    }
                    let corr =
                        compensated((0..m).map(|i| (b[(i, p)] / alpha) * (b[(i, q)] / beta)));
                    if corr.abs() <= convergence {
                        continue;
                    }
                    let max = alpha.max(beta);
                    let aa = alpha / max;
                    let bb = beta / max;
                    let delta = (bb - aa) * (bb + aa);
                    let cross = number::<T>(2) * corr * aa * bb;
                    let denom = delta
                        + if delta >= T::zero() {
                            delta.hypot(cross)
                        } else {
                            -delta.hypot(cross)
                        };
                    if denom == T::zero() {
                        continue;
                    }
                    let t = cross / denom;
                    if t == T::zero() {
                        continue;
                    }
                    let c = T::one() / T::one().hypot(t);
                    let s = c * t;
                    for i in 0..m {
                        let x = b[(i, p)];
                        let y = b[(i, q)];
                        b[(i, p)] = c * x - s * y;
                        b[(i, q)] = s * x + c * y;
                    }
                    for i in 0..n {
                        let x = v[(i, p)];
                        let y = v[(i, q)];
                        v[(i, p)] = c * x - s * y;
                        v[(i, q)] = s * x + c * y;
                    }
                    changed = true;
                }
            }
            sweeps = sweep + 1;
            converged = !changed;
        }
        if !converged {
            return Err(NumericError::NonConvergence {
                iterations: options.max_iterations,
            }
            .into());
        }
        b.finite()?;
        v.finite()?;
        let norms: Vec<_> = (0..n)
            .map(|c| (0..m).fold(T::zero(), |s, r| s.hypot(b[(r, c)])))
            .collect();
        let mut order: Vec<_> = (0..n).collect();
        order.sort_by(|&a, &b| norms[b].partial_cmp(&norms[a]).unwrap());
        let scaled_values: Vec<_> = order.iter().map(|&i| norms[i]).collect();
        let values: Vec<_> = scaled_values.iter().map(|x| *x * scale).collect();
        if values.iter().any(|x| !x.is_finite()) {
            return Err(NumericError::NonFinite.into());
        }
        let mut u = Matrix::try_zeros(m, n)?;
        for c in 0..n {
            let norm = scaled_values[c];
            if norm != T::zero() {
                for r in 0..m {
                    u[(r, c)] = b[(r, order[c])] / norm;
                }
            } else {
                // Complete an orthonormal basis for exact zero singular values.
                let mut best = Vec::new();
                let mut bestnorm = T::zero();
                for basis in 0..m {
                    let mut candidate = vec![T::zero(); m];
                    candidate[basis] = T::one();
                    for _ in 0..2 {
                        for j in 0..c {
                            let dot = compensated((0..m).map(|r| candidate[r] * u[(r, j)]));
                            for r in 0..m {
                                candidate[r] -= dot * u[(r, j)];
                            }
                        }
                    }
                    let cn = candidate.iter().fold(T::zero(), |s, x| s.hypot(*x));
                    if cn > bestnorm {
                        bestnorm = cn;
                        best = candidate;
                    }
                }
                if bestnorm == T::zero() {
                    return Err(NumericError::NonConvergence { iterations: sweeps }.into());
                }
                for r in 0..m {
                    u[(r, c)] = best[r] / bestnorm;
                }
            }
        }
        let v_t = Matrix::try_from_fn(n, n, |r, c| v[(c, order[r])])?;
        // Absolute tolerance is in original input units, not sigma_max units.
        let cutoff = tol.absolute() / scale
            + tol.relative() * scaled_values.first().copied().unwrap_or(T::zero());
        Ok(Self {
            u,
            v_t,
            values,
            scaled_values,
            scale,
            cutoff,
            sweeps,
        })
    }
    /// Economy left singular vectors as orthonormal columns.
    pub fn u(&self) -> &Matrix<T> {
        &self.u
    }
    /// Transposed economy right singular vectors as orthonormal rows.
    pub fn v_t(&self) -> &Matrix<T> {
        &self.v_t
    }
    /// Descending nonnegative singular values in original input units.
    pub fn singular_values(&self) -> &[T] {
        &self.values
    }
    /// Number of complete pair sweeps used to certify convergence.
    pub fn sweeps(&self) -> usize {
        self.sweeps
    }
    /// Numerical rank from singular values above the selected threshold.
    pub fn rank(&self) -> usize {
        self.scaled_values
            .iter()
            .filter(|&&x| x > self.cutoff)
            .count()
    }
    /// Largest singular value, or zero for an empty spectrum.
    pub fn spectral_norm(&self) -> T {
        self.values.first().copied().unwrap_or(T::zero())
    }
    /// Empty matrices have condition number 1; numerically singular matrices infinity.
    pub fn condition_number(&self) -> T {
        if self.values.is_empty() {
            T::one()
        } else if self.rank() < self.values.len() {
            T::infinity()
        } else {
            self.scaled_values[0] / self.scaled_values[self.values.len() - 1]
        }
    }
    /// Compute a tolerance-truncated Moore–Penrose pseudoinverse using SVD.
    pub fn pseudoinverse(&self) -> Result<Matrix<T>> {
        let out = Matrix::try_from_fn(self.v_t.cols(), self.u.rows(), |r, c| {
            compensated(
                (0..self.values.len())
                    .filter(|&k| self.scaled_values[k] > self.cutoff)
                    .map(|k| {
                        scaled_ratio(
                            self.v_t[(k, r)] * self.u[(c, k)],
                            T::one(),
                            self.scale,
                            self.scaled_values[k],
                        )
                    }),
            )
        })?;
        out.finite()?;
        Ok(out)
    }
    /// Minimum-norm least squares using only singular values above the threshold.
    pub fn solve(&self, b: &Matrix<T>) -> Result<Matrix<T>> {
        rhs(self.u.rows(), b)?;
        let bscale = if b.max_abs() == T::zero() {
            T::one()
        } else {
            b.max_abs()
        };
        let projected = Matrix::try_from_fn(self.values.len(), b.cols(), |k, c| {
            if self.scaled_values[k] <= self.cutoff {
                return T::zero();
            }
            let dot =
                compensated((0..self.u.rows()).map(|r| self.u[(r, k)] * (b[(r, c)] / bscale)));
            scaled_ratio(dot, bscale, self.scale, self.scaled_values[k])
        })?;
        let out = self.v_t.transpose().try_matmul(&projected)?;
        out.finite()?;
        Ok(out)
    }
}
