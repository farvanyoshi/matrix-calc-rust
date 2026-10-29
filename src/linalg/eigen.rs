use super::normalized;
use crate::{
    DecompositionOptions, Matrix, Real,
    error::{NumericError, Result},
    numeric::number,
};
/// Real symmetric eigendecomposition `A*V = V*diag(values)`. Eigenvalues are
/// ascending; columns of V are orthonormal. Cyclic Jacobi uses bounded sweeps.
#[derive(Clone, Debug)]
pub struct SymmetricEigen<T> {
    values: Vec<T>,
    vectors: Matrix<T>,
    sweeps: usize,
}
impl<T: Real> SymmetricEigen<T> {
    /// Compute this factorization with explicit numerical options; reject invalid input and exhausted iteration.
    pub fn decompose(input: &Matrix<T>, options: DecompositionOptions<T>) -> Result<Self> {
        let n = input.square()?;
        let tol = options.resolve(n)?;
        input.finite()?;
        if !input.is_symmetric(tol) {
            return Err(NumericError::NotSymmetric.into());
        }
        let (mut a, scale) = normalized(input)?;
        let mut vectors = Matrix::<T>::try_identity(n)?;
        // Symmetrize within the validated tolerance without adding large values.
        for r in 0..n {
            for c in 0..r {
                let x = a[(r, c)] / number::<T>(2) + a[(c, r)] / number::<T>(2);
                a[(r, c)] = x;
                a[(c, r)] = x;
            }
        }
        let convergence = tol.scaled(scale).max(T::epsilon());
        let mut converged = n < 2;
        let mut sweeps = 0;
        for sweep in 0..options.max_iterations {
            if converged {
                break;
            }
            let mut changed = false;
            for p in 0..n {
                for q in p + 1..n {
                    let x = a[(p, q)];
                    if x.abs() <= convergence {
                        continue;
                    }
                    let delta = (a[(q, q)] - a[(p, p)]) / number::<T>(2);
                    let denom = delta
                        + if delta >= T::zero() {
                            delta.hypot(x)
                        } else {
                            -delta.hypot(x)
                        };
                    let t = if denom == T::zero() {
                        T::one()
                    } else {
                        x / denom
                    };
                    let c = T::one() / T::one().hypot(t);
                    let s = c * t;
                    let app = a[(p, p)];
                    let aqq = a[(q, q)];
                    for k in 0..n {
                        if k == p || k == q {
                            continue;
                        }
                        let kp = a[(k, p)];
                        let kq = a[(k, q)];
                        a[(k, p)] = c * kp - s * kq;
                        a[(p, k)] = a[(k, p)];
                        a[(k, q)] = s * kp + c * kq;
                        a[(q, k)] = a[(k, q)];
                    }
                    a[(p, p)] = app - t * x;
                    a[(q, q)] = aqq + t * x;
                    a[(p, q)] = T::zero();
                    a[(q, p)] = T::zero();
                    for k in 0..n {
                        let kp = vectors[(k, p)];
                        let kq = vectors[(k, q)];
                        vectors[(k, p)] = c * kp - s * kq;
                        vectors[(k, q)] = s * kp + c * kq;
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
        a.finite()?;
        vectors.finite()?;
        let mut order: Vec<_> = (0..n).collect();
        order.sort_by(|&i, &j| a[(i, i)].partial_cmp(&a[(j, j)]).unwrap());
        let values: Vec<_> = order.iter().map(|&i| a[(i, i)] * scale).collect();
        if values.iter().any(|x| !x.is_finite()) {
            return Err(NumericError::NonFinite.into());
        }
        let vectors = Matrix::try_from_fn(n, n, |r, c| vectors[(r, order[c])])?;
        Ok(Self {
            values,
            vectors,
            sweeps,
        })
    }
    /// Ascending real eigenvalues.
    pub fn eigenvalues(&self) -> &[T] {
        &self.values
    }
    /// Orthonormal eigenvectors stored as columns, matching the eigenvalue order.
    pub fn eigenvectors(&self) -> &Matrix<T> {
        &self.vectors
    }
    /// Number of complete pair sweeps used to certify convergence.
    pub fn sweeps(&self) -> usize {
        self.sweeps
    }
}
