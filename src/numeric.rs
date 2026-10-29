//! Explicit floating-point numerics and checked primitive integer operations.
use crate::{
    Matrix,
    error::{NumericError, Result},
};
use num_traits::{CheckedNeg, Float, FromPrimitive, PrimInt};
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}
/// Real scalar types supported by the numerical algorithms (`f32` and `f64`).
pub trait Real:
    sealed::Sealed
    + Float
    + FromPrimitive
    + Debug
    + Display
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
}
impl Real for f32 {}
impl Real for f64 {}
pub(crate) fn number<T: Real>(n: usize) -> T {
    T::from_usize(n).expect("dimension representable by real scalar")
}
/// Explicit, nonnegative finite absolute and relative comparison tolerances.
#[derive(Clone, Copy, Debug)]
pub struct Tolerance<T> {
    atol: T,
    rtol: T,
}
impl<T: Real> Tolerance<T> {
    pub fn new(atol: T, rtol: T) -> Result<Self> {
        if !atol.is_finite() || !rtol.is_finite() || atol < T::zero() || rtol < T::zero() {
            return Err(NumericError::InvalidOptions.into());
        }
        Ok(Self { atol, rtol })
    }
    pub fn absolute(&self) -> T {
        self.atol
    }
    pub fn relative(&self) -> T {
        self.rtol
    }
    pub fn for_dimension(dimension: usize) -> Self {
        Self {
            atol: T::zero(),
            rtol: T::epsilon() * number::<T>(dimension.max(1)) * number::<T>(8),
        }
    }
    pub fn threshold(&self, scale: T) -> T {
        self.atol + self.rtol * scale.abs()
    }
    pub(crate) fn scaled(&self, scale: T) -> T {
        self.atol / scale + self.rtol
    }
}
/// Decomposition settings. Iterative methods count sweeps, not scalar rotations.
#[derive(Clone, Copy, Debug)]
pub struct DecompositionOptions<T> {
    pub tolerance: Option<Tolerance<T>>,
    pub max_iterations: usize,
}
impl<T: Real> Default for DecompositionOptions<T> {
    fn default() -> Self {
        Self {
            tolerance: None,
            max_iterations: 100,
        }
    }
}
impl<T: Real> DecompositionOptions<T> {
    pub(crate) fn resolve(&self, dim: usize) -> Result<Tolerance<T>> {
        if self.max_iterations == 0 {
            return Err(NumericError::InvalidOptions.into());
        }
        Ok(self
            .tolerance
            .unwrap_or_else(|| Tolerance::for_dimension(dim)))
    }
}
/// Neumaier accumulation for already-scaled finite terms.
pub(crate) fn compensated<T: Real>(values: impl IntoIterator<Item = T>) -> T {
    let (mut sum, mut correction) = (T::zero(), T::zero());
    for x in values {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            correction += (sum - t) + x;
        } else {
            correction += (x - t) + sum;
        }
        sum = t;
    }
    sum + correction
}
pub(crate) fn rescale_product<T: Real>(x: T, a: T, b: T) -> T {
    if x.abs() < T::one() {
        (x * a.max(b)) * a.min(b)
    } else {
        (x * a.min(b)) * a.max(b)
    }
}
impl<T: Real> Matrix<T> {
    pub fn all_finite(&self) -> bool {
        self.iter().all(|x| x.is_finite())
    }
    pub(crate) fn finite(&self) -> Result<()> {
        if !self.all_finite() {
            Err(NumericError::NonFinite.into())
        } else {
            Ok(())
        }
    }
    pub fn max_abs(&self) -> T {
        self.iter().fold(T::zero(), |m, x| m.max(x.abs()))
    }
    pub fn approx_eq(&self, other: &Self, tolerance: Tolerance<T>) -> bool {
        self.shape() == other.shape()
            && self.iter().zip(other).all(|(a, b)| {
                a.is_finite()
                    && b.is_finite()
                    && (*a - *b).abs() <= tolerance.threshold(a.abs().max(b.abs()))
            })
    }
    /// Scaled compensated sum. Nonfinite inputs retain IEEE sum semantics.
    pub fn sum_compensated(&self) -> T {
        if !self.all_finite() {
            return self.iter().copied().fold(T::zero(), |a, b| a + b);
        }
        let scale = self.max_abs();
        if scale == T::zero() {
            return T::zero();
        }
        compensated(self.iter().map(|x| *x / scale)) * scale
    }
    pub fn mean(&self) -> Result<T> {
        self.finite()?;
        if self.is_empty() {
            return Err(NumericError::Empty.into());
        }
        let scale = self.max_abs();
        if scale == T::zero() {
            return Ok(T::zero());
        }
        let n = number::<T>(self.len());
        Ok(compensated(self.iter().map(|x| (*x / scale) / n)) * scale)
    }
    pub fn row_means(&self) -> Result<Self> {
        if self.cols() == 0 && self.rows() > 0 {
            return Err(NumericError::Empty.into());
        }
        self.finite()?;
        Self::try_from_fn(self.rows(), 1, |r, _| {
            Self::new(self.row(r).unwrap().iter().copied(), 1, self.cols())
                .mean()
                .unwrap()
        })
    }
    pub fn col_means(&self) -> Result<Self> {
        if self.rows() == 0 && self.cols() > 0 {
            return Err(NumericError::Empty.into());
        }
        self.finite()?;
        Self::try_from_fn(1, self.cols(), |_, c| {
            Self::new(self.iter_col(c).copied(), self.rows(), 1)
                .mean()
                .unwrap()
        })
    }
    /// Scaled Euclidean norm over all elements, independent of matrix shape.
    pub fn norm_frobenius(&self) -> T {
        self.iter().fold(T::zero(), |s, x| s.hypot(*x))
    }
    pub fn vector_norm_2(&self) -> Result<T> {
        self.vector_len()?;
        self.finite()?;
        Ok(self.norm_frobenius())
    }
    pub fn vector_norm_1(&self) -> Result<T> {
        self.vector_len()?;
        self.finite()?;
        Ok(self.map(|x| x.abs()).sum_compensated())
    }
    pub fn vector_norm_inf(&self) -> Result<T> {
        self.vector_len()?;
        self.finite()?;
        Ok(self.max_abs())
    }
    pub fn norm_1(&self) -> Result<T> {
        self.finite()?;
        let mut max = T::zero();
        for c in 0..self.cols() {
            let v = Self::new(self.iter_col(c).map(|x| x.abs()), self.rows(), 1);
            max = max.max(v.sum_compensated());
        }
        Ok(max)
    }
    pub fn norm_inf(&self) -> Result<T> {
        self.finite()?;
        let mut max = T::zero();
        for row in self.iter_rows() {
            let v = Self::new(row.iter().map(|x| x.abs()), 1, row.len());
            max = max.max(v.sum_compensated());
        }
        Ok(max)
    }
    pub fn normalized(&self) -> Result<Self> {
        self.vector_len()?;
        self.finite()?;
        let scale = self.max_abs();
        if scale == T::zero() {
            return Err(NumericError::DivideByZero.into());
        }
        let scaled = self.scalar_div(scale);
        let norm = scaled.norm_frobenius();
        Ok(scaled.scalar_div(norm))
    }
    pub fn dot_compensated(&self, other: &Self) -> Result<T> {
        if self.vector_len()? != other.vector_len()? {
            return Err(self.dimension_error(other, "dot"));
        }
        self.finite()?;
        other.finite()?;
        let a = self.max_abs();
        let b = other.max_abs();
        if a == T::zero() || b == T::zero() {
            return Ok(T::zero());
        }
        Ok(rescale_product(
            compensated(self.iter().zip(other).map(|(x, y)| (*x / a) * (*y / b))),
            a,
            b,
        ))
    }
    pub fn abs_elements(&self) -> Self {
        self.map(|x| x.abs())
    }
    pub fn sqrt_elements(&self) -> Self {
        self.map(|x| x.sqrt())
    }
    pub fn exp_elements(&self) -> Self {
        self.map(|x| x.exp())
    }
    pub fn ln_elements(&self) -> Self {
        self.map(|x| x.ln())
    }
    pub fn powf_elements(&self, p: T) -> Self {
        self.map(|x| x.powf(p))
    }
    pub fn is_symmetric(&self, tol: Tolerance<T>) -> bool {
        self.is_square()
            && self.all_finite()
            && (0..self.rows()).all(|r| {
                (0..r).all(|c| {
                    (self[(r, c)] - self[(c, r)]).abs()
                        <= tol.threshold(self[(r, c)].abs().max(self[(c, r)].abs()))
                })
            })
    }
}
impl<T: PrimInt + CheckedNeg> Matrix<T> {
    fn checked_zip(&self, rhs: &Self, f: impl Fn(&T, &T) -> Option<T>) -> Result<Self> {
        self.same_shape(rhs, "checked arithmetic")?;
        let data = self
            .iter()
            .zip(rhs)
            .map(|(a, b)| f(a, b).ok_or(NumericError::Overflow))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Self::try_from_vec(data, self.rows(), self.cols())
    }
    pub fn checked_add(&self, rhs: &Self) -> Result<Self> {
        self.checked_zip(rhs, num_traits::CheckedAdd::checked_add)
    }
    pub fn checked_sub(&self, rhs: &Self) -> Result<Self> {
        self.checked_zip(rhs, num_traits::CheckedSub::checked_sub)
    }
    pub fn checked_hadamard(&self, rhs: &Self) -> Result<Self> {
        self.checked_zip(rhs, num_traits::CheckedMul::checked_mul)
    }
    pub fn checked_element_div(&self, rhs: &Self) -> Result<Self> {
        self.same_shape(rhs, "checked division")?;
        if rhs.iter().any(|x| x.is_zero()) {
            return Err(NumericError::DivideByZero.into());
        }
        self.checked_zip(rhs, num_traits::CheckedDiv::checked_div)
    }
    pub fn checked_negate(&self) -> Result<Self> {
        self.try_map(|x| x.checked_neg().ok_or(NumericError::Overflow.into()))
    }
    pub fn checked_scalar_add(&self, x: T) -> Result<Self> {
        self.try_map(|a| a.checked_add(&x).ok_or(NumericError::Overflow.into()))
    }
    pub fn checked_scalar_sub(&self, x: T) -> Result<Self> {
        self.try_map(|a| a.checked_sub(&x).ok_or(NumericError::Overflow.into()))
    }
    pub fn checked_scalar_mul(&self, x: T) -> Result<Self> {
        self.try_map(|a| a.checked_mul(&x).ok_or(NumericError::Overflow.into()))
    }
    pub fn checked_scalar_div(&self, x: T) -> Result<Self> {
        if x.is_zero() {
            return Err(NumericError::DivideByZero.into());
        }
        self.try_map(|a| a.checked_div(&x).ok_or(NumericError::Overflow.into()))
    }
    pub fn checked_matmul(&self, rhs: &Self) -> Result<Self> {
        if self.cols() != rhs.rows() {
            return Err(self.dimension_error(rhs, "checked multiply"));
        }
        let mut out = Self::try_zeros(self.rows(), rhs.cols())?;
        for r in 0..self.rows() {
            for k in 0..self.cols() {
                for c in 0..rhs.cols() {
                    let term = self[(r, k)]
                        .checked_mul(&rhs[(k, c)])
                        .ok_or(NumericError::Overflow)?;
                    out[(r, c)] = out[(r, c)]
                        .checked_add(&term)
                        .ok_or(NumericError::Overflow)?;
                }
            }
        }
        Ok(out)
    }
    pub fn checked_add_assign(&mut self, rhs: &Self) -> Result<()> {
        let out = self.checked_add(rhs)?;
        *self = out;
        Ok(())
    }
    pub fn checked_sub_assign(&mut self, rhs: &Self) -> Result<()> {
        let out = self.checked_sub(rhs)?;
        *self = out;
        Ok(())
    }
    pub fn checked_scalar_mul_assign(&mut self, x: T) -> Result<()> {
        let out = self.checked_scalar_mul(x)?;
        *self = out;
        Ok(())
    }
    pub fn checked_scalar_div_assign(&mut self, x: T) -> Result<()> {
        let out = self.checked_scalar_div(x)?;
        *self = out;
        Ok(())
    }
}
