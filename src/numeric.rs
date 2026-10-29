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
    /// Construct validated finite nonnegative absolute and relative tolerances.
    pub fn new(atol: T, rtol: T) -> Result<Self> {
        if !atol.is_finite() || !rtol.is_finite() || atol < T::zero() || rtol < T::zero() {
            return Err(NumericError::InvalidOptions.into());
        }
        Ok(Self { atol, rtol })
    }
    /// Absolute tolerance in the original scalar units.
    pub fn absolute(&self) -> T {
        self.atol
    }
    /// Dimensionless relative tolerance.
    pub fn relative(&self) -> T {
        self.rtol
    }
    /// Default tolerance: zero absolute and `8*max(dimension,1)*epsilon` relative.
    pub fn for_dimension(dimension: usize) -> Self {
        Self {
            atol: T::zero(),
            rtol: T::epsilon() * number::<T>(dimension.max(1)) * number::<T>(8),
        }
    }
    /// Compute `atol + rtol*abs(scale)`; this raw expression may overflow for extreme tolerances.
    pub fn threshold(&self, scale: T) -> T {
        self.atol + self.rtol * scale.abs()
    }
    /// Overflow-safe mixed absolute/relative comparison of finite scalar values.
    pub fn close(&self, a: T, b: T) -> bool {
        if !a.is_finite() || !b.is_finite() {
            return false;
        }
        let scale = a.abs().max(b.abs());
        scale == T::zero() || (a / scale - b / scale).abs() <= self.atol / scale + self.rtol
    }
    pub(crate) fn scaled(&self, scale: T) -> T {
        self.atol / scale + self.rtol
    }
}
/// Decomposition settings. Iterative methods count sweeps, not scalar rotations.
#[derive(Clone, Copy, Debug)]
pub struct DecompositionOptions<T> {
    /// Optional explicit tolerance; None uses dimension-aware defaults.
    pub tolerance: Option<Tolerance<T>>,
    /// Positive bound on complete iterative sweeps.
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
// Recombine binary exponents to avoid overflowing an intermediate ratio even
// when the final projected solution is representable. All scale factors are
// finite and positive; zeros are dealt with before decoding.
pub(crate) fn scaled_ratio<T: Real>(x: T, numerator: T, denominator: T, other: T) -> T {
    if x == T::zero() || numerator == T::zero() {
        return T::zero();
    }
    fn part<T: Real>(value: T) -> (T, i32) {
        let (mantissa, exponent, sign) = value.integer_decode();
        let bits = 63 - mantissa.leading_zeros();
        let fraction = T::from_u64(mantissa).unwrap() / T::from_u64(1u64 << bits).unwrap();
        (
            if sign < 0 { -fraction } else { fraction },
            i32::from(exponent) + bits as i32,
        )
    }
    let (xm, xe) = part(x);
    let (nm, ne) = part(numerator);
    let (dm, de) = part(denominator);
    let (om, oe) = part(other);
    let mut value = (xm * nm) / (dm * om);
    let mut exponent = xe + ne - de - oe;
    while exponent > 0 {
        let step = exponent.min(32);
        value *= T::from_f64(2.0).unwrap().powi(step);
        exponent -= step;
    }
    while exponent < 0 {
        let step = (-exponent).min(32);
        value *= T::from_f64(2.0).unwrap().powi(-step);
        exponent += step;
    }
    value
}
impl<T: Real> Matrix<T> {
    /// Whether every element is finite; an empty matrix is finite.
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
    /// Largest absolute element, or zero for empty input; use `all_finite` to reject NaN first.
    pub fn max_abs(&self) -> T {
        self.iter().fold(T::zero(), |m, x| m.max(x.abs()))
    }
    /// Compare identical shapes elementwise using overflow-safe finite scalar comparisons.
    pub fn approx_eq(&self, other: &Self, tolerance: Tolerance<T>) -> bool {
        self.shape() == other.shape()
            && self.iter().zip(other).all(|(a, b)| tolerance.close(*a, *b))
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
    /// Scaled compensated mean; empty or nonfinite input is an error.
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
    /// Row means as `rows×1`; existing empty groups or nonfinite data are errors.
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
    /// Column means as `1×cols`; existing empty groups or nonfinite data are errors.
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
    /// Stable Euclidean norm of a finite row/column vector.
    pub fn vector_norm_2(&self) -> Result<T> {
        self.vector_len()?;
        self.finite()?;
        Ok(self.norm_frobenius())
    }
    /// Sum of absolute values of a finite row/column vector.
    pub fn vector_norm_1(&self) -> Result<T> {
        self.vector_len()?;
        self.finite()?;
        Ok(self.map(|x| x.abs()).sum_compensated())
    }
    /// Largest absolute value of a finite row/column vector.
    pub fn vector_norm_inf(&self) -> Result<T> {
        self.vector_len()?;
        self.finite()?;
        Ok(self.max_abs())
    }
    /// Induced matrix 1-norm: maximum absolute column sum; nonfinite input is an error.
    pub fn norm_1(&self) -> Result<T> {
        self.finite()?;
        if self.is_empty() {
            return Ok(T::zero());
        }
        let mut max = T::zero();
        for c in 0..self.cols() {
            let v = Self::new(self.iter_col(c).map(|x| x.abs()), self.rows(), 1);
            max = max.max(v.sum_compensated());
        }
        Ok(max)
    }
    /// Induced matrix infinity-norm: maximum absolute row sum; nonfinite input is an error.
    pub fn norm_inf(&self) -> Result<T> {
        self.finite()?;
        if self.is_empty() {
            return Ok(T::zero());
        }
        let mut max = T::zero();
        for row in self.iter_rows() {
            let v = Self::new(row.iter().map(|x| x.abs()), 1, row.len());
            max = max.max(v.sum_compensated());
        }
        Ok(max)
    }
    /// Normalize a finite row/column vector after scaling; a zero vector is an error.
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
    /// Scaled compensated dot product of equally sized finite row/column vectors.
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
    /// Allocate absolute element values.
    pub fn abs_elements(&self) -> Self {
        self.map(|x| x.abs())
    }
    /// Allocate elementwise square roots; invalid real arguments follow IEEE NaN semantics.
    pub fn sqrt_elements(&self) -> Self {
        self.map(|x| x.sqrt())
    }
    /// Allocate elementwise exponentials, retaining IEEE overflow behavior.
    pub fn exp_elements(&self) -> Self {
        self.map(|x| x.exp())
    }
    /// Allocate elementwise natural logarithms, retaining IEEE exceptional values.
    pub fn ln_elements(&self) -> Self {
        self.map(|x| x.ln())
    }
    /// Allocate elementwise real powers using the scalar's IEEE semantics.
    pub fn powf_elements(&self, p: T) -> Self {
        self.map(|x| x.powf(p))
    }
    /// Whether finite square input matches its transpose at the requested entrywise tolerance.
    pub fn is_symmetric(&self, tol: Tolerance<T>) -> bool {
        self.is_square()
            && self.all_finite()
            && (0..self.rows()).all(|r| (0..r).all(|c| tol.close(self[(r, c)], self[(c, r)])))
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
    /// Add same-shaped integer matrices, returning shape or overflow errors.
    pub fn checked_add(&self, rhs: &Self) -> Result<Self> {
        self.checked_zip(rhs, num_traits::CheckedAdd::checked_add)
    }
    /// Subtract same-shaped integer matrices, returning shape or overflow errors.
    pub fn checked_sub(&self, rhs: &Self) -> Result<Self> {
        self.checked_zip(rhs, num_traits::CheckedSub::checked_sub)
    }
    /// Elementwise integer multiplication with shape and overflow checks.
    pub fn checked_hadamard(&self, rhs: &Self) -> Result<Self> {
        self.checked_zip(rhs, num_traits::CheckedMul::checked_mul)
    }
    /// Integer elementwise division with shape, zero, and overflow checks.
    pub fn checked_element_div(&self, rhs: &Self) -> Result<Self> {
        self.same_shape(rhs, "checked division")?;
        if rhs.iter().any(|x| x.is_zero()) {
            return Err(NumericError::DivideByZero.into());
        }
        self.checked_zip(rhs, num_traits::CheckedDiv::checked_div)
    }
    /// Integer negation with overflow checks, including unsigned nonzero values.
    pub fn checked_negate(&self) -> Result<Self> {
        self.try_map(|x| x.checked_neg().ok_or(NumericError::Overflow.into()))
    }
    /// Integer scalar addition with overflow checking.
    pub fn checked_scalar_add(&self, x: T) -> Result<Self> {
        self.try_map(|a| a.checked_add(&x).ok_or(NumericError::Overflow.into()))
    }
    /// Integer scalar subtraction with overflow checking.
    pub fn checked_scalar_sub(&self, x: T) -> Result<Self> {
        self.try_map(|a| a.checked_sub(&x).ok_or(NumericError::Overflow.into()))
    }
    /// Integer scalar multiplication with overflow checking.
    pub fn checked_scalar_mul(&self, x: T) -> Result<Self> {
        self.try_map(|a| a.checked_mul(&x).ok_or(NumericError::Overflow.into()))
    }
    /// Integer scalar division with zero and signed-overflow checks.
    pub fn checked_scalar_div(&self, x: T) -> Result<Self> {
        if x.is_zero() {
            return Err(NumericError::DivideByZero.into());
        }
        self.try_map(|a| a.checked_div(&x).ok_or(NumericError::Overflow.into()))
    }
    /// Integer matrix multiplication with checked products and sums.
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
    /// Compute checked addition before replacing the destination, preserving it on error.
    pub fn checked_add_assign(&mut self, rhs: &Self) -> Result<()> {
        let out = self.checked_add(rhs)?;
        *self = out;
        Ok(())
    }
    /// Compute checked subtraction before replacing the destination, preserving it on error.
    pub fn checked_sub_assign(&mut self, rhs: &Self) -> Result<()> {
        let out = self.checked_sub(rhs)?;
        *self = out;
        Ok(())
    }
    /// Compute checked scaling before replacing the destination, preserving it on error.
    pub fn checked_scalar_mul_assign(&mut self, x: T) -> Result<()> {
        let out = self.checked_scalar_mul(x)?;
        *self = out;
        Ok(())
    }
    /// Compute checked division before replacing the destination, preserving it on error.
    pub fn checked_scalar_div_assign(&mut self, x: T) -> Result<()> {
        let out = self.checked_scalar_div(x)?;
        *self = out;
        Ok(())
    }
}
