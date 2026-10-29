//! Arithmetic. Operators panic on shape/zero-divisor errors; named `try_*`
//! methods return them. Ordinary element arithmetic retains scalar semantics.
use crate::Matrix;
use crate::error::{NumericError, Result};
use num_traits::{Num, Zero};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

impl<T: Clone> Matrix<T> {
    /// Allocate elementwise negation using the scalar's ordinary arithmetic.
    pub fn negate(&self) -> Matrix<<T as Neg>::Output>
    where
        T: Neg,
    {
        self.map(|x| -x.clone())
    }
    /// Add identically shaped matrices; return a dimension error on mismatch.
    pub fn try_add<U: Clone>(&self, rhs: &Matrix<U>) -> Result<Matrix<<T as Add<U>>::Output>>
    where
        T: Add<U>,
    {
        self.try_zip_map(rhs, |a, b| a.clone() + b.clone())
    }
    /// Subtract identically shaped matrices; return a dimension error on mismatch.
    pub fn try_sub<U: Clone>(&self, rhs: &Matrix<U>) -> Result<Matrix<<T as Sub<U>>::Output>>
    where
        T: Sub<U>,
    {
        self.try_zip_map(rhs, |a, b| a.clone() - b.clone())
    }
    /// Multiply identically shaped matrices element by element.
    pub fn try_hadamard<U: Clone>(&self, rhs: &Matrix<U>) -> Result<Matrix<<T as Mul<U>>::Output>>
    where
        T: Mul<U>,
    {
        self.try_zip_map(rhs, |a, b| a.clone() * b.clone())
    }
    /// Elementwise multiplication; panics if shapes differ.
    pub fn hadamard<U: Clone>(&self, rhs: &Matrix<U>) -> Matrix<<T as Mul<U>>::Output>
    where
        T: Mul<U>,
    {
        self.try_hadamard(rhs).expect("incompatible matrix shapes")
    }
    /// Validate identical shapes, then reject zero divisors before elementwise division.
    pub fn try_element_div<U: Clone + Zero>(
        &self,
        rhs: &Matrix<U>,
    ) -> Result<Matrix<<T as Div<U>>::Output>>
    where
        T: Div<U>,
    {
        self.same_shape(rhs, "element division")?;
        if rhs.iter().any(Zero::is_zero) {
            return Err(NumericError::DivideByZero.into());
        }
        self.try_zip_map(rhs, |a, b| a.clone() / b.clone())
    }
    /// Elementwise division; panics for shape mismatch or a zero divisor.
    pub fn element_div<U: Clone + Zero>(&self, rhs: &Matrix<U>) -> Matrix<<T as Div<U>>::Output>
    where
        T: Div<U>,
    {
        self.try_element_div(rhs).expect("invalid element division")
    }
    /// Compatibility alias for [`Self::try_element_div`].
    pub fn try_el_div<U: Clone + Zero>(
        &self,
        rhs: &Matrix<U>,
    ) -> Result<Matrix<<T as Div<U>>::Output>>
    where
        T: Div<U>,
    {
        self.try_element_div(rhs)
    }
    /// Compatibility alias for `element_div`, including its panic behavior.
    pub fn el_div<U: Clone + Zero>(&self, rhs: &Matrix<U>) -> Matrix<<T as Div<U>>::Output>
    where
        T: Div<U>,
    {
        self.element_div(rhs)
    }
    /// Allocate elementwise addition of a scalar.
    pub fn scalar_add<S: Clone>(&self, x: S) -> Matrix<<T as Add<S>>::Output>
    where
        T: Add<S>,
    {
        self.map(|a| a.clone() + x.clone())
    }
    /// Allocate elementwise subtraction of a scalar.
    pub fn scalar_sub<S: Clone>(&self, x: S) -> Matrix<<T as Sub<S>>::Output>
    where
        T: Sub<S>,
    {
        self.map(|a| a.clone() - x.clone())
    }
    /// Allocate elementwise multiplication by a scalar; ordinary scalar overflow rules apply.
    pub fn scalar_mul<S: Clone>(&self, x: S) -> Matrix<<T as Mul<S>>::Output>
    where
        T: Mul<S>,
    {
        self.map(|a| a.clone() * x.clone())
    }
    /// Compatibility wrapper. Numeric overflow is checked by `checked_scalar_mul`.
    pub fn try_scalar_mul<S: Clone>(&self, x: S) -> Result<Matrix<<T as Mul<S>>::Output>>
    where
        T: Mul<S>,
    {
        Ok(self.scalar_mul(x))
    }
    /// Reject a zero scalar, then divide each element using scalar arithmetic.
    pub fn try_scalar_div<S: Clone + Zero>(&self, x: S) -> Result<Matrix<<T as Div<S>>::Output>>
    where
        T: Div<S>,
    {
        if x.is_zero() {
            return Err(NumericError::DivideByZero.into());
        }
        Ok(self.map(|a| a.clone() / x.clone()))
    }
    /// Divide every element by a scalar; panics for a zero divisor.
    pub fn scalar_div<S: Clone + Zero>(&self, x: S) -> Matrix<<T as Div<S>>::Output>
    where
        T: Div<S>,
    {
        self.try_scalar_div(x).expect("invalid scalar division")
    }
    /// Multiply compatible matrices, checking output dimensions and reservation failure.
    pub fn try_matmul<U: Clone>(&self, rhs: &Matrix<U>) -> Result<Matrix<<T as Mul<U>>::Output>>
    where
        T: Mul<U>,
        <T as Mul<U>>::Output: Zero + Clone + AddAssign,
    {
        if self.cols() != rhs.rows() {
            return Err(self.dimension_error(rhs, "multiply"));
        }
        let mut out = Matrix::try_zeros(self.rows(), rhs.cols())?;
        multiply_owned_into(self, rhs, &mut out)?;
        Ok(out)
    }

    /// Matrix multiplication; panics on incompatible dimensions or allocation failure.
    pub fn matmul<U: Clone>(&self, rhs: &Matrix<U>) -> Matrix<<T as Mul<U>>::Output>
    where
        T: Mul<U>,
        <T as Mul<U>>::Output: Zero + Clone + AddAssign,
    {
        self.try_matmul(rhs).expect("invalid matrix multiplication")
    }
    /// Compute into a pre-sized output, validating all shapes before writing.
    pub fn matmul_into(&self, rhs: &Self, out: &mut Self) -> Result<()>
    where
        T: Mul<Output = T> + Zero + AddAssign,
    {
        multiply_owned_into(self, rhs, out)
    }
}
fn multiply_owned_into<T: Clone + Mul<U>, U: Clone>(
    a: &Matrix<T>,
    b: &Matrix<U>,
    out: &mut Matrix<T::Output>,
) -> Result<()>
where
    T::Output: Zero + Clone + AddAssign,
{
    if a.cols() != b.rows() {
        return Err(a.dimension_error(b, "multiply"));
    }
    if out.shape() != (a.rows(), b.cols()) {
        return Err(crate::ShapeError::Dimension {
            op: "write multiplication",
            l_rows: a.rows(),
            l_cols: b.cols(),
            r_rows: out.rows(),
            r_cols: out.cols(),
        }
        .into());
    }
    for cell in out.data_mut() {
        *cell = T::Output::zero();
    }
    for r in 0..a.rows() {
        let output = out.row_mut(r).unwrap();
        for (k, x) in a.row(r).unwrap().iter().enumerate() {
            for (cell, y) in output.iter_mut().zip(b.row(k).unwrap()) {
                *cell += x.clone() * y.clone();
            }
        }
    }
    Ok(())
}
macro_rules! binary {
    ($trait:ident,$method:ident,$try_method:ident) => {
        impl<T: Clone + std::ops::$trait<U>, U: Clone> std::ops::$trait<&Matrix<U>> for &Matrix<T> {
            type Output = Matrix<<T as std::ops::$trait<U>>::Output>;
            fn $method(self, rhs: &Matrix<U>) -> Self::Output {
                self.$try_method(rhs).expect("incompatible matrix shapes")
            }
        }
        impl<T: Clone + std::ops::$trait<U>, U: Clone> std::ops::$trait<Matrix<U>> for Matrix<T> {
            type Output = Matrix<<T as std::ops::$trait<U>>::Output>;
            fn $method(self, rhs: Matrix<U>) -> Self::Output {
                (&self).$method(&rhs)
            }
        }
        impl<T: Clone + std::ops::$trait<U>, U: Clone> std::ops::$trait<&Matrix<U>> for Matrix<T> {
            type Output = Matrix<<T as std::ops::$trait<U>>::Output>;
            fn $method(self, rhs: &Matrix<U>) -> Self::Output {
                (&self).$method(rhs)
            }
        }
        impl<T: Clone + std::ops::$trait<U>, U: Clone> std::ops::$trait<Matrix<U>> for &Matrix<T> {
            type Output = Matrix<<T as std::ops::$trait<U>>::Output>;
            fn $method(self, rhs: Matrix<U>) -> Self::Output {
                self.$method(&rhs)
            }
        }
    };
}
binary!(Add, add, try_add);
binary!(Sub, sub, try_sub);
impl<T: Clone + Neg> Neg for &Matrix<T> {
    type Output = Matrix<T::Output>;
    fn neg(self) -> Self::Output {
        self.negate()
    }
}
impl<T: Neg> Neg for Matrix<T> {
    type Output = Matrix<T::Output>;
    fn neg(self) -> Self::Output {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.into_iter().map(Neg::neg).collect(),
        }
    }
}
impl<T: Clone + Mul<U>, U: Clone> Mul<&Matrix<U>> for &Matrix<T>
where
    T::Output: Zero + Clone + AddAssign,
{
    type Output = Matrix<T::Output>;
    fn mul(self, rhs: &Matrix<U>) -> Self::Output {
        self.matmul(rhs)
    }
}
impl<T: Clone + Mul<U>, U: Clone> Mul<Matrix<U>> for Matrix<T>
where
    T::Output: Zero + Clone + AddAssign,
{
    type Output = Matrix<T::Output>;
    fn mul(self, rhs: Matrix<U>) -> Self::Output {
        self.matmul(&rhs)
    }
}
impl<T: Clone + Mul<U>, U: Clone> Mul<&Matrix<U>> for Matrix<T>
where
    T::Output: Zero + Clone + AddAssign,
{
    type Output = Matrix<T::Output>;
    fn mul(self, rhs: &Matrix<U>) -> Self::Output {
        self.matmul(rhs)
    }
}
impl<T: Clone + Mul<U>, U: Clone> Mul<Matrix<U>> for &Matrix<T>
where
    T::Output: Zero + Clone + AddAssign,
{
    type Output = Matrix<T::Output>;
    fn mul(self, rhs: Matrix<U>) -> Self::Output {
        self.matmul(&rhs)
    }
}
impl<T: Clone + Mul<S>, S: Num + Clone> Mul<S> for &Matrix<T> {
    type Output = Matrix<T::Output>;
    fn mul(self, rhs: S) -> Self::Output {
        self.scalar_mul(rhs)
    }
}
impl<T: Clone + Mul<S>, S: Num + Clone> Mul<S> for Matrix<T> {
    type Output = Matrix<T::Output>;
    fn mul(self, rhs: S) -> Self::Output {
        self.scalar_mul(rhs)
    }
}
impl<T: Clone + Div<S>, S: Num + Clone> Div<S> for &Matrix<T> {
    type Output = Matrix<T::Output>;
    fn div(self, rhs: S) -> Self::Output {
        self.scalar_div(rhs)
    }
}
impl<T: Clone + Div<S>, S: Num + Clone> Div<S> for Matrix<T> {
    type Output = Matrix<T::Output>;
    fn div(self, rhs: S) -> Self::Output {
        self.scalar_div(rhs)
    }
}
impl<T: Clone + Add<Output = T>> AddAssign<&Matrix<T>> for Matrix<T> {
    fn add_assign(&mut self, rhs: &Matrix<T>) {
        self.same_shape(rhs, "add assign")
            .expect("incompatible matrix shapes");
        for (a, b) in self.data.iter_mut().zip(&rhs.data) {
            *a = a.clone() + b.clone();
        }
    }
}
impl<T: Clone + Sub<Output = T>> SubAssign<&Matrix<T>> for Matrix<T> {
    fn sub_assign(&mut self, rhs: &Matrix<T>) {
        self.same_shape(rhs, "subtract assign")
            .expect("incompatible matrix shapes");
        for (a, b) in self.data.iter_mut().zip(&rhs.data) {
            *a = a.clone() - b.clone();
        }
    }
}
impl<T: Clone + Mul<Output = T> + Num> MulAssign<T> for Matrix<T> {
    fn mul_assign(&mut self, rhs: T) {
        for a in &mut self.data {
            *a = a.clone() * rhs.clone();
        }
    }
}
impl<T: Clone + Div<Output = T> + Num> DivAssign<T> for Matrix<T> {
    fn div_assign(&mut self, rhs: T) {
        assert!(!rhs.is_zero(), "cannot divide by zero");
        for a in &mut self.data {
            *a = a.clone() / rhs.clone();
        }
    }
}
impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;
    fn index(&self, (r, c): (usize, usize)) -> &T {
        self.get(r, c).expect("matrix coordinate out of bounds")
    }
}
impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut T {
        self.get_mut(r, c).expect("matrix coordinate out of bounds")
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ops() {
        let x = Matrix::new(1..5, 2, 2);
        let y = x.negate();
        assert_eq!(y[(1, 1)], -4);

        assert_eq!(&x + &x, &x * 2);
        assert_eq!(&x - &x, Matrix::zeros(2, 2));
    }

    #[test]
    fn test_element_div_mul() {
        let m = Matrix::new([2; 9], 3, 3);
        let product = m.hadamard(&Matrix::new(0..9, 3, 3));
        for (i, j) in product.iter().enumerate() {
            assert_eq!(i * 2, *j);
        }

        assert_eq!(m.el_div(&Matrix::new([1; 9], 3, 3)), m);
        let res = m.try_el_div(&Matrix::<usize>::zeros(3, 3));
        assert!(res.is_err());
    }

    #[test]
    fn test_matmul() {
        let m = Matrix::new(0..24, 8, 3);
        let n = Matrix::new((1..=15).map(|x| -x), 3, 5);
        #[rustfmt::skip]
        let exp = Matrix::new(
            [
                -28, -31, -34, -37, -40,
                -82, -94, -106, -118, -130,
                -136, -157, -178, -199, -220,
                -190, -220, -250, -280, -310,
                -244, -283, -322, -361, -400,
                -298, -346, -394, -442, -490,
                -352, -409, -466, -523, -580,
                -406, -472, -538, -604, -670,
            ], 
            8,
            5
        );
        assert_eq!(m.try_matmul(&n).unwrap(), exp);
    }

    #[test]
    #[should_panic]
    fn test_bad_matmul() {
        let x = Matrix::new(0..24, 3, 8);
        let y = Matrix::new(6..12, 2, 3);
        let _res = &x * &y;
    }
}
