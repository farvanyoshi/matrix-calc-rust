use crate::error::{MatrixError, NumericError, ShapeError};
use crate::matrix::Matrix;
use num_traits::{Num, Zero};
use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Sub};

type AddResult<T, U> = <T as Add<U>>::Output;
type SubResult<T, U> = <T as Sub<U>>::Output;
type MulResult<T, U> = <T as Mul<U>>::Output;
type DivResult<T, U> = <T as Div<U>>::Output;

impl<T: Copy> Matrix<T> {
    pub fn negate(&self) -> Matrix<<T as Neg>::Output>
    where
        T: Neg,
    {
        self.map(|x| -(*x))
    }

    pub fn try_add<U: Copy>(
        &self,
        other: &Matrix<U>,
    ) -> Result<Matrix<AddResult<T, U>>, MatrixError>
    where
        T: Add<U>,
    {
        self.try_zip_map(other, |a, b| *a + *b)
    }

    pub fn try_sub<U: Copy>(
        &self,
        other: &Matrix<U>,
    ) -> Result<Matrix<SubResult<T, U>>, MatrixError>
    where
        T: Sub<U>,
    {
        self.try_zip_map(other, |a, b| *a - *b)
    }

    pub fn try_hadamard<U>(&self, other: &Matrix<U>) -> Result<Matrix<MulResult<T, U>>, MatrixError>
    where
        T: Mul<U>,
        U: Copy,
    {
        self.try_zip_map(other, |a, b| *a * *b)
    }

    pub fn hadamard<U>(&self, other: &Matrix<U>) -> Matrix<MulResult<T, U>>
    where
        T: Mul<U>,
        U: Copy,
    {
        self.try_hadamard(other).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_el_div<U>(&self, other: &Matrix<U>) -> Result<Matrix<DivResult<T, U>>, MatrixError>
    where
        T: Div<U>,
        U: Copy + Zero,
    {
        if self.rows() != other.rows() || self.cols() != other.cols() {
            return Err(ShapeError::Dimension { op: "element division", l_rows: self.rows(), l_cols: self.cols(), r_rows: other.rows(), r_cols: other.cols() }.into());
        }
        for x in other {
            if x.is_zero() {
                return Err(NumericError::DivideByZero.into());
            }
        }
        self.try_zip_map(other, |a, b| *a / *b)
    }

    pub fn el_div<U>(&self, other: &Matrix<U>) -> Matrix<DivResult<T, U>>
    where
        T: Div<U>,
        U: Copy + Zero,
    {
        self.try_el_div(other).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_scalar_mul<S>(&self, scalar: S) -> Result<Matrix<MulResult<T, S>>, MatrixError>
    where
        T: Mul<S>,
        S: Copy,
    {
        self.try_map(|x| *x * scalar)
    }

    pub fn scalar_mul<S>(&self, scalar: S) -> Matrix<MulResult<T, S>>
    where
        T: Mul<S>,
        S: Copy,
    {
        self.try_scalar_mul(scalar)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_scalar_div<S>(&self, scalar: S) -> Result<Matrix<DivResult<T, S>>, MatrixError>
    where
        T: Div<S>,
        S: Copy + Zero,
    {
        if scalar.is_zero() { return Err(NumericError::DivideByZero.into()); }
        self.try_map(|x| *x / scalar)
    }

    pub fn scalar_div<S>(&self, scalar: S) -> Matrix<DivResult<T, S>>
    where
        T: Div<S>,
        S: Copy + Zero,
    {
        self.try_scalar_div(scalar)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_matmul<U>(&self, other: &Matrix<U>) -> Result<Matrix<MulResult<T, U>>, MatrixError>
    where
        T: Mul<U>,
        U: Copy,
        MulResult<T, U>: Zero + Copy + AddAssign,
    {
        if self.cols() != other.rows() {
            return Err(ShapeError::Dimension {
                op: "matrix multiply",
                l_rows: self.rows(),
                l_cols: self.cols(),
                r_rows: other.rows(),
                r_cols: other.cols(),
            }
            .into());
        }

        let rows = self.rows();
        let cols = other.cols();
        let shared = self.cols();

        let mut data = vec![MulResult::<T, U>::zero(); crate::matrix::checked_size(rows, cols)?];

        for i in 0..rows {
            for k in 0..shared {
                let a = self.data()[i * shared + k];
                for j in 0..cols {
                    data[i * cols + j] += a * other.data()[k * cols + j];
                }
            }
        }

        Matrix::try_new(data, rows, cols)
    }

    pub fn matmul<U>(&self, other: &Matrix<U>) -> Matrix<MulResult<T, U>>
    where
        T: Mul<U>,
        U: Copy,
        MulResult<T, U>: Zero + Copy + AddAssign,
    {
        self.try_matmul(other).unwrap_or_else(|e| panic!("{e}"))
    }
}

impl<T, U> Add<&Matrix<U>> for &Matrix<T>
where
    T: Copy + Add<U>,
    U: Copy,
{
    type Output = Matrix<AddResult<T, U>>;

    fn add(self, rhs: &Matrix<U>) -> Self::Output {
        self.try_add(rhs).unwrap()
    }
}

impl<T, U> Sub<&Matrix<U>> for &Matrix<T>
where
    T: Copy + Sub<U>,
    U: Copy,
{
    type Output = Matrix<SubResult<T, U>>;

    fn sub(self, rhs: &Matrix<U>) -> Self::Output {
        self.try_sub(rhs).unwrap()
    }
}

impl<T, S> Mul<S> for &Matrix<T>
where
    T: Copy + Mul<S>,
    S: Copy + Num,
    MulResult<T, S>: Copy,
{
    type Output = Matrix<MulResult<T, S>>;

    fn mul(self, rhs: S) -> Self::Output {
        self.scalar_mul(rhs)
    }
}

impl<T, S> Div<S> for &Matrix<T>
where
    T: Copy + Div<S>,
    S: Copy + Num,
    DivResult<T, S>: Copy,
{
    type Output = Matrix<DivResult<T, S>>;

    fn div(self, rhs: S) -> Self::Output {
        self.scalar_div(rhs)
    }
}

impl<T, U> Mul<&Matrix<U>> for &Matrix<T>
where
    T: Copy + Mul<U>,
    U: Copy,
    MulResult<T, U>: Zero + Copy + AddAssign,
{
    type Output = Matrix<MulResult<T, U>>;

    fn mul(self, rhs: &Matrix<U>) -> Self::Output {
        self.matmul(rhs)
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        self.get(row, col).unwrap()
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        self.get_mut(row, col).unwrap()
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
