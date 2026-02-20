use crate::error::{MatrixError, ShapeError, NumericError};
use crate::matrix::Matrix;
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, Index, IndexMut};
use num_traits::{Zero, Num};

type AddResult<T, U> = <T as Add<U>>::Output;
type SubResult<T, U> = <T as Sub<U>>::Output;
type MulResult<T, U> = <T as Mul<U>>::Output;
type DivResult<T, U> = <T as Div<U>>::Output;

impl<T: Copy> Matrix<T> {
    pub fn neg<U: Copy>(&self) -> Matrix<<T as Neg>::Output> where T: Neg {
        self.map(|x| x.neg())
    }

    pub fn try_add<U: Copy>(&self, other: &Matrix<U>) -> Result<Matrix<AddResult<T, U>>, MatrixError> where T: Add<U> {
        self.try_zip_map(other, |a, b| Ok(*a + *b))
    }

    pub fn add<U: Copy>(&self, other: &Matrix<U>) -> Matrix<AddResult<T, U>> where T: Add<U> {
        self.try_add(other).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_sub<U: Copy>(&self, other: &Matrix<U>) -> Result<Matrix<SubResult<T, U>>, MatrixError> where T: Sub<U> {
        self.try_zip_map(other, |a, b| Ok(*a - *b))
    }

    pub fn sub<U: Copy>(&self, other: &Matrix<U>) -> Matrix<SubResult<T, U>> where T: Sub<U> {
        self.try_sub(other).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_hadamard<U>(&self, other: &Matrix<U>) -> Result<Matrix<MulResult<T, U>>, MatrixError>
    where
        T: Mul<U>,
        U: Copy
    {
        self.try_zip_map(other, |a, b| Ok(*a * *b))
    }

    pub fn hadamard<U>(&self, other: &Matrix<U>) -> Matrix<MulResult<T, U>>
    where
        T: Mul<U>,
        U: Copy
    {
        self.try_hadamard(other).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_el_div<U>(&self, other: &Matrix<U>) -> Result<Matrix<DivResult<T, U>>, MatrixError>
    where
        T: Div<U>,
        U: Copy + Zero
    {
        self.try_zip_map(other, |a, b| {
            if b.is_zero() {
                Err(NumericError::DivideByZero.into())
            } else {
                Ok(*a / *b)
            }
        })
    }

    pub fn el_div<U>(&self, other: &Matrix<U>) -> Matrix<DivResult<T, U>>
    where
        T: Div<U>,
        U: Copy + Zero
    {
        self.try_el_div(other).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_scalar_mul<S>(&self, scalar: S) -> Result<Matrix<MulResult<T, S>>, MatrixError>
    where
        T: Mul<S>,
        S: Copy
    {
        self.try_map(|x| Ok(*x * scalar))
    }

    pub fn scalar_mul<S>(&self, scalar: S) -> Matrix<MulResult<T, S>> 
    where
        T: Mul<S>,
        S: Copy
    {
        self.try_scalar_mul(scalar).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_scalar_div<S>(&self, scalar: S) -> Result<Matrix<DivResult<T, S>>, MatrixError>
    where
        T: Div<S>,
        S: Copy
    {
        self.try_map(|x| Ok(*x / scalar))
    }

    pub fn scalar_div<S>(&self, scalar: S) -> Matrix<DivResult<T, S>> 
    where
        T: Div<S>,
        S: Copy
    {
        self.try_scalar_div(scalar).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_matmul<U>(&self, other: &Matrix<U>) -> Result<Matrix<MulResult<T, U>>, MatrixError>
    where
        T: Mul<U>,
        U: Copy,
        MulResult<T, U>: Zero + Copy + AddAssign
    
    {
        if self.cols != other.rows {
            return Err(ShapeError::Dimension {
                op: "matrix multiply",
                l_rows: self.rows,
                l_cols: self.cols,
                r_rows: other.rows,
                r_cols: other.cols
            }.into());
        }
        
        let rows = self.rows;
        let cols = other.cols;
        let shared = self.cols;
        
        let mut data = vec![MulResult::<T, U>::zero(); rows * cols];

        for i in 0..rows {
            for k in 0..shared {
                let a = self.data[i * shared + k];
                for j in 0..cols {
                    data[i * cols + j] += a * other.data[k * cols + j];
                }
            }
        }

        Ok(Matrix { rows, cols, data })
    }

    pub fn matmul<U>(&self, other: &Matrix<U>) -> Matrix<MulResult<T, U>>
    where
        T: Mul<U>,
        U: Copy,
        MulResult<T, U>: Zero + Copy + AddAssign
    {
        self.try_matmul(other).unwrap_or_else(|e| panic!("{e}"))
    }
}
    
impl<T, U> Add<&Matrix<U>> for &Matrix<T>
    where T: Copy + Add<U>, U: Copy
{
    type Output = Matrix<AddResult<T, U>>;

    fn add(self, rhs: &Matrix<U>) -> Self::Output {
        self.add(rhs)
    }
}

impl<T, U> Sub<&Matrix<U>> for &Matrix<T>
    where T: Copy + Sub<U>, U: Copy
{
    type Output = Matrix<SubResult<T, U>>;

    fn sub(self, rhs: &Matrix<U>) -> Self::Output {
        self.sub(rhs)
    }
}

impl<T, S> Mul<S> for &Matrix<T>
where
    T: Copy + Mul<S>,
    S: Copy + Num,
    MulResult<T, S>: Copy
{
    type Output = Matrix<MulResult<T, S>>;

    fn mul(self, rhs: S) -> Self::Output {
        self.map(|x| *x * rhs)
    }
}

impl<T, U> Mul<&Matrix<U>> for &Matrix<T>
where
    T: Copy + Mul<U>,
    U: Copy,
    MulResult<T, U>: Zero + Copy + AddAssign
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