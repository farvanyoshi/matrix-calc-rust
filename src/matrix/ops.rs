use crate::error;
use crate::matrix::base::Matrix;
use std::ops;
use num_traits::Zero;

impl<T> Matrix<T>
where T: Copy
{
    pub fn neg(&self) -> Matrix<<T as ops::Neg>::Output> where T: ops::Neg {
        self.map(|x| x.neg())
    }

    pub fn try_add(&self, other: &Self) -> Result<Matrix<<T as ops::Add>::Output>, error::MatrixError> where T: ops::Add {
        self.try_zip_map(other, |a, b| Ok(*a + *b))
    }

    pub fn add(&self, other: &Self) -> Matrix<<T as ops::Add>::Output> where T: ops::Add {
        self.try_add(other).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_sub(&self, other: &Self) -> Result<Matrix<<T as ops::Sub>::Output>, error::MatrixError> where T: ops::Sub {
        self.try_zip_map(other, |a, b| Ok(*a - *b))
    }

    pub fn sub(&self, other: &Self) -> Matrix<<T as ops::Sub>::Output> where T: ops::Sub {
        self.try_sub(other).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn matmul(&self, other: &Self) -> Result<Self, error::MatrixError>
    where T: ops::AddAssign + ops::Mul<Output=T> + Zero {
        if self.cols != other.rows {
            return Err(error::ShapeError::Dimension {
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
        
        let mut data = vec![T::zero(); rows * cols];

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

    pub fn hadamard(&self, other: &Self) -> Self where T: ops::Mul<Output=T> {
        self.zip_map(other, |a, b| *a * *b)
    }

    pub fn el_div(&self, other: &Self) -> Result<Self, error::MatrixError> where T: ops::Div<Output=T> + PartialEq + Zero {
        self.try_zip_map(other, |a, b| {
            if *b == T::zero() {
                Err(error::NumericError::DivideByZero.into())
            } else {
                Ok(*a / *b)
            }
        })
    }
}
    
impl<T> ops::Add<&Matrix<T>> for &Matrix<T>
    where T: Copy + ops::Add<Output = T>
{
    type Output = Matrix<T>;

    fn add(self, rhs: &Matrix<T>) -> Self::Output {
        self.add(rhs)
    }
}

impl<T> ops::Sub<&Matrix<T>> for &Matrix<T>
    where T: Copy + ops::Sub<Output = T>
{
    type Output = Matrix<T>;

    fn sub(self, rhs: &Matrix<T>) -> Self::Output {
        self.sub(rhs)
    }
}

impl<T> ops::Mul<&Matrix<T>> for &Matrix<T>
    where T: Copy + ops::Mul<Output = T> + ops::AddAssign + Zero
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        self.matmul(rhs).unwrap()
    }
}

impl<T> ops::Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        self.get(row, col).unwrap()
    }
}

impl<T> ops::IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        self.get_mut(row, col).unwrap()
    }
}