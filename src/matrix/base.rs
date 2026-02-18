use crate::error;
use num_traits::{Zero, One};
use std::iter::IntoIterator;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    pub rows: usize, 
    pub cols: usize,
    pub data: Vec<T>
}

impl<T> Matrix<T> {
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

// TODO: implement matrix multiplication (outer product)
// TODO: convert to TensorStorage<T> in the future
impl<T> Matrix<T> 
where T: Copy
{
    pub fn new<I>(rows: usize, cols: usize, input: I) -> Result<Matrix<T>, error::MatrixError>
    where I: IntoIterator<Item = T> {
        let data: Vec<T> = input
                        .into_iter()
                        .collect();

        if rows * cols != data.len() {
            return Err(error::ShapeError::Reshape{
                exp_rows: rows,
                exp_cols: cols,
                len: data.len()
            }.into())
        }

        Ok(Matrix { rows, cols, data })
        
    }

    pub fn zeros(rows: usize, cols: usize) -> Matrix<T>
    where T: Zero {
        Matrix {
            rows, cols,
            data: vec![T::zero(); rows * cols]
        }
    }

    pub fn ones(rows: usize, cols: usize) -> Matrix<T> where T: One {
        Matrix {
            rows, cols,
            data: vec![T::one(); rows * cols]
        }
    }

    pub fn identity(size: usize) -> Matrix<T> where T: Zero + One {
        let mut data = vec![T::zero(); size * size];
        for i in 0..size {
            data[(i + 1) * size] = T::one();
        }

        Matrix {
            rows: size,
            cols: size,
            data
        }
    }

    pub fn try_map<U>(&self, f: impl Fn(&T) -> Result<U, error::MatrixError>) 
    -> Result<Matrix<U>, error::MatrixError> {
        let mut data = Vec::with_capacity(self.data.len());
        for value in &self.data {
            data.push(f(value)?);
        }

        Ok(Matrix { rows: self.rows, cols: self.cols, data } )
    }

    pub fn map<U>(&self, f: impl Fn(&T) -> U) -> Matrix<U> {
        self.try_map(|x| Ok(f(x))).unwrap()
    }

    pub fn try_zip_map<U>(&self, other: &Self, f: impl Fn(&T, &T) -> Result<U, error::MatrixError>)
    -> Result<Matrix<U>, error::MatrixError> {
        if (self.rows != other.rows) && (self.cols != other.cols) {
            return Err(error::ShapeError::Dimension {
                op: "do element-wise operations for",
                l_rows: self.rows,
                l_cols: self.cols,
                r_rows: self.rows,
                r_cols: self.cols
            }.into())
        }

        let mut data = Vec::with_capacity(self.data.len());
        for (a, b) in self.data.iter().zip(&other.data) {
            data.push(f(a, b)?)
        }

        Ok(Matrix { rows: self.rows, cols: self.cols, data })
    }

    pub fn zip_map<U>(&self, other: &Self, f: impl Fn(&T, &T) -> U)
    -> Matrix<U> {
        self.try_zip_map(other, |a, b| Ok(f(a, b))).unwrap_or_else(|e| panic!("{e}"))
    }
}