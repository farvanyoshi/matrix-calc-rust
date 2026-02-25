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

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.data.get(row * self.cols + col)
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        self.data.get_mut(row * self.cols + col)
    }

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

    pub fn from_vec(data: Vec<T>, rows: usize, cols: usize) -> Result<Self, error::MatrixError> {
        if data.len() != rows * cols {
            return Err(error::ShapeError::Reshape {
                exp_rows: rows,
                exp_cols: cols,
                len: data.len()
            }.into())
        }

        Ok(Matrix {
            rows,
            cols,
            data
        })
    }
}

impl<T> Matrix<T> {

    pub fn try_zip_map<U, R>(&self, other: &Matrix<U>, f: impl Fn(&T, &U) -> R)
    -> Result<Matrix<R>, error::MatrixError> {
        Matrix::from_vec(
        self.into_iter()
            .zip(other.into_iter())
            .map(|(a, b)| f(a, b))
            .collect::<Vec<_>>(),
            self.rows,
            self.cols
        )
    }

    pub fn zip_map<U, R>(&self, other: &Matrix<U>, f: impl Fn(&T, &U) -> R)
    -> Matrix<R> {
        self.try_zip_map(other, f).unwrap()
    }
}

impl<T: Clone> Matrix<T> {
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

    pub fn identity(side: usize) -> Matrix<T> where T: Zero + One {
        let mut data = vec![T::zero(); side * side];
        for i in 0..side {
            data[i * side + i] = T::one();
        }

        Matrix {
            rows: side,
            cols: side,
            data
        }
    }
}

impl<T> IntoIterator for Matrix<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Matrix<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<T> Matrix<T> {
    pub fn iter_rows(&'_ self) -> std::slice::Chunks<'_, T> {
        self.data.chunks(self.cols)
    }

    pub fn iter_cols(&self, col: usize) -> impl ExactSizeIterator<Item = &T> + DoubleEndedIterator {
        assert!(col < self.cols);

        self.data.iter().skip(col).step_by(self.cols)
    }
}