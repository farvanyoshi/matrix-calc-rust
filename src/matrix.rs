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
}

impl<T> Matrix<T> {
    pub fn try_map<U, E>(&self, f: impl Fn(&T) -> Result<U, E>) -> Result<Matrix<U>, E> {
        let data: Vec<U> = self.data.iter()
                            .map(f)
                            .collect::<Result<_, _>>()?;
        
        Ok(Matrix {
            rows: self.rows,
            cols: self.cols,
            data
        })
    }

    pub fn map<U>(&self, f: impl Fn(&T) -> U) -> Matrix<U> {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(f).collect()
        }
    }

    pub fn try_zip_map<U, R>(&self, other: &Matrix<U>, f: impl Fn(&T, &U) -> Result<R, error::MatrixError>)
    -> Result<Matrix<R>, error::MatrixError> {
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

    pub fn zip_map<U, R>(&self, other: &Matrix<U>, f: impl Fn(&T, &U) -> R)
    -> Matrix<R> {
        self.try_zip_map(other, |a, b| Ok(f(a, b))).unwrap_or_else(|e| panic!("{e}"))
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