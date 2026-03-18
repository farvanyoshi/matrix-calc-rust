use crate::error;
use num_traits::{Zero, One};
use std::iter::IntoIterator;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    rows: usize, 
    cols: usize,
    data: Vec<T>
}

impl<T> Matrix<T> {
    pub fn new<I>(input: I, rows: usize, cols: usize) -> Result<Self, error::MatrixError>
    where I: IntoIterator<Item = T> {
        let data: Vec<T> = input
                        .into_iter()
                        .collect();

        if rows * cols != data.len() {
            return Err(error::ShapeError::Reshape {
                exp_rows: rows,
                exp_cols: cols,
                len: data.len()
            }.into())
        }

        Ok(Self { rows, cols, data })
        
    }

    pub fn zeros(rows: usize, cols: usize) -> Self
    where T: Zero + Clone {
        Self {
            rows, cols,
            data: vec![T::zero(); rows * cols]
        }
    }

    pub fn identity(side: usize) -> Self
    where T: Zero + One + Clone {
        let mut data = vec![T::zero(); side * side];
        for i in 0..side {
            data[i * side + i] = T::one();
        }

        Self {
            rows: side,
            cols: side,
            data
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.data.get(row * self.cols + col)
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        self.data.get_mut(row * self.cols + col)
    }
}

impl<T> Matrix<T> {

    pub fn try_zip_map<U, R>(&self, other: &Matrix<U>, f: impl Fn(&T, &U) -> R)
    -> Result<Matrix<R>, error::MatrixError> {
        Matrix::new(
        self.into_iter()
            .zip(other.into_iter())
            .map(|(a, b)| f(a, b)),
            self.rows,
            self.cols
        )
    }

    pub fn zip_map<U, R>(&self, other: &Matrix<U>, f: impl Fn(&T, &U) -> R)
    -> Matrix<R> {
        self.try_zip_map(other, f).unwrap()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_matrix() {

    }
}