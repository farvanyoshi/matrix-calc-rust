use crate::matrix::base::Matrix;
use std::{iter::IntoIterator};

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
        self.data.chunks(self.dims.cols)
    }
}

struct ColumnIter<T> {
    matrix: Matrix<T>,
    col_index: usize,
    next_row: usize
}

impl<T> Iterator for ColumnIter<T> {
    type Item = T;

    fn next(&self) -> Option<Self::Item> {
        if self.next_row < self.matrix.dims.rows {
            let index = self.next_row * self.matrix.dims.cols + self.col_index
        }
    }
}