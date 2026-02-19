use crate::matrix::base::Matrix;
use std::iter::IntoIterator;

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