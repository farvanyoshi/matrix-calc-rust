pub(crate) fn checked_size(rows: usize, cols: usize) -> Result<usize, crate::MatrixError> {
    rows.checked_mul(cols).ok_or(crate::error::ShapeError::Overflow { rows, cols }.into())
}

use crate::error;
use num_traits::{One, Zero};
use std::iter::IntoIterator;

#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T> Matrix<T> {
    pub fn try_new<I>(input: I, rows: usize, cols: usize) -> Result<Self, error::MatrixError>
    where
        I: IntoIterator<Item = T>,
    {
        let len = checked_size(rows, cols)?;
        let data: Vec<T> = input.into_iter().collect();

        if len != data.len() {
            return Err(error::ShapeError::Reshape {
                exp_rows: rows,
                exp_cols: cols,
                len: data.len(),
            }
            .into());
        }

        Ok(Self { rows, cols, data })
    }

    pub fn new<I>(input: I, rows: usize, cols: usize) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Matrix::try_new(input, rows, cols).unwrap()
    }

    pub fn zeros(rows: usize, cols: usize) -> Self
    where
        T: Zero + Clone,
    {
        Self {
            rows,
            cols,
            data: vec![T::zero(); checked_size(rows, cols).expect("matrix dimensions overflow")],
        }
    }

    pub fn ones(rows: usize, cols: usize) -> Self
    where
        T: One + Clone,
    {
        Self {
            rows,
            cols,
            data: vec![T::one(); checked_size(rows, cols).expect("matrix dimensions overflow")],
        }
    }

    pub fn identity(side: usize) -> Self
    where
        T: Zero + One + Clone,
    {
        let mut data = vec![T::zero(); checked_size(side, side).expect("matrix dimensions overflow")];
        for i in 0..side {
            data[i * side + i] = T::one();
        }

        Self {
            rows: side,
            cols: side,
            data,
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
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.rows || col >= self.cols { return None; }
        self.data.get(row * self.cols + col)
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row >= self.rows || col >= self.cols { return None; }
        self.data.get_mut(row * self.cols + col)
    }
}

impl<T> Matrix<T> {
    pub fn try_map<U>(&self, f: impl Fn(&T) -> U) -> Result<Matrix<U>, error::MatrixError> {
        Matrix::try_new(self.iter().map(f), self.rows, self.cols)
    }

    pub fn map<U>(&self, f: impl Fn(&T) -> U) -> Matrix<U> {
        self.try_map(f).unwrap()
    }

    pub fn try_zip_map<U, R>(
        &self,
        other: &Matrix<U>,
        f: impl Fn(&T, &U) -> R,
    ) -> Result<Matrix<R>, error::MatrixError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(error::ShapeError::Dimension { op: "zip", l_rows: self.rows, l_cols: self.cols, r_rows: other.rows, r_cols: other.cols }.into());
        }
        Matrix::try_new(
            self.iter().zip(other).map(|(a, b)| f(a, b)),
            self.rows,
            self.cols,
        )
    }

    pub fn zip_map<U, R>(&self, other: &Matrix<U>, f: impl Fn(&T, &U) -> R) -> Matrix<R> {
        self.try_zip_map(other, f).unwrap()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
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
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Matrix<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T> Matrix<T> {
    pub fn iter_rows(&self) -> impl ExactSizeIterator<Item = &[T]> + DoubleEndedIterator {
        (0..self.rows).map(|r| &self.data[r*self.cols..(r+1)*self.cols])
    }

    pub fn iter_col(&self, col: usize) -> impl ExactSizeIterator<Item = &T> + DoubleEndedIterator {
        assert!(col < self.cols);

        self.data.iter().skip(col).step_by(self.cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_matrix() {
        let manual_matrix = Matrix {
            rows: 2,
            cols: 3,
            data: vec![1, 2, 3, 4, 5, 6],
        };
        let function_matrix = Matrix::new([1, 2, 3, 4, 5, 6], 2, 3);

        assert_eq!(manual_matrix, function_matrix);
    }

    #[test]
    #[should_panic]
    fn test_bad_new_matrix() {
        let _ = Matrix::new([2.0, 3.0, 4.0], 4, 5);
    }

    #[test]
    fn test_specific_matrices() {
        let mut zero_matrix: Matrix<f64> = Matrix::zeros(2, 2);
        let identity_matrix: Matrix<f64> = Matrix::identity(2);

        for i in 1..3 {
            zero_matrix.data[i * i - 1] = 1.0;
        }

        assert_eq!(zero_matrix, identity_matrix);
    }

    #[test]
    fn test_basic_functions() {
        let mut test_matrix = Matrix::new([4, 3, 3, 2, 1, 0], 2, 3);
        assert_eq!(test_matrix.rows(), test_matrix.rows);
        assert_eq!(test_matrix.rows(), 2);
        assert_eq!(test_matrix.cols(), test_matrix.cols);
        assert_eq!(test_matrix.cols(), 3);
        assert_eq!(test_matrix.len(), 6);
        assert_eq!(*test_matrix.get(1, 2).unwrap(), 0);

        *test_matrix.get_mut(1, 2).unwrap() = 54;

        assert_eq!(*test_matrix.get(1, 2).unwrap(), 54);
        assert_eq!(
            *test_matrix.get(1, 2).unwrap(),
            test_matrix.data()[1 * 3 + 2]
        )
    }

    #[test]
    fn test_map() {
        let base_matrix = Matrix::new([1, 3, 7, 15, 31, 63, 127, 255].map(|x| x as u8), 4, 2);

        let mapped_matrix = base_matrix.try_map(|x| *x ^ 0b11111111);
        assert!(mapped_matrix.is_ok());
    }

    #[test]
    #[should_panic]
    fn test_bad_map() {
        let m = Matrix::new(0..16, 4, 4);
        let _ = m.map(|x| x / 0);
    }

    #[test]
    fn test_zip_map() {
        let m = Matrix::new(0..4, 2, 2);
        let n: Matrix<i32> = Matrix::ones(2, 2);

        let zipped = m.try_zip_map(&n, |a, b: &i32| {
            let x = *a as f64;
            let y = *b as f64;
            1.0 / ((1.0 / x) + (1.0 / y))
        });
        assert!(zipped.is_ok());
    }

    #[test]
    fn test_iter() {
        let m = Matrix::new([9, 8, 7, 6, 5, 4, 3, 2, 1], 3, 3);
        let mut n = Matrix::new(45 as u16..54 as u16, 3, 3);

        let mut x = m.iter();
        let mut y = m.iter();
        assert_eq!(x.next(), y.next());

        *n.iter_mut().next().unwrap() = 9;
        *n.iter_mut().nth(4).unwrap() = 27;

        assert_eq!(*n.get(1, 1).unwrap(), 27 as u16);
        assert_eq!(m[(0, 0)], n[(0, 0)]);
    }

    #[test]
    fn test_iter_rows() {
        let m = Matrix::new(0..9, 3, 3);
        let mut x = m.iter_rows();
        let temp = x.next().unwrap();
        let temp2 = &m.data[0..3];

        assert_eq!(*temp, *temp2)
    }

    #[test]
    fn test_iter_cols() {
        let m: Matrix<&str> = Matrix::new(["a", "collection", "of", "flowers"], 2, 2);
        let mut x = m.iter_col(1);
        assert_eq!(*x.next().unwrap(), "collection");
        assert_eq!(*x.next().unwrap(), "flowers");
    }

    #[test]
    #[should_panic]
    fn test_bad_iter_cols() {
        let m: Matrix<&str> = Matrix::new(["a", "collection", "of", "flowers"], 2, 2);
        m.iter_col(12).next();
    }
}
