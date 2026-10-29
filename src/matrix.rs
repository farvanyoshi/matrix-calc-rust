//! Owned, row-major matrix storage and validated element access.
use crate::error::{IndexError, MatrixError, NumericError, Result, ShapeError};
use num_traits::{NumCast, One, ToPrimitive, Zero};
use std::fmt;
use std::iter::FusedIterator;

pub(crate) fn checked_size(rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .ok_or(ShapeError::Overflow { rows, cols }.into())
}

/// A dense row-major matrix. Shapes with zero rows or columns are supported.
///
/// Fields are private so all safe public operations preserve `len == rows*cols`.
/// Infallible constructors and operators panic on structural errors; use their
/// `try_*` counterparts to recover. Element callbacks and scalar arithmetic may
/// panic independently, and ordinary allocation may abort on exhaustion.
#[must_use]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matrix<T> {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) data: Vec<T>,
}
impl<T> Matrix<T> {
    /// Collect row-major input. Consumes at most the expected element count + 1.
    pub fn try_new<I: IntoIterator<Item = T>>(input: I, rows: usize, cols: usize) -> Result<Self> {
        let len = checked_size(rows, cols)?;
        let mut data = Vec::new();
        for value in input {
            if data.len() == len {
                return Err(ShapeError::Reshape {
                    exp_rows: rows,
                    exp_cols: cols,
                    len: len.saturating_add(1),
                }
                .into());
            }
            data.try_reserve(1)?;
            data.push(value);
        }
        Self::try_from_vec(data, rows, cols)
    }
    /// Panicking counterpart to [`Self::try_new`].
    pub fn new<I: IntoIterator<Item = T>>(input: I, rows: usize, cols: usize) -> Self {
        Self::try_new(input, rows, cols).expect("invalid matrix input")
    }
    /// Validate and take ownership of an existing vector without copying.
    pub fn try_from_vec(data: Vec<T>, rows: usize, cols: usize) -> Result<Self> {
        if checked_size(rows, cols)? != data.len() {
            return Err(ShapeError::Reshape {
                exp_rows: rows,
                exp_cols: cols,
                len: data.len(),
            }
            .into());
        }
        Ok(Self { rows, cols, data })
    }
    /// Construct from a coordinate callback, visiting coordinates in row order.
    pub fn try_from_fn(
        rows: usize,
        cols: usize,
        mut f: impl FnMut(usize, usize) -> T,
    ) -> Result<Self> {
        let mut data = Vec::new();
        data.try_reserve_exact(checked_size(rows, cols)?)?;
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        Ok(Self { rows, cols, data })
    }
    pub fn from_fn(rows: usize, cols: usize, f: impl FnMut(usize, usize) -> T) -> Self {
        Self::try_from_fn(rows, cols, f).expect("invalid matrix dimensions")
    }
    pub fn try_filled(rows: usize, cols: usize, value: T) -> Result<Self>
    where
        T: Clone,
    {
        let len = checked_size(rows, cols)?;
        let mut data = Vec::new();
        data.try_reserve_exact(len)?;
        data.resize(len, value);
        Ok(Self { rows, cols, data })
    }
    pub fn filled(rows: usize, cols: usize, value: T) -> Self
    where
        T: Clone,
    {
        Self::try_filled(rows, cols, value).expect("invalid matrix dimensions")
    }
    pub fn try_zeros(rows: usize, cols: usize) -> Result<Self>
    where
        T: Zero + Clone,
    {
        Self::try_filled(rows, cols, T::zero())
    }
    pub fn zeros(rows: usize, cols: usize) -> Self
    where
        T: Zero + Clone,
    {
        Self::try_zeros(rows, cols).expect("invalid matrix dimensions")
    }
    pub fn try_ones(rows: usize, cols: usize) -> Result<Self>
    where
        T: One + Clone,
    {
        Self::try_filled(rows, cols, T::one())
    }
    pub fn ones(rows: usize, cols: usize) -> Self
    where
        T: One + Clone,
    {
        Self::try_ones(rows, cols).expect("invalid matrix dimensions")
    }
    pub fn try_identity(side: usize) -> Result<Self>
    where
        T: Zero + One + Clone,
    {
        let mut m = Self::try_zeros(side, side)?;
        for i in 0..side {
            m.data[i * side + i] = T::one();
        }
        Ok(m)
    }
    pub fn identity(side: usize) -> Self
    where
        T: Zero + One + Clone,
    {
        Self::try_identity(side).expect("invalid matrix dimensions")
    }
    /// Empty outer input denotes 0×0; ragged rows are rejected.
    pub fn try_from_rows(rows: Vec<Vec<T>>) -> Result<Self> {
        let nr = rows.len();
        let nc = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|r| r.len() != nc) {
            return Err(ShapeError::Ragged.into());
        }
        Self::try_new(rows.into_iter().flatten(), nr, nc)
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }
    pub fn data(&self) -> &[T] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
    pub fn as_slice(&self) -> &[T] {
        self.data()
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data_mut()
    }
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub(crate) fn coordinate(&self, row: usize, col: usize) -> Result<usize> {
        match (row < self.rows, col < self.cols) {
            (true, true) => Ok(row * self.cols + col),
            (false, true) => Err(IndexError::RowOOB {
                row,
                rows: self.rows,
            }
            .into()),
            (true, false) => Err(IndexError::ColOOB {
                col,
                cols: self.cols,
            }
            .into()),
            (false, false) => Err(IndexError::FullOOB {
                row,
                col,
                rows: self.rows,
                cols: self.cols,
            }
            .into()),
        }
    }
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.coordinate(row, col).ok().map(|i| &self.data[i])
    }
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        let i = self.coordinate(row, col).ok()?;
        Some(&mut self.data[i])
    }
    pub fn try_get(&self, row: usize, col: usize) -> Result<&T> {
        Ok(&self.data[self.coordinate(row, col)?])
    }
    pub fn try_get_mut(&mut self, row: usize, col: usize) -> Result<&mut T> {
        let i = self.coordinate(row, col)?;
        Ok(&mut self.data[i])
    }
    pub fn set(&mut self, row: usize, col: usize, value: T) -> Result<()> {
        *self.try_get_mut(row, col)? = value;
        Ok(())
    }
    pub fn row(&self, row: usize) -> Option<&[T]> {
        if row >= self.rows {
            None
        } else {
            Some(&self.data[row * self.cols..(row + 1) * self.cols])
        }
    }
    pub fn row_mut(&mut self, row: usize) -> Option<&mut [T]> {
        if row >= self.rows {
            None
        } else {
            Some(&mut self.data[row * self.cols..(row + 1) * self.cols])
        }
    }
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }
    pub fn iter_rows(
        &self,
    ) -> impl ExactSizeIterator<Item = &[T]> + DoubleEndedIterator + FusedIterator {
        (0..self.rows).map(|r| &self.data[r * self.cols..(r + 1) * self.cols])
    }
    pub fn iter_rows_mut(&mut self) -> RowsMut<'_, T> {
        RowsMut {
            rest: &mut self.data,
            cols: self.cols,
            rows: self.rows,
        }
    }
    /// Panics if the requested column is outside the logical shape.
    pub fn iter_col(
        &self,
        col: usize,
    ) -> impl ExactSizeIterator<Item = &T> + DoubleEndedIterator + FusedIterator {
        assert!(col < self.cols, "column out of bounds");
        self.data.iter().skip(col).step_by(self.cols).fuse()
    }
    pub fn iter_col_mut(
        &mut self,
        col: usize,
    ) -> impl ExactSizeIterator<Item = &mut T> + DoubleEndedIterator + FusedIterator {
        assert!(col < self.cols, "column out of bounds");
        self.data.iter_mut().skip(col).step_by(self.cols).fuse()
    }
    pub fn map<U>(&self, mut f: impl FnMut(&T) -> U) -> Matrix<U> {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(&mut f).collect(),
        }
    }
    /// Fallible callback mapping. Stops at the first callback error.
    pub fn try_map<U, E>(
        &self,
        f: impl FnMut(&T) -> std::result::Result<U, E>,
    ) -> std::result::Result<Matrix<U>, E> {
        Ok(Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .map(f)
                .collect::<std::result::Result<Vec<_>, _>>()?,
        })
    }
    pub(crate) fn same_shape<U>(&self, other: &Matrix<U>, op: &'static str) -> Result<()> {
        if self.shape() != other.shape() {
            return Err(self.dimension_error(other, op));
        }
        Ok(())
    }
    pub(crate) fn dimension_error<U>(&self, other: &Matrix<U>, op: &'static str) -> MatrixError {
        ShapeError::Dimension {
            op,
            l_rows: self.rows,
            l_cols: self.cols,
            r_rows: other.rows,
            r_cols: other.cols,
        }
        .into()
    }
    pub(crate) fn square(&self) -> Result<usize> {
        if !self.is_square() {
            return Err(ShapeError::NotSquare {
                rows: self.rows,
                cols: self.cols,
            }
            .into());
        }
        Ok(self.rows)
    }
    pub fn try_zip_map<U, R>(
        &self,
        other: &Matrix<U>,
        mut f: impl FnMut(&T, &U) -> R,
    ) -> Result<Matrix<R>> {
        self.same_shape(other, "zip")?;
        Matrix::try_new(
            self.iter().zip(other).map(|(a, b)| f(a, b)),
            self.rows,
            self.cols,
        )
    }
    pub fn zip_map<U, R>(&self, other: &Matrix<U>, f: impl FnMut(&T, &U) -> R) -> Matrix<R> {
        self.try_zip_map(other, f)
            .expect("incompatible matrix shapes")
    }
    /// Explicit numeric conversion with range checking; floats may truncate to integers.
    pub fn try_cast<U: NumCast>(&self) -> Result<Matrix<U>>
    where
        T: ToPrimitive + Copy,
    {
        self.try_map(|x| U::from(*x).ok_or(NumericError::Conversion.into()))
    }
    /// Numeric conversion that must round-trip exactly through the source type.
    pub fn try_cast_exact<U: NumCast + ToPrimitive + Copy>(&self) -> Result<Matrix<U>>
    where
        T: NumCast + Copy + PartialEq,
    {
        self.try_map(|x| {
            let y = U::from(*x).ok_or(NumericError::Conversion)?;
            if T::from(y) != Some(*x) {
                return Err(NumericError::Conversion.into());
            }
            Ok(y)
        })
    }
}
/// Disjoint mutable rows, including zero-length rows. No unsafe code is used.
pub struct RowsMut<'a, T> {
    rest: &'a mut [T],
    cols: usize,
    rows: usize,
}
impl<'a, T> Iterator for RowsMut<'a, T> {
    type Item = &'a mut [T];
    fn next(&mut self) -> Option<Self::Item> {
        if self.rows == 0 {
            return None;
        }
        self.rows -= 1;
        let (row, rest) = std::mem::take(&mut self.rest).split_at_mut(self.cols);
        self.rest = rest;
        Some(row)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.rows, Some(self.rows))
    }
}
impl<T> DoubleEndedIterator for RowsMut<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.rows == 0 {
            return None;
        }
        self.rows -= 1;
        let split = self.rest.len() - self.cols;
        let (rest, row) = std::mem::take(&mut self.rest).split_at_mut(split);
        self.rest = rest;
        Some(row)
    }
}
impl<T> ExactSizeIterator for RowsMut<'_, T> {}
impl<T> FusedIterator for RowsMut<'_, T> {}
impl<T> IntoIterator for Matrix<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
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
impl<T, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T> {
    fn from(rows: [[T; C]; R]) -> Self {
        Self::new(rows.into_iter().flatten(), R, C)
    }
}
impl<T> TryFrom<Vec<Vec<T>>> for Matrix<T> {
    type Error = MatrixError;
    fn try_from(rows: Vec<Vec<T>>) -> Result<Self> {
        Self::try_from_rows(rows)
    }
}
impl<T: fmt::Display> fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{} [", self.rows, self.cols)?;
        for (i, row) in self.iter_rows().enumerate() {
            if i > 0 {
                write!(f, "; ")?;
            }
            for (j, x) in row.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{x}")?;
            }
        }
        write!(f, "]")
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

        let mapped_matrix = base_matrix.try_map(|x| Ok::<_, ()>(*x ^ 0b11111111));
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
