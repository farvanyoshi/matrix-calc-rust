//! Shape changes, block operations, vectors, and generic reductions.
use crate::{
    Matrix, MatrixRead,
    error::{IndexError, NumericError, Result, ShapeError},
};
use num_traits::{One, Zero};
use std::ops::{Add, AddAssign, Mul};
impl<T> Matrix<T> {
    pub fn try_reshape(&mut self, rows: usize, cols: usize) -> Result<()> {
        if crate::matrix::checked_size(rows, cols)? != self.len() {
            return Err(ShapeError::Reshape {
                exp_rows: rows,
                exp_cols: cols,
                len: self.len(),
            }
            .into());
        }
        self.rows = rows;
        self.cols = cols;
        Ok(())
    }
    pub fn reshape(&mut self, rows: usize, cols: usize) {
        self.try_reshape(rows, cols).expect("invalid reshape")
    }
    pub fn try_reshaped(mut self, rows: usize, cols: usize) -> Result<Self> {
        self.try_reshape(rows, cols)?;
        Ok(self)
    }
    pub fn into_transpose(self) -> Self {
        let rows = self.rows;
        let cols = self.cols;
        let mut data: Vec<_> = self.data.into_iter().map(Some).collect();
        Self::from_fn(cols, rows, |r, c| data[c * cols + r].take().unwrap())
    }
    /// In-place transpose for square matrices; rectangular transpose uses `into_transpose`.
    pub fn transpose_in_place(&mut self) -> Result<()> {
        let n = self.square()?;
        for r in 0..n {
            for c in 0..r {
                self.data.swap(r * n + c, c * n + r);
            }
        }
        Ok(())
    }
    pub fn swap_rows(&mut self, a: usize, b: usize) -> Result<()> {
        for r in [a, b] {
            if r >= self.rows {
                return Err(IndexError::RowOOB {
                    row: r,
                    rows: self.rows,
                }
                .into());
            }
        }
        for c in 0..self.cols {
            self.data.swap(a * self.cols + c, b * self.cols + c);
        }
        Ok(())
    }
    pub fn swap_cols(&mut self, a: usize, b: usize) -> Result<()> {
        for c in [a, b] {
            if c >= self.cols {
                return Err(IndexError::ColOOB {
                    col: c,
                    cols: self.cols,
                }
                .into());
            }
        }
        for r in 0..self.rows {
            self.data.swap(r * self.cols + a, r * self.cols + b);
        }
        Ok(())
    }
    pub(crate) fn vector_len(&self) -> Result<usize> {
        if self.rows == 1 || self.cols == 1 {
            Ok(self.len())
        } else {
            Err(ShapeError::NotVector.into())
        }
    }
}
impl<T: Clone> Matrix<T> {
    pub fn transpose(&self) -> Self {
        self.view().transpose().to_owned()
    }
    pub fn diagonal(&self) -> Vec<T> {
        (0..self.rows.min(self.cols))
            .map(|i| self[(i, i)].clone())
            .collect()
    }
    pub fn from_diagonal(diagonal: &[T]) -> Self
    where
        T: Zero,
    {
        Self::from_fn(diagonal.len(), diagonal.len(), |r, c| {
            if r == c {
                diagonal[r].clone()
            } else {
                T::zero()
            }
        })
    }
    pub fn to_rows(&self) -> Vec<Vec<T>> {
        self.iter_rows().map(<[T]>::to_vec).collect()
    }
    pub fn set_row(&mut self, row: usize, values: &[T]) -> Result<()> {
        if row >= self.rows {
            return Err(IndexError::RowOOB {
                row,
                rows: self.rows,
            }
            .into());
        }
        if values.len() != self.cols {
            return Err(ShapeError::Reshape {
                exp_rows: 1,
                exp_cols: self.cols,
                len: values.len(),
            }
            .into());
        }
        self.row_mut(row).unwrap().clone_from_slice(values);
        Ok(())
    }
    pub fn set_col(&mut self, col: usize, values: &[T]) -> Result<()> {
        if col >= self.cols {
            return Err(IndexError::ColOOB {
                col,
                cols: self.cols,
            }
            .into());
        }
        if values.len() != self.rows {
            return Err(ShapeError::Reshape {
                exp_rows: self.rows,
                exp_cols: 1,
                len: values.len(),
            }
            .into());
        }
        for (a, b) in self.iter_col_mut(col).zip(values) {
            *a = b.clone();
        }
        Ok(())
    }
    pub fn hcat(&self, other: &Self) -> Result<Self> {
        if self.rows != other.rows {
            return Err(self.dimension_error(other, "concatenate columns"));
        }
        let cols = self
            .cols
            .checked_add(other.cols)
            .ok_or(ShapeError::Overflow {
                rows: self.rows,
                cols: usize::MAX,
            })?;
        Self::try_from_fn(self.rows, cols, |r, c| {
            if c < self.cols {
                self[(r, c)].clone()
            } else {
                other[(r, c - self.cols)].clone()
            }
        })
    }
    pub fn vcat(&self, other: &Self) -> Result<Self> {
        if self.cols != other.cols {
            return Err(self.dimension_error(other, "concatenate rows"));
        }
        let rows = self
            .rows
            .checked_add(other.rows)
            .ok_or(ShapeError::Overflow {
                rows: usize::MAX,
                cols: self.cols,
            })?;
        Self::try_new(self.iter().chain(other).cloned(), rows, self.cols)
    }
    pub fn insert_block<R: MatrixRead<T>>(
        &mut self,
        row: usize,
        col: usize,
        block: &R,
    ) -> Result<()> {
        if row > self.rows
            || col > self.cols
            || block.rows() > self.rows - row
            || block.cols() > self.cols - col
        {
            return Err(IndexError::Range.into());
        }
        for r in 0..block.rows() {
            for c in 0..block.cols() {
                self[(row + r, col + c)] = block.at(r, c).clone();
            }
        }
        Ok(())
    }
    pub fn try_matmul_with<U: Clone, R: MatrixRead<U>>(
        &self,
        other: &R,
    ) -> Result<Matrix<<T as Mul<U>>::Output>>
    where
        T: Mul<U>,
        T::Output: Zero + Clone + AddAssign,
    {
        crate::view::multiply(self, other)
    }
    pub fn dot(&self, other: &Self) -> Result<T>
    where
        T: Zero + Add<Output = T> + Mul<Output = T>,
    {
        if self.vector_len()? != other.vector_len()? {
            return Err(self.dimension_error(other, "dot"));
        }
        Ok(self
            .iter()
            .zip(other)
            .fold(T::zero(), |s, (a, b)| s + a.clone() * b.clone()))
    }
    pub fn outer(&self, other: &Self) -> Result<Self>
    where
        T: Mul<Output = T>,
    {
        let m = self.vector_len()?;
        let n = other.vector_len()?;
        Self::try_from_fn(m, n, |r, c| self.data[r].clone() * other.data[c].clone())
    }
    pub fn sum(&self) -> T
    where
        T: Zero + Add<Output = T>,
    {
        self.iter().cloned().fold(T::zero(), Add::add)
    }
    pub fn product(&self) -> T
    where
        T: One + Mul<Output = T>,
    {
        self.iter().cloned().fold(T::one(), Mul::mul)
    }
    pub fn trace(&self) -> Result<T>
    where
        T: Zero + Add<Output = T>,
    {
        let n = self.square()?;
        Ok((0..n).fold(T::zero(), |s, i| s + self[(i, i)].clone()))
    }
    pub fn row_sums(&self) -> Self
    where
        T: Zero + Add<Output = T>,
    {
        Self::from_fn(self.rows, 1, |r, _| {
            self.row(r)
                .unwrap()
                .iter()
                .cloned()
                .fold(T::zero(), Add::add)
        })
    }
    pub fn col_sums(&self) -> Self
    where
        T: Zero + Add<Output = T>,
    {
        Self::from_fn(1, self.cols, |_, c| {
            self.iter_col(c).cloned().fold(T::zero(), Add::add)
        })
    }
    pub fn row_products(&self) -> Self
    where
        T: One + Mul<Output = T>,
    {
        Self::from_fn(self.rows, 1, |r, _| {
            self.row(r)
                .unwrap()
                .iter()
                .cloned()
                .fold(T::one(), Mul::mul)
        })
    }
    pub fn col_products(&self) -> Self
    where
        T: One + Mul<Output = T>,
    {
        Self::from_fn(1, self.cols, |_, c| {
            self.iter_col(c).cloned().fold(T::one(), Mul::mul)
        })
    }
    pub fn argmin(&self) -> Result<(usize, usize)>
    where
        T: PartialOrd,
    {
        self.argextreme(false)
    }
    pub fn argmax(&self) -> Result<(usize, usize)>
    where
        T: PartialOrd,
    {
        self.argextreme(true)
    }
    fn argextreme(&self, max: bool) -> Result<(usize, usize)>
    where
        T: PartialOrd,
    {
        if self.is_empty() {
            return Err(NumericError::Empty.into());
        }
        let mut best = 0;
        for i in 0..self.len() {
            let ord = self.data[i]
                .partial_cmp(&self.data[best])
                .ok_or(NumericError::NonFinite)?;
            if (max && ord.is_gt()) || (!max && ord.is_lt()) {
                best = i;
            }
        }
        Ok((best / self.cols, best % self.cols))
    }
    pub fn min(&self) -> Result<T>
    where
        T: PartialOrd,
    {
        Ok(self[self.argmin()?].clone())
    }
    pub fn max(&self) -> Result<T>
    where
        T: PartialOrd,
    {
        Ok(self[self.argmax()?].clone())
    }
    pub fn pow(&self, mut exponent: u32) -> Result<Self>
    where
        T: Zero + One + Mul<Output = T> + AddAssign,
    {
        let n = self.square()?;
        let mut out = Self::try_identity(n)?;
        let mut base = self.clone();
        while exponent > 0 {
            if exponent & 1 == 1 {
                out = out.try_matmul(&base)?;
            }
            exponent >>= 1;
            if exponent > 0 {
                base = base.try_matmul(&base)?;
            }
        }
        Ok(out)
    }
}
