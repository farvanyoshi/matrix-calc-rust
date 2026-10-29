//! Borrowed strided views. All layouts originate in validated owned matrices.
use crate::{
    Matrix,
    error::{IndexError, Result},
};
use num_traits::Zero;
use std::ops::{AddAssign, Index, IndexMut, Mul, Range};

/// Read-only matrix access shared by owned matrices and strided views.
pub trait MatrixRead<T> {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn at(&self, row: usize, col: usize) -> &T;
}
impl<T> MatrixRead<T> for Matrix<T> {
    fn rows(&self) -> usize {
        self.rows()
    }
    fn cols(&self) -> usize {
        self.cols()
    }
    fn at(&self, r: usize, c: usize) -> &T {
        &self[(r, c)]
    }
}
#[derive(Clone, Copy, Debug)]
struct Layout {
    rows: usize,
    cols: usize,
    offset: usize,
    rs: usize,
    cs: usize,
}
impl Layout {
    fn index(self, r: usize, c: usize) -> Option<usize> {
        if r < self.rows && c < self.cols {
            Some(self.offset + r * self.rs + c * self.cs)
        } else {
            None
        }
    }
    fn transpose(self) -> Self {
        Self {
            rows: self.cols,
            cols: self.rows,
            rs: self.cs,
            cs: self.rs,
            ..self
        }
    }
    fn slice(self, rr: Range<usize>, cc: Range<usize>) -> Result<Self> {
        if rr.start > rr.end || cc.start > cc.end || rr.end > self.rows || cc.end > self.cols {
            return Err(IndexError::Range.into());
        }
        let rows = rr.end - rr.start;
        let cols = cc.end - cc.start;
        let offset = if rows == 0 || cols == 0 {
            self.offset
        } else {
            self.offset + rr.start * self.rs + cc.start * self.cs
        };
        Ok(Self {
            rows,
            cols,
            offset,
            ..self
        })
    }
}
/// Immutable rectangular view; transpose and slice are constant-time.
#[derive(Debug)]
pub struct MatrixView<'a, T> {
    data: &'a [T],
    layout: Layout,
}
impl<T> Copy for MatrixView<'_, T> {}
impl<T> Clone for MatrixView<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
/// Exclusive view. Reborrowing and consuming transformations preserve exclusivity.
#[derive(Debug)]
pub struct MatrixViewMut<'a, T> {
    data: &'a mut [T],
    layout: Layout,
}
impl<T> Matrix<T> {
    pub fn view(&self) -> MatrixView<'_, T> {
        MatrixView {
            data: &self.data,
            layout: Layout {
                rows: self.rows,
                cols: self.cols,
                offset: 0,
                rs: self.cols,
                cs: 1,
            },
        }
    }
    pub fn view_mut(&mut self) -> MatrixViewMut<'_, T> {
        MatrixViewMut {
            data: &mut self.data,
            layout: Layout {
                rows: self.rows,
                cols: self.cols,
                offset: 0,
                rs: self.cols,
                cs: 1,
            },
        }
    }
    pub fn slice(&self, rows: Range<usize>, cols: Range<usize>) -> Result<MatrixView<'_, T>> {
        self.view().slice(rows, cols)
    }
    pub fn slice_mut(
        &mut self,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Result<MatrixViewMut<'_, T>> {
        self.view_mut().slice(rows, cols)
    }
}
impl<'a, T> MatrixView<'a, T> {
    pub fn rows(&self) -> usize {
        self.layout.rows
    }
    pub fn cols(&self) -> usize {
        self.layout.cols
    }
    pub fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }
    pub fn get(&self, r: usize, c: usize) -> Option<&'a T> {
        self.layout.index(r, c).map(|i| &self.data[i])
    }
    pub fn transpose(self) -> Self {
        Self {
            layout: self.layout.transpose(),
            ..self
        }
    }
    pub fn slice(self, rows: Range<usize>, cols: Range<usize>) -> Result<Self> {
        Ok(Self {
            layout: self.layout.slice(rows, cols)?,
            ..self
        })
    }
    pub fn to_owned(&self) -> Matrix<T>
    where
        T: Clone,
    {
        self.map(Clone::clone)
    }
    pub fn map<U>(&self, mut f: impl FnMut(&T) -> U) -> Matrix<U> {
        Matrix::from_fn(self.rows(), self.cols(), |r, c| f(&self[(r, c)]))
    }
    pub fn iter(&self) -> impl Iterator<Item = &'a T> + DoubleEndedIterator + '_ {
        (0..self.rows()).flat_map(move |r| (0..self.cols()).map(move |c| self.get(r, c).unwrap()))
    }
    pub fn try_matmul<U: Clone, R: MatrixRead<U>>(
        &self,
        rhs: &R,
    ) -> Result<Matrix<<T as Mul<U>>::Output>>
    where
        T: Clone + Mul<U>,
        T::Output: Zero + Clone + AddAssign,
    {
        multiply(self, rhs)
    }
}
impl<T> MatrixRead<T> for MatrixView<'_, T> {
    fn rows(&self) -> usize {
        self.rows()
    }
    fn cols(&self) -> usize {
        self.cols()
    }
    fn at(&self, r: usize, c: usize) -> &T {
        &self[(r, c)]
    }
}
impl<T> Index<(usize, usize)> for MatrixView<'_, T> {
    type Output = T;
    fn index(&self, (r, c): (usize, usize)) -> &T {
        self.get(r, c).expect("view coordinate out of bounds")
    }
}
impl<'a, T> MatrixViewMut<'a, T> {
    pub fn rows(&self) -> usize {
        self.layout.rows
    }
    pub fn cols(&self) -> usize {
        self.layout.cols
    }
    pub fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }
    pub fn get(&self, r: usize, c: usize) -> Option<&T> {
        self.layout.index(r, c).map(|i| &self.data[i])
    }
    pub fn get_mut(&mut self, r: usize, c: usize) -> Option<&mut T> {
        let i = self.layout.index(r, c)?;
        Some(&mut self.data[i])
    }
    pub fn as_view(&self) -> MatrixView<'_, T> {
        MatrixView {
            data: self.data,
            layout: self.layout,
        }
    }
    pub fn reborrow(&mut self) -> MatrixViewMut<'_, T> {
        MatrixViewMut {
            data: self.data,
            layout: self.layout,
        }
    }
    pub fn transpose(self) -> Self {
        Self {
            layout: self.layout.transpose(),
            ..self
        }
    }
    pub fn slice(self, rows: Range<usize>, cols: Range<usize>) -> Result<Self> {
        Ok(Self {
            layout: self.layout.slice(rows, cols)?,
            ..self
        })
    }
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        for r in 0..self.rows() {
            for c in 0..self.cols() {
                self[(r, c)] = value.clone();
            }
        }
    }
}
impl<T> MatrixRead<T> for MatrixViewMut<'_, T> {
    fn rows(&self) -> usize {
        self.rows()
    }
    fn cols(&self) -> usize {
        self.cols()
    }
    fn at(&self, r: usize, c: usize) -> &T {
        &self[(r, c)]
    }
}
impl<T> Index<(usize, usize)> for MatrixViewMut<'_, T> {
    type Output = T;
    fn index(&self, (r, c): (usize, usize)) -> &T {
        self.get(r, c).expect("view coordinate out of bounds")
    }
}
impl<T> IndexMut<(usize, usize)> for MatrixViewMut<'_, T> {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut T {
        self.get_mut(r, c).expect("view coordinate out of bounds")
    }
}
pub(crate) fn multiply<T: Clone + Mul<U>, U: Clone, A: MatrixRead<T>, B: MatrixRead<U>>(
    a: &A,
    b: &B,
) -> Result<Matrix<T::Output>>
where
    T::Output: Zero + Clone + AddAssign,
{
    if a.cols() != b.rows() {
        return Err(crate::error::ShapeError::Dimension {
            op: "multiply",
            l_rows: a.rows(),
            l_cols: a.cols(),
            r_rows: b.rows(),
            r_cols: b.cols(),
        }
        .into());
    }
    let mut out = Matrix::<T::Output>::try_zeros(a.rows(), b.cols())?;
    for i in 0..a.rows() {
        for k in 0..a.cols() {
            for j in 0..b.cols() {
                out[(i, j)] += a.at(i, k).clone() * b.at(k, j).clone();
            }
        }
    }
    Ok(out)
}
