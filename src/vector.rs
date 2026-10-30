//! Specialized contiguous one-dimensional vectors and their matrix interaction.
use crate::{Matrix, NumericError, Real, Result, ShapeError};
use num_traits::Zero;
use std::{
    fmt,
    ops::{Add, Index, IndexMut, Mul, Neg, Sub},
};

/// An owned contiguous 1-D vector; direct slice kernels avoid matrix indexing.
#[must_use]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vector<T> {
    data: Vec<T>,
}
impl<T> Vector<T> {
    /// Own a vector buffer. Shape Requirements: any length. Complexity: O(1).
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }
    /// Collect elements. Shape Requirements: any finite iterator. Complexity: O(n).
    pub fn from_iter_values(values: impl IntoIterator<Item = T>) -> Self {
        Self::new(values.into_iter().collect())
    }
    /// Logical length. Shape Requirements: none. Complexity: O(1).
    pub fn len(&self) -> usize {
        self.data.len()
    }
    /// Whether length is zero. Shape Requirements: none. Complexity: O(1).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    /// Contiguous elements. Shape Requirements: none. Complexity: O(1).
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
    /// Mutable elements without resizing. Shape Requirements: none. Complexity: O(1).
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
    /// Consume the buffer. Shape Requirements: none. Complexity: O(1).
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
    /// Checked element access. Shape Requirements: index<n. Complexity: O(1).
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }
    /// Shape-preserving callback transformation. Shape Requirements: none. Complexity: O(n).
    pub fn map<U>(&self, f: impl FnMut(&T) -> U) -> Vector<U> {
        Vector::new(self.data.iter().map(f).collect())
    }
    /// Consume into a 1×n matrix. Shape Requirements: none. Complexity: O(1).
    pub fn into_row_matrix(self) -> Matrix<T> {
        let n = self.len();
        Matrix::try_from_vec(self.data, 1, n).expect("valid vector shape")
    }
    /// Consume into an n×1 matrix. Shape Requirements: none. Complexity: O(1).
    pub fn into_col_matrix(self) -> Matrix<T> {
        let n = self.len();
        Matrix::try_from_vec(self.data, n, 1).expect("valid vector shape")
    }
    fn same_len<U>(&self, rhs: &Vector<U>) -> Result<()> {
        if self.len() != rhs.len() {
            return Err(ShapeError::Dimension {
                op: "vector arithmetic",
                l_rows: 1,
                l_cols: self.len(),
                r_rows: 1,
                r_cols: rhs.len(),
            }
            .into());
        }
        Ok(())
    }
}
impl<T: Clone> Vector<T> {
    /// Clone the buffer. Shape Requirements: none. Complexity: O(n).
    pub fn to_vec(&self) -> Vec<T> {
        self.data.clone()
    }
    /// Copy into a 1×n matrix. Shape Requirements: none. Complexity: O(n).
    pub fn to_row_matrix(&self) -> Matrix<T> {
        self.clone().into_row_matrix()
    }
    /// Copy into an n×1 matrix. Shape Requirements: none. Complexity: O(n).
    pub fn to_col_matrix(&self) -> Matrix<T> {
        self.clone().into_col_matrix()
    }
    /// Pairwise addition. Shape Requirements: equal lengths. Complexity: O(n).
    pub fn try_add<U: Clone>(&self, rhs: &Vector<U>) -> Result<Vector<<T as Add<U>>::Output>>
    where
        T: Add<U>,
    {
        self.same_len(rhs)?;
        Ok(Vector::new(
            self.data
                .iter()
                .zip(&rhs.data)
                .map(|(a, b)| a.clone() + b.clone())
                .collect(),
        ))
    }
    /// Pairwise subtraction. Shape Requirements: equal lengths. Complexity: O(n).
    pub fn try_sub<U: Clone>(&self, rhs: &Vector<U>) -> Result<Vector<<T as Sub<U>>::Output>>
    where
        T: Sub<U>,
    {
        self.same_len(rhs)?;
        Ok(Vector::new(
            self.data
                .iter()
                .zip(&rhs.data)
                .map(|(a, b)| a.clone() - b.clone())
                .collect(),
        ))
    }
    /// Scalar multiplication. Shape Requirements: none. Complexity: O(n).
    pub fn scalar_mul<S: Clone>(&self, scalar: S) -> Vector<<T as Mul<S>>::Output>
    where
        T: Mul<S>,
    {
        self.map(|x| x.clone() * scalar.clone())
    }
    /// Direct 1-D dot product. Shape Requirements: equal lengths; empty gives zero. Complexity: O(n).
    pub fn try_dot(&self, rhs: &Self) -> Result<T>
    where
        T: Zero + Add<Output = T> + Mul<Output = T>,
    {
        self.same_len(rhs)?;
        Ok(self
            .data
            .iter()
            .zip(&rhs.data)
            .fold(T::zero(), |sum, (a, b)| sum + a.clone() * b.clone()))
    }
    /// Direct dot product, panicking on length mismatch. Shape Requirements: equal lengths. Complexity: O(n).
    pub fn dot(&self, rhs: &Self) -> T
    where
        T: Zero + Add<Output = T> + Mul<Output = T>,
    {
        self.try_dot(rhs).expect("vector length mismatch")
    }
    /// Right-handed 3-D cross product. Shape Requirements: both lengths exactly 3. Complexity: O(1).
    pub fn cross(&self, rhs: &Self) -> Result<Self>
    where
        T: Mul<Output = T> + Sub<Output = T>,
    {
        self.same_len(rhs)?;
        if self.len() != 3 {
            return Err(ShapeError::Reshape {
                exp_rows: 1,
                exp_cols: 3,
                len: self.len(),
            }
            .into());
        }
        Ok(Self::new(vec![
            self[1].clone() * rhs[2].clone() - self[2].clone() * rhs[1].clone(),
            self[2].clone() * rhs[0].clone() - self[0].clone() * rhs[2].clone(),
            self[0].clone() * rhs[1].clone() - self[1].clone() * rhs[0].clone(),
        ]))
    }
}
impl<T: Real> Vector<T> {
    /// L1 norm. Shape Requirements: any length; IEEE nonfinite values propagate. Complexity: O(n).
    pub fn norm_l1(&self) -> T {
        if !self.data.iter().all(|x| x.is_finite()) {
            return self.data.iter().fold(T::zero(), |s, x| s + x.abs());
        }
        let scale = self.data.iter().fold(T::zero(), |s, x| s.max(x.abs()));
        if scale == T::zero() {
            return T::zero();
        }
        crate::numeric::compensated(self.data.iter().map(|x| x.abs() / scale)) * scale
    }
    /// Stable Euclidean norm. Shape Requirements: any length. Complexity: O(n).
    pub fn norm_l2(&self) -> T {
        self.data.iter().fold(T::zero(), |s, x| s.hypot(*x))
    }
    /// Infinity norm. Shape Requirements: any length, empty gives zero. Complexity: O(n).
    pub fn norm_inf(&self) -> T {
        if self.data.iter().any(|x| x.is_nan()) {
            return T::nan();
        }
        self.data.iter().fold(T::zero(), |s, x| s.max(x.abs()))
    }
    /// Normalize a finite nonzero vector. Shape Requirements: at least one nonzero element. Complexity: O(n).
    pub fn normalized(&self) -> Result<Self> {
        if self.data.iter().any(|x| !x.is_finite()) {
            return Err(NumericError::NonFinite.into());
        }
        let scale = self.norm_inf();
        if scale == T::zero() {
            return Err(NumericError::DivideByZero.into());
        }
        let norm = self.data.iter().fold(T::zero(), |s, x| s.hypot(*x / scale));
        Ok(self.map(|x| (*x / scale) / norm))
    }
}
impl<T> From<Vec<T>> for Vector<T> {
    fn from(data: Vec<T>) -> Self {
        Self::new(data)
    }
}
impl<T, const N: usize> From<[T; N]> for Vector<T> {
    fn from(data: [T; N]) -> Self {
        Self::new(Vec::from(data))
    }
}
impl<T> TryFrom<Matrix<T>> for Vector<T> {
    type Error = crate::MatrixError;
    fn try_from(m: Matrix<T>) -> Result<Self> {
        if m.rows() != 1 && m.cols() != 1 {
            return Err(ShapeError::NotVector.into());
        }
        Ok(Self::new(m.into_vec()))
    }
}
impl<T> Index<usize> for Vector<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}
impl<T> IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }
}
impl<T: Clone + Add<U>, U: Clone> Add<&Vector<U>> for &Vector<T> {
    type Output = Vector<T::Output>;
    fn add(self, rhs: &Vector<U>) -> Self::Output {
        self.try_add(rhs).expect("vector length mismatch")
    }
}
impl<T: Clone + Sub<U>, U: Clone> Sub<&Vector<U>> for &Vector<T> {
    type Output = Vector<T::Output>;
    fn sub(self, rhs: &Vector<U>) -> Self::Output {
        self.try_sub(rhs).expect("vector length mismatch")
    }
}
impl<T: Neg> Neg for Vector<T> {
    type Output = Vector<T::Output>;
    fn neg(self) -> Self::Output {
        Vector::new(self.data.into_iter().map(Neg::neg).collect())
    }
}
impl<T: fmt::Display> fmt::Display for Vector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector[{}] [", self.len())?;
        for (i, x) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{x}")?;
        }
        write!(f, "]")
    }
}
