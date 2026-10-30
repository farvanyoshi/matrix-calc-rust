//! General n-dimensional tensors, right-aligned broadcasting and safe COW views.
use crate::{Matrix, Vector};
use num_traits::Zero;
use std::{
    fmt,
    ops::{Add, Div, Index, Mul, Sub},
    sync::Arc,
};
use thiserror::Error;

/// Tensor shape, coordinate, broadcasting and arithmetic validation errors.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TensorError {
    /// Requested output allocation or capacity cannot be reserved.
    #[error("tensor allocation/capacity failure")]
    Allocation,
    /// Element count, shape, or stride arithmetic overflow.
    #[error("tensor shape/stride arithmetic overflow")]
    Overflow,
    /// Logical shape and buffer length disagree.
    #[error("tensor shape {shape:?} does not match {len} elements")]
    Shape {
        /// Requested logical dimensions.
        shape: Vec<usize>,
        /// Observed storage length.
        len: usize,
    },
    /// Index rank or coordinate is invalid.
    #[error("invalid tensor coordinate {index:?} for shape {shape:?}")]
    Index {
        /// Requested indices.
        index: Vec<usize>,
        /// Logical dimensions.
        shape: Vec<usize>,
    },
    /// The requested axes are not a permutation of all dimensions.
    #[error("tensor transpose axes must be a permutation of all dimensions")]
    Axes,
    /// The half-open slice or its positive step is invalid.
    #[error("invalid tensor slice axis/range/step")]
    Slice,
    /// Broadcasting dimensions disagree and neither equals one.
    #[error("cannot broadcast {0:?} with {1:?}")]
    Broadcast(Vec<usize>, Vec<usize>),
    /// Operation requires a particular rank.
    #[error("operation requires rank {expected}, got {actual}")]
    Rank {
        /// Required rank.
        expected: usize,
        /// Observed rank.
        actual: usize,
    },
    /// Reshaping a noncontiguous view requires an explicit contiguous copy.
    #[error("reshape requires contiguous storage; call make_contiguous first")]
    NonContiguous,
    /// At least one divisor is zero.
    #[error("tensor division by zero")]
    DivideByZero,
}
fn layout(shape: &[usize]) -> Result<(usize, Vec<usize>), TensorError> {
    if shape.contains(&0) {
        return Ok((0, vec![0; shape.len()]));
    }
    let mut strides = vec![0; shape.len()];
    let mut n = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = n;
        n = n.checked_mul(shape[i]).ok_or(TensorError::Overflow)?;
    }
    Ok((n, strides))
}
/// An n-D tensor with retained, positive-stride, copy-on-write backing storage.
/// Clones/transposes/slices share data until mutation detaches a private backing
/// buffer. Mutable views consequently never modify their source tensor. Rank-zero
/// tensors hold one scalar; tensors with any zero axis hold no elements.
#[must_use]
#[derive(Debug)]
pub struct Tensor<T> {
    data: Arc<Vec<T>>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    len: usize,
}
impl<T> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            len: self.len,
        }
    }
}
impl<T> Tensor<T> {
    /// Take a row-major buffer. Shape Requirements: product(shape)==data.len(); [] requires one scalar. Complexity: O(rank).
    pub fn new(data: Vec<T>, shape: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        let shape = shape.into();
        let (len, strides) = layout(&shape)?;
        if len != data.len() {
            return Err(TensorError::Shape {
                shape,
                len: data.len(),
            });
        }
        Ok(Self {
            data: Arc::new(data),
            shape,
            strides,
            offset: 0,
            len,
        })
    }
    /// Construct a scalar. Shape Requirements: rank zero. Complexity: O(1).
    pub fn scalar(value: T) -> Self {
        Self::new(vec![value], vec![]).expect("valid scalar")
    }
    /// Logical dimensions. Shape Requirements: none. Complexity: O(1).
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Element strides; empty tensors have canonical zero strides. Shape Requirements: none. Complexity: O(1).
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    /// Number of dimensions. Shape Requirements: none. Complexity: O(1).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
    /// Logical element count. Shape Requirements: none. Complexity: O(1).
    pub fn len(&self) -> usize {
        self.len
    }
    /// Whether logical storage is empty. Shape Requirements: none. Complexity: O(1).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Whether logical iteration follows contiguous row-major memory. Shape Requirements: none. Complexity: O(rank).
    pub fn is_contiguous(&self) -> bool {
        if self.len == 0 {
            return true;
        }
        let mut stride = 1;
        for i in (0..self.rank()).rev() {
            if self.shape[i] > 1 && self.strides[i] != stride {
                return false;
            }
            stride *= self.shape[i];
        }
        true
    }
    /// Borrow contiguous logical storage, or None for a strided view. Shape Requirements: none. Complexity: O(rank).
    pub fn as_slice(&self) -> Option<&[T]> {
        if self.is_contiguous() {
            Some(&self.data[self.offset..self.offset + self.len])
        } else {
            None
        }
    }
    fn physical(&self, index: &[usize]) -> Option<usize> {
        if index.len() != self.rank() {
            return None;
        }
        let mut i = self.offset;
        for ((coordinate, size), stride) in index.iter().zip(&self.shape).zip(&self.strides) {
            if coordinate >= size {
                return None;
            }
            i = i.checked_add(coordinate.checked_mul(*stride)?)?;
        }
        Some(i)
    }
    /// Borrow a logical element. Shape Requirements: one in-bounds index per axis. Complexity: O(rank).
    pub fn get(&self, index: &[usize]) -> Option<&T> {
        self.physical(index).and_then(|i| self.data.get(i))
    }
    fn flat_offset(&self, mut flat: usize) -> usize {
        let mut i = self.offset;
        for axis in (0..self.rank()).rev() {
            i += (flat % self.shape[axis]) * self.strides[axis];
            flat /= self.shape[axis];
        }
        i
    }
    /// Iterate references in logical row-major order. Shape Requirements: none. Complexity: O(n*rank).
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &T> + DoubleEndedIterator {
        let contiguous = self.is_contiguous();
        (0..self.len).map(move |i| {
            &self.data[if contiguous {
                self.offset + i
            } else {
                self.flat_offset(i)
            }]
        })
    }
    /// Reorder axes without copying elements. Shape Requirements: axes is a complete permutation. Complexity: O(rank).
    pub fn transpose(&self, axes: &[usize]) -> Result<Self, TensorError> {
        if axes.len() != self.rank() {
            return Err(TensorError::Axes);
        }
        let mut seen = vec![false; self.rank()];
        for &axis in axes {
            if axis >= self.rank() || seen[axis] {
                return Err(TensorError::Axes);
            }
            seen[axis] = true;
        }
        Ok(Self {
            data: Arc::clone(&self.data),
            shape: axes.iter().map(|&a| self.shape[a]).collect(),
            strides: axes.iter().map(|&a| self.strides[a]).collect(),
            offset: self.offset,
            len: self.len,
        })
    }
    /// Reverse all axes. Shape Requirements: none. Complexity: O(rank).
    pub fn reversed_axes(&self) -> Self {
        self.transpose(&(0..self.rank()).rev().collect::<Vec<_>>())
            .expect("valid axes")
    }
    /// Positive-step half-open view. Shape Requirements: axis<rank, start<=stop<=axis length, step>0. Complexity: O(rank).
    pub fn slice(
        &self,
        axis: usize,
        start: usize,
        stop: usize,
        step: usize,
    ) -> Result<Self, TensorError> {
        if axis >= self.rank() || step == 0 || start > stop || stop > self.shape[axis] {
            return Err(TensorError::Slice);
        }
        let mut out = self.clone();
        let distance = stop - start;
        out.shape[axis] = distance / step + usize::from(distance % step != 0);
        out.len = layout(&out.shape)?.0;
        if out.len == 0 {
            out.offset = 0;
            out.strides = vec![0; out.rank()];
        } else {
            out.offset = out
                .offset
                .checked_add(
                    start
                        .checked_mul(out.strides[axis])
                        .ok_or(TensorError::Overflow)?,
                )
                .ok_or(TensorError::Overflow)?;
            out.strides[axis] = out.strides[axis]
                .checked_mul(step)
                .ok_or(TensorError::Overflow)?;
        }
        Ok(out)
    }
    /// Reinterpret contiguous logical order without copying. Shape Requirements: same element count and contiguous input. Complexity: O(rank).
    pub fn reshape(&self, shape: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        let shape = shape.into();
        let (len, strides) = layout(&shape)?;
        if len != self.len {
            return Err(TensorError::Shape {
                shape,
                len: self.len,
            });
        }
        if !self.is_contiguous() {
            return Err(TensorError::NonContiguous);
        }
        Ok(Self {
            data: Arc::clone(&self.data),
            shape,
            strides,
            offset: self.offset,
            len,
        })
    }
    /// Right-aligned broadcast result shape. Shape Requirements: corresponding axes equal or one is 1. Complexity: O(max rank).
    pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, TensorError> {
        let rank = a.len().max(b.len());
        let mut out = vec![1; rank];
        for i in 0..rank {
            let x = if i >= rank - a.len() {
                a[i - (rank - a.len())]
            } else {
                1
            };
            let y = if i >= rank - b.len() {
                b[i - (rank - b.len())]
            } else {
                1
            };
            out[i] = if x == y {
                x
            } else if x == 1 {
                y
            } else if y == 1 {
                x
            } else {
                return Err(TensorError::Broadcast(a.to_vec(), b.to_vec()));
            };
        }
        layout(&out)?;
        Ok(out)
    }
    fn broadcast_offset(&self, mut flat: usize, out_shape: &[usize]) -> usize {
        let mut offset = self.offset;
        let leading = out_shape.len() - self.rank();
        for i in (0..out_shape.len()).rev() {
            let coordinate = flat % out_shape[i];
            flat /= out_shape[i];
            if i >= leading && self.shape[i - leading] != 1 {
                offset += coordinate * self.strides[i - leading];
            }
        }
        offset
    }
    /// Allocate a shape-preserving callback map. Shape Requirements: none. Complexity: O(n*rank).
    pub fn map<U>(&self, f: impl FnMut(&T) -> U) -> Tensor<U> {
        Tensor::new(self.iter().map(f).collect(), self.shape.clone())
            .expect("validated source shape")
    }
    /// Broadcast two tensors into callback results. Shape Requirements: broadcast-compatible axes. Complexity: O(output elements*rank).
    pub fn try_zip_map<U, R>(
        &self,
        rhs: &Tensor<U>,
        mut f: impl FnMut(&T, &U) -> R,
    ) -> Result<Tensor<R>, TensorError> {
        let shape = Self::broadcast_shape(&self.shape, &rhs.shape)?;
        let n = layout(&shape)?.0;
        let mut values = Vec::new();
        values
            .try_reserve_exact(n)
            .map_err(|_| TensorError::Allocation)?;
        if self.shape == rhs.shape {
            if let (Some(a), Some(b)) = (self.as_slice(), rhs.as_slice()) {
                values.extend(a.iter().zip(b).map(|(x, y)| f(x, y)));
                return Tensor::new(values, shape);
            }
        }
        for i in 0..n {
            values.push(f(
                &self.data[self.broadcast_offset(i, &shape)],
                &rhs.data[rhs.broadcast_offset(i, &shape)],
            ));
        }
        Tensor::new(values, shape)
    }
}
impl<T: Clone> Tensor<T> {
    /// Copy logical row-major elements. Shape Requirements: none. Complexity: O(n*rank).
    pub fn to_vec(&self) -> Vec<T> {
        self.iter().cloned().collect()
    }
    /// Return a contiguous handle, copying a noncontiguous view. Shape Requirements: none. Complexity: O(rank) or O(n*rank).
    pub fn make_contiguous(&self) -> Self {
        if self.is_contiguous() {
            self.clone()
        } else {
            Self::new(self.to_vec(), self.shape.clone()).expect("valid shape")
        }
    }
    /// COW element mutation; shared views detach rather than mutate other handles. Shape Requirements: in-bounds coordinate per axis. Complexity: O(rank), plus O(backing length) if shared.
    pub fn set(&mut self, index: &[usize], value: T) -> Result<(), TensorError> {
        let i = self.physical(index).ok_or_else(|| TensorError::Index {
            index: index.to_vec(),
            shape: self.shape.clone(),
        })?;
        Arc::make_mut(&mut self.data)[i] = value;
        Ok(())
    }
    /// Clone into a matrix. Shape Requirements: exactly rank 2. Complexity: O(n*rank).
    pub fn to_matrix(&self) -> Result<Matrix<T>, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::Rank {
                expected: 2,
                actual: self.rank(),
            });
        }
        Ok(Matrix::new(self.to_vec(), self.shape[0], self.shape[1]))
    }
    /// Clone into a vector. Shape Requirements: exactly rank 1. Complexity: O(n).
    pub fn to_vector(&self) -> Result<Vector<T>, TensorError> {
        if self.rank() != 1 {
            return Err(TensorError::Rank {
                expected: 1,
                actual: self.rank(),
            });
        }
        Ok(Vector::new(self.to_vec()))
    }
    /// Broadcast addition. Shape Requirements: equal or singleton corresponding axes. Complexity: O(output elements*rank).
    pub fn try_add<U: Clone>(
        &self,
        rhs: &Tensor<U>,
    ) -> Result<Tensor<<T as Add<U>>::Output>, TensorError>
    where
        T: Add<U>,
    {
        self.try_zip_map(rhs, |a, b| a.clone() + b.clone())
    }
    /// Broadcast subtraction. Shape Requirements: equal or singleton corresponding axes. Complexity: O(output elements*rank).
    pub fn try_sub<U: Clone>(
        &self,
        rhs: &Tensor<U>,
    ) -> Result<Tensor<<T as Sub<U>>::Output>, TensorError>
    where
        T: Sub<U>,
    {
        self.try_zip_map(rhs, |a, b| a.clone() - b.clone())
    }
    /// Broadcast elementwise multiplication. Shape Requirements: equal or singleton axes. Complexity: O(output elements*rank).
    pub fn try_hadamard<U: Clone>(
        &self,
        rhs: &Tensor<U>,
    ) -> Result<Tensor<<T as Mul<U>>::Output>, TensorError>
    where
        T: Mul<U>,
    {
        self.try_zip_map(rhs, |a, b| a.clone() * b.clone())
    }
    /// Broadcast elementwise division, validating shape before rejecting zero divisors. Shape Requirements: broadcast-compatible axes. Complexity: O(output elements*rank).
    pub fn try_element_div<U: Clone + Zero>(
        &self,
        rhs: &Tensor<U>,
    ) -> Result<Tensor<<T as Div<U>>::Output>, TensorError>
    where
        T: Div<U>,
    {
        Self::broadcast_shape(&self.shape, &rhs.shape)?;
        if rhs.iter().any(Zero::is_zero) {
            return Err(TensorError::DivideByZero);
        }
        self.try_zip_map(rhs, |a, b| a.clone() / b.clone())
    }
    /// Allocate scalar addition. Shape Requirements: none. Complexity: O(n*rank).
    pub fn scalar_add<S: Clone>(&self, value: S) -> Tensor<<T as Add<S>>::Output>
    where
        T: Add<S>,
    {
        self.map(|x| x.clone() + value.clone())
    }
    /// Allocate scalar subtraction. Shape Requirements: none. Complexity: O(n*rank).
    pub fn scalar_sub<S: Clone>(&self, value: S) -> Tensor<<T as Sub<S>>::Output>
    where
        T: Sub<S>,
    {
        self.map(|x| x.clone() - value.clone())
    }
    /// Allocate scalar multiplication. Shape Requirements: none. Complexity: O(n*rank).
    pub fn scalar_mul<S: Clone>(&self, value: S) -> Tensor<<T as Mul<S>>::Output>
    where
        T: Mul<S>,
    {
        self.map(|x| x.clone() * value.clone())
    }
    /// Allocate scalar division, rejecting zero. Shape Requirements: none. Complexity: O(n*rank).
    pub fn scalar_div<S: Clone + Zero>(
        &self,
        value: S,
    ) -> Result<Tensor<<T as Div<S>>::Output>, TensorError>
    where
        T: Div<S>,
    {
        if value.is_zero() {
            return Err(TensorError::DivideByZero);
        }
        Ok(self.map(|x| x.clone() / value.clone()))
    }
}
impl Tensor<f64> {
    /// Deterministic uniform [0,1) samples using SplitMix64. Shape Requirements: valid element count. Complexity: O(n+rank).
    pub fn random(shape: impl Into<Vec<usize>>, seed: u64) -> Result<Self, TensorError> {
        let shape = shape.into();
        let n = layout(&shape)?.0;
        let mut state = seed;
        let mut data = Vec::new();
        data.try_reserve_exact(n)
            .map_err(|_| TensorError::Allocation)?;
        for _ in 0..n {
            let value = {
                state = state.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                z ^= z >> 31;
                (z >> 11) as f64 * (1.0 / 9007199254740992.0)
            };
            data.push(value);
        }
        Self::new(data, shape)
    }
}
impl<T: PartialEq> PartialEq for Tensor<T> {
    fn eq(&self, rhs: &Self) -> bool {
        self.shape == rhs.shape && self.iter().eq(rhs.iter())
    }
}
impl<T: Eq> Eq for Tensor<T> {}
impl<T> From<Matrix<T>> for Tensor<T> {
    fn from(m: Matrix<T>) -> Self {
        let shape = vec![m.rows(), m.cols()];
        Self::new(m.into_vec(), shape).expect("valid matrix shape")
    }
}
impl<T: Clone> From<&Matrix<T>> for Tensor<T> {
    fn from(m: &Matrix<T>) -> Self {
        Self::new(m.data().to_vec(), vec![m.rows(), m.cols()]).expect("valid matrix shape")
    }
}
impl<T> From<Vector<T>> for Tensor<T> {
    fn from(v: Vector<T>) -> Self {
        let n = v.len();
        Self::new(v.into_vec(), vec![n]).expect("valid vector shape")
    }
}
impl<T: Clone> TryFrom<Tensor<T>> for Matrix<T> {
    type Error = TensorError;
    fn try_from(t: Tensor<T>) -> Result<Self, Self::Error> {
        t.to_matrix()
    }
}
impl<T: Clone> TryFrom<Tensor<T>> for Vector<T> {
    type Error = TensorError;
    fn try_from(t: Tensor<T>) -> Result<Self, Self::Error> {
        t.to_vector()
    }
}
impl<T> Matrix<T> {
    /// Consume into a rank-two tensor. Shape Requirements: none. Complexity: O(1).
    pub fn into_tensor(self) -> Tensor<T> {
        self.into()
    }
}
impl<T: Clone> Matrix<T> {
    /// Clone into a rank-two tensor. Shape Requirements: none. Complexity: O(n).
    pub fn to_tensor(&self) -> Tensor<T> {
        self.into()
    }
}
impl<T> Vector<T> {
    /// Consume into a rank-one tensor. Shape Requirements: none. Complexity: O(1).
    pub fn into_tensor(self) -> Tensor<T> {
        self.into()
    }
}
impl<T: Clone> Vector<T> {
    /// Clone into a rank-one tensor. Shape Requirements: none. Complexity: O(n).
    pub fn to_tensor(&self) -> Tensor<T> {
        Tensor::new(self.to_vec(), vec![self.len()]).expect("valid vector shape")
    }
}
impl<T> Index<&[usize]> for Tensor<T> {
    type Output = T;
    fn index(&self, index: &[usize]) -> &T {
        self.get(index).expect("tensor coordinate out of bounds")
    }
}
impl<T, const N: usize> Index<[usize; N]> for Tensor<T> {
    type Output = T;
    fn index(&self, index: [usize; N]) -> &T {
        self.get(&index).expect("tensor coordinate out of bounds")
    }
}
impl<T: Clone + Add<U>, U: Clone> Add<&Tensor<U>> for &Tensor<T> {
    type Output = Tensor<T::Output>;
    fn add(self, rhs: &Tensor<U>) -> Self::Output {
        self.try_add(rhs).expect("incompatible broadcast shapes")
    }
}
impl<T: Clone + Sub<U>, U: Clone> Sub<&Tensor<U>> for &Tensor<T> {
    type Output = Tensor<T::Output>;
    fn sub(self, rhs: &Tensor<U>) -> Self::Output {
        self.try_sub(rhs).expect("incompatible broadcast shapes")
    }
}
impl<T: Clone + Mul<U>, U: Clone> Mul<&Tensor<U>> for &Tensor<T> {
    type Output = Tensor<T::Output>;
    fn mul(self, rhs: &Tensor<U>) -> Self::Output {
        self.try_hadamard(rhs)
            .expect("incompatible broadcast shapes")
    }
}
impl<T: Clone + Div<U>, U: Clone + Zero> Div<&Tensor<U>> for &Tensor<T> {
    type Output = Tensor<T::Output>;
    fn div(self, rhs: &Tensor<U>) -> Self::Output {
        self.try_element_div(rhs).expect("invalid tensor division")
    }
}
impl<T: fmt::Display> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn dimension<T: fmt::Display>(
            t: &Tensor<T>,
            f: &mut fmt::Formatter<'_>,
            index: &mut Vec<usize>,
            axis: usize,
        ) -> fmt::Result {
            if axis == t.rank() {
                return write!(f, "{}", t.get(index).expect("valid display index"));
            }
            write!(f, "[")?;
            for i in 0..t.shape[axis] {
                if i > 0 {
                    write!(f, ", ")?;
                }
                index.push(i);
                dimension(t, f, index, axis + 1)?;
                index.pop();
            }
            write!(f, "]")
        }
        write!(f, "Tensor{:?} ", self.shape)?;
        if self.is_empty() {
            return write!(f, "[]");
        }
        if self.rank() > 8 || self.len() > 128 {
            write!(f, "[")?;
            for (i, value) in self.iter().take(64).enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{value}")?;
            }
            if self.len() > 64 {
                write!(f, ", …")?;
            }
            return write!(f, "]");
        }
        dimension(self, f, &mut Vec::new(), 0)
    }
}
