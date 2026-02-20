use crate::error;
use num_traits::{Zero, One};

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    pub rows: usize, 
    pub cols: usize,
    // ! The following line will likely never happen:
    // TODO: convert to TensorStorage<T> in the future
    pub data: Vec<T>
}

impl<T> Matrix<T> {
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    fn check_index(&self, row: usize, col: usize) -> Result<(), error::MatrixError> {
        match (row < self.rows, col < self.cols) {
            (true, true) => Ok(()),
            (false, false) => Err(error::IndexError::FullOOB { row, col, rows: self.rows, cols: self.cols }.into()),
            (false, _) => Err(error::IndexError::RowOOB { row, rows: self.rows }.into()),
            (_, false) => Err(error::IndexError::ColOOB { col, cols: self.cols }.into())
        }
    }

    pub fn try_get(&self, row: usize, col: usize) -> Result<&T, error::MatrixError> {
        self.check_index(row, col)?;
        Ok(&self.data[row * self.cols + col])
    }

    pub fn try_get_mut(&mut self, row: usize, col: usize) -> Result<&T, error::MatrixError> {
        self.check_index(row, col)?;
        Ok(&mut self.data[row * self.cols + col])
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows && col < self.cols {
            Some(&self.data[row * self.cols + col])
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.rows && col < self.cols {
            Some(&mut self.data[row * self.cols + col])
        } else {
            None
        }
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
    pub fn try_map<U>(&self, f: impl Fn(&T) -> Result<U, error::MatrixError>) 
    -> Result<Matrix<U>, error::MatrixError> {
        let mut data = Vec::with_capacity(self.data.len());
        for value in &self.data {
            data.push(f(value)?);
        }

        Ok(Matrix { rows: self.rows, cols: self.cols, data } )
    }

    pub fn map<U>(&self, f: impl Fn(&T) -> U) -> Matrix<U> {
        self.try_map(|x| Ok(f(x))).unwrap()
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