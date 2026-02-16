use std::error::Error;
use std::fmt;
use std::ops::Add;

#[derive(Debug, Clone, PartialEq)]
struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>
}

#[derive(Debug, Clone)]
pub enum DimensionError {
    Addition {
        lhs: (usize, usize),
        rhs: (usize, usize)
    },
    Multiplication {
        lhs: (usize, usize),
        rhs: (usize, usize)
    }
}

impl fmt::Display for DimensionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimensionError::Addition {lhs, rhs} => write!(
                f,
                "Cannot add matrices of different dimensions ({} by {} and {} by {})",
                lhs.0, lhs.1, rhs.0, rhs.1
            ),
            DimensionError::Multiplication {lhs, rhs} => write!(
                f,
                "Cannot multply matrices of dimensions ({}x{}) and ({}x{}): \
                left columns ({}) must equal right rows ({})",
                lhs.0, lhs.1, rhs.0, rhs.1,
                lhs.1, rhs.0
            )
        }
    }
}

impl Error for DimensionError {}

impl<T> Matrix<T> 
where T: Copy + Add<Output=T>
{
    fn add(&self, other: &Self) -> Result<Self, DimensionError> {
        if self.rows != other.rows || self.cols != other.cols {
            
        }
        let mut data = Vec::with_capacity(self.data.len());
        for i in 0..self.data.len() {
            data[i] = self.data[i] + other.data[i]
        }
        Matrix{
            rows: self.rows,
            cols: self.cols,
            data
        }
    }
}

impl<T> Add<&Matrix<T>> for &Matrix<T>
    where T: Copy + Add<Output = T>
{
    type Output = Matrix<T>;

    fn add(self, rhs: &Matrix<T>) -> Self::Output {

    }
}