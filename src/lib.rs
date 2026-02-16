use std::error::Error;
use std::fmt;
use std::ops;

#[derive(Debug, Clone, PartialEq)]
struct Dimensions {
    rows: usize,
    cols: usize
}

#[derive(Debug, Clone, PartialEq)]
struct Matrix<T> {
    dims: Dimensions,
    data: Vec<T>
}

pub enum MatrixOperation {
    Addition,
    Subtraction,
    Multiplication,
    Division,
    Hadamard
}

// while this seems redundant, it is to allow cleaner future designs with matrix errors.
pub enum MatrixError {
    Shape(ShapeError)
    // IndexError { index: Dimensions, dims: Dimensions },
    // NotSquare { dims: Dimensions },
    // Singular { dims: Dimensions },
}

pub enum ShapeError {
    DimensionError { op: MatrixOperation, lhs: Dimensions, rhs: Dimensions }
}



// impl fmt::Display for MatrixError {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         match self {
//             MatrixOperation::Addition => write!(
//                 f,
//                 "Cannot add matrices of dimensions ({}x{}) and ({}x{}): \
//                 dimensions must be equal",
//                 self.lhs.0, self.lhs.1, self.rhs.0, self.rhs.1
//             ),
//             MatrixOperation::Multiplication => write!(
//                 f,
//                 "Cannot multply matrices of dimensions ({}x{}) and ({}x{}): \
//                 left columns ({}) must equal right rows ({})",
//                 self.lhs.0, self.lhs.1, self.rhs.0, self.rhs.1,
//                 self.lhs.1, self.rhs.0
//             )
//         }
//     }
// }

impl Error for DimensionError {}

impl<T> Matrix<T> 
where T: Copy + ops::Add<Output=T>
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

impl<T> ops::Add<&Matrix<T>> for &Matrix<T>
    where T: Copy + ops::Add<Output = T>
{
    type Output = Matrix<T>;

    fn add(self, rhs: &Matrix<T>) -> Self::Output {

    }
}