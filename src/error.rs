use crate::dimensions::Dimensions;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ShapeError {
    #[error("Cannot {} matrices of shapes {} and {}", .op, .lhs, .rhs)]
    Dimension { op: &'static str, lhs: Dimensions, rhs: Dimensions },
    #[error("Dimensions of {} (total {}) cannot contain {} objects", .expected, .expected.size(), .len)]
    Reshape { expected: Dimensions, len: usize }
}

#[derive(Error, Debug)]
pub enum NumericError {
    #[error("Cannot divide by zero!")]
    DivideByZero
}

#[derive(Error, Debug)]
pub enum MatrixError {
    #[error(transparent)]
    Shape(#[from] ShapeError),
    // IndexError { index: Dimensions, dims: Dimensions },
    // NotSquare { dims: Dimensions },
    // Singular { dims: Dimensions },
    #[error(transparent)]
    Numeric(#[from] NumericError)
}