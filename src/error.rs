use crate::dimensions::Dimensions;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ShapeError {
    #[error("Cannot {} for matrices of shapes {} and {}", .op, .lhs, .rhs)]
    Dimension { op: &'static str, lhs: Dimensions, rhs: Dimensions },
    // include other shape errors here
}

#[derive(Error, Debug)]
pub enum MatrixError {
    #[error(transparent)]
    Shape(#[from] ShapeError),
    // IndexError { index: Dimensions, dims: Dimensions },
    // NotSquare { dims: Dimensions },
    // Singular { dims: Dimensions },
}