use thiserror::Error;

use crate::vector::error::VectorError;
use crate::matrix::error::MatrixError;
use crate::tensor::error::TensorError;

#[derive(Error, Debug)]
#[error("There's a flaw in the Matrix.")] // ! This is a placeholder
pub enum MathError {
    VectorErr(#[from] VectorError),
    MatrixErr(#[from] MatrixError),
    TensorErr(#[from] TensorError)
}