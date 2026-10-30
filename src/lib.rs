#![doc = include_str!("../README.md")]
#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod matrix;
pub mod ops;

pub use error::MatrixError;
pub use matrix::Matrix;

pub mod structural;
pub mod view;
pub use error::{IndexError, NumericError, Result, ShapeError};
pub use view::{MatrixRead, MatrixView, MatrixViewMut};
pub mod numeric;
pub use numeric::{DecompositionOptions, Real, Tolerance};
pub mod linalg;
pub use linalg::{Cholesky, Lu};

pub use linalg::Qr;

pub use linalg::{LeastSquares, LeastSquaresMethod};
pub use linalg::{Svd, SymmetricEigen};

#[cfg(feature = "serde")]
mod serialization;

#[allow(unsafe_code)]
pub mod accelerated;

pub mod tensor;
pub mod vector;
pub use tensor::{Tensor, TensorError};
pub use vector::Vector;
pub mod repl;
