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
