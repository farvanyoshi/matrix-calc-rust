//! Structured structural and numerical errors.
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
/// Invalid dimensions, element counts, or structural requirements.
pub enum ShapeError {
    /// Matrix dimensions {rows}x{cols} overflow.
    #[error("matrix dimensions {rows}x{cols} overflow")]
    Overflow {
        /// Logical row count.
        rows: usize,
        /// Logical column count.
        cols: usize,
    },
    /// Cannot {op} matrices of shapes {l_rows}x{l_cols} and {r_rows}x{r_cols}.
    #[error("cannot {op} matrices of shapes {l_rows}x{l_cols} and {r_rows}x{r_cols}")]
    Dimension {
        /// Operation being validated.
        op: &'static str,
        /// Left operand row count.
        l_rows: usize,
        /// Left operand column count.
        l_cols: usize,
        /// Right operand row count.
        r_rows: usize,
        /// Right operand column count.
        r_cols: usize,
    },
    /// Dimensions {exp_rows}x{exp_cols} cannot contain {len} elements.
    #[error("dimensions {exp_rows}x{exp_cols} cannot contain {len} elements")]
    Reshape {
        /// Requested output row count.
        exp_rows: usize,
        /// Requested output column count.
        exp_cols: usize,
        /// Observed input element count.
        len: usize,
    },
    /// Operation requires a square matrix, got {rows}x{cols}.
    #[error("operation requires a square matrix, got {rows}x{cols}")]
    NotSquare {
        /// Logical row count.
        rows: usize,
        /// Logical column count.
        cols: usize,
    },
    /// Operation requires a row or column vector.
    #[error("operation requires a row or column vector")]
    NotVector,
    /// Ragged nested matrix input.
    #[error("ragged nested matrix input")]
    Ragged,
}
#[derive(Error, Debug, Clone, PartialEq, Eq)]
/// Invalid logical coordinates or slice ranges.
pub enum IndexError {
    /// Row {row} is out of bounds for {rows} rows.
    #[error("row {row} is out of bounds for {rows} rows")]
    RowOOB {
        /// Requested row index.
        row: usize,
        /// Logical row count.
        rows: usize,
    },
    /// Column {col} is out of bounds for {cols} columns.
    #[error("column {col} is out of bounds for {cols} columns")]
    ColOOB {
        /// Requested column index.
        col: usize,
        /// Logical column count.
        cols: usize,
    },
    /// Coordinate ({row}, {col}) is out of bounds for {rows}x{cols}.
    #[error("coordinate ({row}, {col}) is out of bounds for {rows}x{cols}")]
    FullOOB {
        /// Requested row index.
        row: usize,
        /// Requested column index.
        col: usize,
        /// Logical row count.
        rows: usize,
        /// Logical column count.
        cols: usize,
    },
    /// Invalid matrix slice range.
    #[error("invalid matrix slice range")]
    Range,
}
#[derive(Error, Debug, Clone, PartialEq, Eq)]
/// Invalid or failed arithmetic and numerical operations.
pub enum NumericError {
    /// Cannot divide by zero.
    #[error("cannot divide by zero")]
    DivideByZero,
    /// Scalar arithmetic overflow.
    #[error("scalar arithmetic overflow")]
    Overflow,
    /// Numeric conversion is out of range or loses information.
    #[error("numeric conversion is out of range or loses information")]
    Conversion,
    /// Reduction requires at least one element.
    #[error("reduction requires at least one element")]
    Empty,
    /// Input or computed result is nonfinite.
    #[error("input or computed result is nonfinite")]
    NonFinite,
    /// Invalid tolerance or iteration limit.
    #[error("invalid tolerance or iteration limit")]
    InvalidOptions,
    /// Matrix is singular at the requested tolerance.
    #[error("matrix is singular at the requested tolerance")]
    Singular,
    /// Matrix is rank deficient at the requested tolerance.
    #[error("matrix is rank deficient at the requested tolerance")]
    RankDeficient,
    /// Matrix is not symmetric at the requested tolerance.
    #[error("matrix is not symmetric at the requested tolerance")]
    NotSymmetric,
    /// Matrix is not positive definite at the requested tolerance.
    #[error("matrix is not positive definite at the requested tolerance")]
    NotPositiveDefinite,
    /// Iteration did not converge within {iterations} sweeps.
    #[error("iteration did not converge within {iterations} sweeps")]
    NonConvergence {
        /// Maximum number of complete sweeps.
        iterations: usize,
    },
}
#[derive(Error, Debug)]
/// Unified structural, index, numerical, and allocation error.
pub enum MatrixError {
    /// Shape validation failed.
    #[error(transparent)]
    ShapeErr(#[from] ShapeError),
    /// Coordinate or range validation failed.
    #[error(transparent)]
    IdxErr(#[from] IndexError),
    /// Numerical validation or computation failed.
    #[error(transparent)]
    NumErr(#[from] NumericError),
    /// Matrix allocation/capacity failure.
    #[error("matrix allocation/capacity failure")]
    Allocation(#[from] std::collections::TryReserveError),
}
/// Library-wide result type for structural and numerical operations.
pub type Result<T> = std::result::Result<T, MatrixError>;
