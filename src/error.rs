//! Structured structural and numerical errors.
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    #[error("matrix dimensions {rows}x{cols} overflow")]
    Overflow { rows: usize, cols: usize },
    #[error("cannot {op} matrices of shapes {l_rows}x{l_cols} and {r_rows}x{r_cols}")]
    Dimension {
        op: &'static str,
        l_rows: usize,
        l_cols: usize,
        r_rows: usize,
        r_cols: usize,
    },
    #[error("dimensions {exp_rows}x{exp_cols} cannot contain {len} elements")]
    Reshape {
        exp_rows: usize,
        exp_cols: usize,
        len: usize,
    },
    #[error("operation requires a square matrix, got {rows}x{cols}")]
    NotSquare { rows: usize, cols: usize },
    #[error("operation requires a row or column vector")]
    NotVector,
    #[error("ragged nested matrix input")]
    Ragged,
}
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum IndexError {
    #[error("row {row} is out of bounds for {rows} rows")]
    RowOOB { row: usize, rows: usize },
    #[error("column {col} is out of bounds for {cols} columns")]
    ColOOB { col: usize, cols: usize },
    #[error("coordinate ({row}, {col}) is out of bounds for {rows}x{cols}")]
    FullOOB {
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    },
    #[error("invalid matrix slice range")]
    Range,
}
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum NumericError {
    #[error("cannot divide by zero")]
    DivideByZero,
    #[error("scalar arithmetic overflow")]
    Overflow,
    #[error("numeric conversion is out of range or loses information")]
    Conversion,
    #[error("reduction requires at least one element")]
    Empty,
    #[error("input or computed result is nonfinite")]
    NonFinite,
    #[error("invalid tolerance or iteration limit")]
    InvalidOptions,
    #[error("matrix is singular at the requested tolerance")]
    Singular,
    #[error("matrix is rank deficient at the requested tolerance")]
    RankDeficient,
    #[error("matrix is not symmetric at the requested tolerance")]
    NotSymmetric,
    #[error("matrix is not positive definite at the requested tolerance")]
    NotPositiveDefinite,
    #[error("iteration did not converge within {iterations} sweeps")]
    NonConvergence { iterations: usize },
}
#[derive(Error, Debug)]
pub enum MatrixError {
    #[error(transparent)]
    ShapeErr(#[from] ShapeError),
    #[error(transparent)]
    IdxErr(#[from] IndexError),
    #[error(transparent)]
    NumErr(#[from] NumericError),
    #[error("matrix allocation/capacity failure")]
    Allocation(#[from] std::collections::TryReserveError),
}
/// Library-wide result type for structural and numerical operations.
pub type Result<T> = std::result::Result<T, MatrixError>;
