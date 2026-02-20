use thiserror::Error;

#[derive(Error, Debug)]
pub enum ShapeError {
    #[error("Cannot {} matrices of shapes {}x{} and {}x{}", .op, .l_rows, .l_cols, .r_rows, .r_cols)]
    Dimension { op: &'static str, l_rows: usize, l_cols: usize, r_rows: usize, r_cols: usize },
    #[error("Dimensions of {}x{} (total {}) cannot contain {} objects", .exp_rows, .exp_cols, .exp_rows * .exp_cols, .len)]
    Reshape { exp_rows: usize, exp_cols: usize, len: usize }
}

#[derive(Error, Debug)]
pub enum IndexError {
    #[error("{} out of bounds for matrix with {} rows", .row, .rows)]
    RowOOB {
        row: usize,
        rows: usize
    },
    #[error("{} out of bounds for matrix with {} columns", .col, .cols)]
    ColOOB {
        col: usize,
        cols: usize
    },
    #[error("{} and {} out of bounds for matrix of dimensions {}x{}", .row, .col, .rows, .cols)]
    FullOOB {
        row: usize,
        col: usize,
        rows: usize,
        cols: usize
    }
}

#[derive(Error, Debug)]
pub enum NumericError {
    #[error("Cannot divide by zero!")]
    DivideByZero
}

#[derive(Error, Debug)]
pub enum MatrixError {
    #[error(transparent)]
    ShapeErr(#[from] ShapeError),
    #[error(transparent)]
    IdxErr(#[from] IndexError),
    // IndexError { index: Dimensions, dims: Dimensions },
    // NotSquare { dims: Dimensions },
    // Singular { dims: Dimensions },
    #[error(transparent)]
    NumErr(#[from] NumericError)
}