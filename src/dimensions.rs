use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dimensions {
    rows: usize,
    cols: usize
}

impl Dimensions {
    pub fn size(self) -> usize {
        self.rows * self.cols
    }

    pub fn is_square(self) -> bool {
        self.rows == self.cols
    }
}

impl fmt::Display for Dimensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.rows, self.cols)
    }
}