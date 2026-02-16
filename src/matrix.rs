use crate::dimensions::Dimensions;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    dims: Dimensions,
    data: Vec<T>
}

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