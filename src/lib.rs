use std::ops::Add;

#[derive(Debug, Clone, PartialEq)]
struct Matrix<T> {
    rows: i32,
    cols: i32,
    data: Vec<T>
}

impl<T> Matrix<T> 
where T: Copy + Add<Output=T>
{
    fn add(&self, other: &Self) -> Self {
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

impl<T> Add<&Matrix<T>> for &Matrix<T>
    where T: Copy + Add<Output = T>
{
    type Output = Matrix<T>;

    fn add(self, rhs: &Matrix<T>) -> Self::Output {

    }
}