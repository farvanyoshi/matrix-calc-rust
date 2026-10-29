use matrix_calc::{Matrix, Result};
fn main() -> Result<()> {
    let mut matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
    matrix.slice_mut(0..2, 1..3)?.transpose().fill(9);
    assert_eq!(matrix, Matrix::from([[1, 9, 9], [4, 9, 9]]));
    println!("{}", matrix.view().transpose().to_owned());
    Ok(())
}
