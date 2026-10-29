use matrix_calc::{Matrix, Result, Tolerance};
fn main() -> Result<()> {
    let a = Matrix::from([[3f64, 1.], [1., 2.]]);
    let b = Matrix::from([[9.], [8.]]);
    let lu = a.lu()?;
    let x = lu.solve(&b)?;
    assert!(a.matmul(&x).approx_eq(&b, Tolerance::new(1e-12, 1e-12)?));
    println!("solution: {x}; determinant: {}", lu.determinant());
    let wide = Matrix::from([[1f64, 1.]]);
    let fit = wide.least_squares(&Matrix::from([[2.]]))?;
    println!(
        "minimum-norm solution: {}; rank: {}; residual: {}",
        fit.solution, fit.rank, fit.residual_norm
    );
    Ok(())
}
