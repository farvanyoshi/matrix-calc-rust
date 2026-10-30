//! Deterministic public-API fixture exporter. Shape requirements are explicit in each case.
use matrix_calc::{Matrix, Tensor, Vector};
use serde_json::{Value, json};
fn record(cases: &mut Vec<Value>, name: String, shape: &[usize], data: Vec<f64>) {
    cases.push(json!({"name":name,"shape":shape,"data":data}));
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = Vec::new();
    for m in 0..6 {
        for n in 0..6 {
            let a = Tensor::new((0..2 * m * n).map(|i| i as f64).collect(), vec![2, m, n])?;
            let b = Tensor::new((0..n).map(|i| (i + 1) as f64).collect(), vec![1, n])?;
            for (op, t) in [
                ("broadcast", a.try_add(&b)?),
                ("transpose", a.transpose(&[2, 0, 1])?),
                ("slice", a.slice(2, 0, n, 2)?),
            ] {
                record(&mut cases, format!("{op}_{m}_{n}"), t.shape(), t.to_vec());
            }
        }
    }
    let v = Vector::from([3., 4., 0.]);
    record(
        &mut cases,
        "vector_norms".into(),
        &[3],
        vec![v.dot(&v), v.norm_l1(), v.norm_l2()],
    );
    record(
        &mut cases,
        "cross".into(),
        &[3],
        Vector::from([1., 0., 0.])
            .cross(&Vector::from([0., 1., 0.]))?
            .to_vec(),
    );
    let a = Matrix::from([[0f64, 2., 3.], [4., 5., 6.], [7., 8., 10.]]);
    record(
        &mut cases,
        "inverse".into(),
        &[3, 3],
        a.inverse()?.into_vec(),
    );
    record(
        &mut cases,
        "determinant".into(),
        &[],
        vec![a.determinant()?],
    );
    record(
        &mut cases,
        "svd".into(),
        &[2],
        Matrix::from([[1f64, 2.], [2., 4.]])
            .svd()?
            .singular_values()
            .to_vec(),
    );
    record(
        &mut cases,
        "eigen".into(),
        &[2],
        Matrix::from([[2f64, 1.], [1., 2.]])
            .symmetric_eigen()?
            .eigenvalues()
            .to_vec(),
    );
    println!("{}", serde_json::to_string(&cases)?);
    Ok(())
}
