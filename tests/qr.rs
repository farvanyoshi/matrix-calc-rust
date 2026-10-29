use matrix_calc::{Matrix, Tolerance};
fn close(a: &Matrix<f64>, b: &Matrix<f64>) {
    assert!(
        a.approx_eq(b, Tolerance::new(1e-11, 1e-11).unwrap()),
        "{a}\n!=\n{b}"
    );
}
#[test]
fn reflectors_pivoting_and_rectangular_shapes() {
    for a in [
        Matrix::from([[1., 2.], [3., 4.], [5., 7.]]),
        Matrix::from([[1., 2., 3.], [4., 5., 7.]]),
        Matrix::zeros(0, 3),
        Matrix::zeros(3, 0),
    ] {
        for pivot in [false, true] {
            let qr = a.qr_with_options(pivot, Default::default()).unwrap();
            let q = qr.q();
            let r = qr.r();
            close(&a.matmul(&qr.permutation_matrix()), &q.matmul(&r));
            close(
                &q.transpose().matmul(&q),
                &Matrix::identity(a.rows().min(a.cols())),
            );
            let b = Matrix::from_fn(a.rows(), 2, |r, c| (r + c) as f64);
            close(&qr.apply_q(&qr.apply_q_transpose(&b).unwrap()).unwrap(), &b);
        }
    }
}
#[test]
fn qr_least_squares() {
    let a = Matrix::from([[1f64, 0.], [1., 1.], [1., 2.]]);
    let b = Matrix::from([[1.], [2.], [2.]]);
    let x = a.pivoted_qr().unwrap().solve(&b).unwrap();
    close(&x, &Matrix::from([[7. / 6.], [0.5]]));
    let residual = &a.matmul(&x) - &b;
    close(&a.transpose().matmul(&residual), &Matrix::zeros(2, 1));
    assert!(
        Matrix::from([[1., 2.], [2., 4.], [3., 6.]])
            .pivoted_qr()
            .unwrap()
            .solve(&b)
            .is_err()
    );
    let a = Matrix::from([[0f32, 1.], [2., 3.], [4., 5.]]);
    let qr = a.qr().unwrap();
    assert!(
        qr.q()
            .matmul(&qr.r())
            .approx_eq(&a, Tolerance::new(1e-5, 1e-5).unwrap())
    );
}
#[test]
fn orthogonal_application_scales_large_finite_rhs() {
    let a = Matrix::from([[1f64, 1.], [1., -1.]]);
    let q = a.qr().unwrap();
    let b = Matrix::from([[1e308], [0.]]);
    let transformed = q.apply_q(&b).unwrap();
    assert!(transformed.all_finite());
    assert!((transformed.norm_frobenius() / 1e308 - 1.).abs() < 1e-14);
    let a = a.scalar_mul(1e300);
    let b = Matrix::from([[1e300], [1e300]]);
    let solution = a.pivoted_qr().unwrap().solve(&b).unwrap();
    close(&solution, &Matrix::from([[1.], [0.]]));
}
