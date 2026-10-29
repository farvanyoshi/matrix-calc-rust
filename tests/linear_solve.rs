use matrix_calc::{Matrix, Real, Tolerance};
fn close<T: Real>(a: &Matrix<T>, b: &Matrix<T>) {
    assert!(
        a.approx_eq(
            b,
            Tolerance::new(T::from_f64(1e-5).unwrap(), T::from_f64(1e-5).unwrap()).unwrap()
        ),
        "{a}\n!=\n{b}"
    );
}
#[test]
fn pivoted_lu_and_multiple_rhs() {
    let a = Matrix::from([[0f64, 2., 3.], [4., 5., 6.], [7., 8., 10.]]);
    let lu = a.lu().unwrap();
    close(
        &lu.permutation_matrix().matmul(&a),
        &lu.lower().matmul(&lu.upper()),
    );
    let b = Matrix::from([[1., 2.], [3., 4.], [5., 6.]]);
    let x = lu.solve(&b).unwrap();
    close(&a.matmul(&x), &b);
    let inverse = lu.inverse().unwrap();
    close(&a.matmul(&inverse), &Matrix::identity(3));
    close(&inverse.matmul(&a), &Matrix::identity(3));
    assert!((lu.determinant() + 5.).abs() < 1e-10);
    assert!(
        Matrix::from([[1f64, 2.], [2., 4.]])
            .solve(&Matrix::ones(2, 1))
            .is_err()
    );
    assert_eq!(Matrix::<f64>::zeros(0, 0).determinant().unwrap(), 1.);
    assert_eq!(
        Matrix::<f64>::zeros(0, 0)
            .solve(&Matrix::zeros(0, 3))
            .unwrap()
            .shape(),
        (0, 3)
    );
    assert!(a.solve(&Matrix::zeros(2, 1)).is_err());
    assert!(Matrix::from([[f64::NAN]]).lu().is_err());
    for scale in [1e-200, 1e200] {
        let scaled = a.scalar_mul(scale);
        close(&scaled.solve(&b.scalar_mul(scale)).unwrap(), &x);
    }
}
#[test]
fn cholesky_and_triangular() {
    let a = Matrix::from([[4f32, 2.], [2., 3.]]);
    let c = a.cholesky().unwrap();
    let l = c.lower();
    close(&l.matmul(&l.transpose()), &a);
    let b = Matrix::from([[1f32, 2.], [3., 4.]]);
    close(&a.matmul(&c.solve(&b).unwrap()), &b);
    close(&l.matmul(&l.solve_lower(&b, false).unwrap()), &b);
    let u = l.transpose();
    close(&u.matmul(&u.solve_upper(&b, false).unwrap()), &b);
    assert!(Matrix::from([[1f64, 2.], [2., 1.]]).cholesky().is_err());
    assert!(Matrix::from([[1f64, 2.], [0., 1.]]).cholesky().is_err());
    assert_eq!(
        Matrix::<f64>::zeros(0, 0)
            .cholesky()
            .unwrap()
            .log_determinant(),
        0.
    );
    assert!(
        Matrix::<f64>::zeros(1, 1)
            .solve_lower(&Matrix::ones(1, 1), false)
            .is_err()
    );
}
