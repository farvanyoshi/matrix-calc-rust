use matrix_calc::{
    DecompositionOptions, LeastSquaresMethod, Matrix, MatrixError, NumericError, Tolerance,
};
fn close(a: &Matrix<f64>, b: &Matrix<f64>) {
    assert!(
        a.approx_eq(b, Tolerance::new(1e-10, 1e-10).unwrap()),
        "{a}\n!=\n{b}"
    );
}
#[test]
fn svd_reconstruction_orthogonality_and_pseudoinverse() {
    for a in [
        Matrix::from([[1., 2.], [3., 4.], [5., 6.]]),
        Matrix::from([[1., 2., 3.], [4., 5., 6.]]),
        Matrix::from([[1., 2.], [2., 4.]]),
        Matrix::zeros(3, 2),
        Matrix::zeros(0, 3),
        Matrix::zeros(3, 0),
        Matrix::identity(3),
    ] {
        let svd = a.svd().unwrap();
        let s = Matrix::from_diagonal(svd.singular_values());
        let r = a.rows().min(a.cols());
        close(&svd.u().matmul(&s).matmul(svd.v_t()), &a);
        close(&svd.u().transpose().matmul(svd.u()), &Matrix::identity(r));
        close(
            &svd.v_t().matmul(&svd.v_t().transpose()),
            &Matrix::identity(r),
        );
        let p = svd.pseudoinverse().unwrap();
        close(&a.matmul(&p).matmul(&a), &a);
        close(&p.matmul(&a).matmul(&p), &p);
        let ap = a.matmul(&p);
        let pa = p.matmul(&a);
        close(&ap, &ap.transpose());
        close(&pa, &pa.transpose());
        assert!(
            svd.singular_values()
                .windows(2)
                .all(|w| w[0] >= w[1] && w[1] >= 0.)
        );
    }
}
#[test]
fn minimum_norm_rank_and_scaled_systems() {
    let a = Matrix::from([[1f64, 1.]]);
    let result = a.least_squares(&Matrix::from([[2.]])).unwrap();
    close(&result.solution, &Matrix::from([[1.], [1.]]));
    assert_eq!(result.method, LeastSquaresMethod::Svd);
    let a = Matrix::from([[1f64, 2.], [2., 4.]]);
    assert_eq!(a.rank().unwrap(), 1);
    assert!(a.condition_number().unwrap().is_infinite());
    let b = Matrix::from([[1.], [2.]]);
    let x = a.least_squares(&b).unwrap();
    close(&a.matmul(&x.solution), &b);
    close(&x.solution, &Matrix::from([[0.2], [0.4]]));
    for scale in [1e-200, 1e200] {
        let x = a
            .scalar_mul(scale)
            .least_squares(&b.scalar_mul(scale))
            .unwrap();
        close(&x.solution, &Matrix::from([[0.2], [0.4]]));
    }
    let a = Matrix::from([[1f32, 2.], [3., 4.], [5., 6.]]);
    let s = a.svd().unwrap();
    assert!(
        s.u()
            .matmul(&Matrix::from_diagonal(s.singular_values()))
            .matmul(s.v_t())
            .approx_eq(&a, Tolerance::new(1e-5, 1e-5).unwrap())
    );
}
#[test]
fn symmetric_eigen_and_limits() {
    for a in [
        Matrix::from([[2f64, 1.], [1., 2.]]),
        Matrix::identity(3),
        Matrix::zeros(0, 0),
        Matrix::from([[2., 1., 3.], [1., 4., 2.], [3., 2., 5.]]),
    ] {
        let e = a.symmetric_eigen().unwrap();
        let v = e.eigenvectors();
        close(
            &a.matmul(v),
            &v.matmul(&Matrix::from_diagonal(e.eigenvalues())),
        );
        close(&v.transpose().matmul(v), &Matrix::identity(a.rows()));
        assert!(e.eigenvalues().windows(2).all(|w| w[0] <= w[1]));
    }
    assert!(
        Matrix::from([[1f64, 2.], [0., 1.]])
            .symmetric_eigen()
            .is_err()
    );
    let options = DecompositionOptions {
        max_iterations: 1,
        ..Default::default()
    };
    let a = Matrix::from([[1f64, 2., 3.], [4., 5., 6.], [7., 8., 10.]]);
    assert!(matches!(
        a.svd_with_options(options),
        Err(MatrixError::NumErr(NumericError::NonConvergence { .. }))
    ));
    assert!(
        a.svd_with_options(DecompositionOptions {
            max_iterations: 0,
            ..Default::default()
        })
        .is_err()
    );
    assert!(Matrix::from([[f64::INFINITY]]).svd().is_err());
}
#[test]
fn single_precision_eigen_residual_is_scaled_to_the_input() {
    let a = Matrix::from([
        [2f32, -2., -7., 7., 1.],
        [0., -5., -3., -2., -4.],
        [7., -10., 10., 4., -3.],
        [-8., -9., -4., -2., 0.],
        [-7., -5., -9., -7., -7.],
    ]);
    let a = &a + &a.transpose();
    let e = a.symmetric_eigen().unwrap();
    let residual = &a.matmul(e.eigenvectors())
        - &e.eigenvectors()
            .matmul(&Matrix::from_diagonal(e.eigenvalues()));
    assert!(residual.norm_frobenius() <= 64. * f32::EPSILON * 5. * a.norm_frobenius());
}
