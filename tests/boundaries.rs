use matrix_calc::{
    DecompositionOptions, LeastSquaresMethod, Matrix, MatrixError, NumericError, Real, Tolerance,
};
#[test]
fn empty_and_failed_operations_preserve_shapes_and_storage() {
    for (m, n) in [(0, 0), (0, 4), (3, 0), (1, 0), (0, 1)] {
        let a = Matrix::<f64>::zeros(m, n);
        let svd = a.svd().unwrap();
        assert_eq!(svd.u().shape(), (m, m.min(n)));
        assert_eq!(svd.v_t().shape(), (m.min(n), n));
        assert_eq!(svd.rank(), 0);
        assert_eq!(svd.spectral_norm(), 0.);
        assert_eq!(svd.condition_number(), 1.);
        let b = Matrix::zeros(m, 2);
        let x = a.least_squares(&b).unwrap();
        assert_eq!(x.solution.shape(), (n, 2));
        assert_eq!(x.residual_norm, 0.);
        assert_eq!(a.transpose().transpose(), a);
    }
    let mut a = Matrix::from([[1, 2], [3, 4]]);
    let original = a.clone();
    assert!(a.set_row(0, &[9]).is_err());
    assert!(a.set_col(0, &[9]).is_err());
    assert!(a.swap_rows(0, 2).is_err());
    assert!(a.swap_cols(0, 2).is_err());
    assert_eq!(a, original);
    let mut output = Matrix::filled(2, 2, 99);
    let old = output.clone();
    assert!(a.matmul_into(&Matrix::ones(1, 2), &mut output).is_err());
    assert_eq!(output, old);
    let mut nonsquare = Matrix::from([[1, 2, 3]]);
    let old = nonsquare.clone();
    assert!(nonsquare.transpose_in_place().is_err());
    assert_eq!(nonsquare, old);
    let mut a = Matrix::from([[1, 2], [3, 4]]);
    a.transpose_in_place().unwrap();
    assert_eq!(a, original.transpose());
    assert_eq!(a.row_mins().unwrap(), Matrix::from([[1], [2]]));
    assert_eq!(a.col_maxs().unwrap(), Matrix::from([[2, 4]]));
    assert_eq!(
        a.matvec(&Matrix::from([[1, 1]])).unwrap(),
        Matrix::from([[4], [6]])
    );
    assert!(Matrix::<f64>::zeros(2, 0).row_means().is_err());
    assert!(Matrix::<i32>::zeros(0, 2).col_mins().is_err());
    assert_eq!(
        Matrix::<u8>::try_from_fn(usize::MAX, 0, |_, _| panic!("empty callback"))
            .unwrap()
            .shape(),
        (usize::MAX, 0)
    );
}
fn numerical_edges<T: Real>() {
    let num = |x| T::from_f64(x).unwrap();
    let a = Matrix::from([[num(2.), num(1.)], [num(1.), num(3.)]]);
    let b = Matrix::from([[num(1.)], [num(4.)]]);
    let tol = Tolerance::new(num(2e-5), num(2e-5)).unwrap();
    let lu = a.lu().unwrap();
    let xt = lu.solve_transpose(&b).unwrap();
    assert!(a.transpose().matmul(&xt).approx_eq(&b, tol));
    assert!(a.powi(-1).unwrap().approx_eq(&a.inverse().unwrap(), tol));
    assert!(
        a.powi(-2)
            .unwrap()
            .approx_eq(&a.inverse().unwrap().pow(2).unwrap(), tol)
    );
    let e = a.symmetric_eigen().unwrap();
    assert!(
        a.matmul(e.eigenvectors()).approx_eq(
            &e.eigenvectors()
                .matmul(&Matrix::from_diagonal(e.eigenvalues())),
            tol
        )
    );
    let least = a
        .least_squares_with_options(&b, LeastSquaresMethod::Qr, Default::default())
        .unwrap();
    assert_eq!(least.method, LeastSquaresMethod::Qr);
    let nonfinite = Matrix::from([[T::nan()]]);
    assert!(nonfinite.lu().is_err());
    assert!(nonfinite.cholesky().is_err());
    assert!(nonfinite.qr().is_err());
    assert!(nonfinite.svd().is_err());
    assert!(nonfinite.symmetric_eigen().is_err());
    let zero = DecompositionOptions {
        max_iterations: 0,
        ..Default::default()
    };
    assert!(a.lu_with_options(zero).is_err());
    assert!(a.qr_with_options(true, zero).is_err());
    let options = DecompositionOptions {
        tolerance: Some(Tolerance::new(T::zero(), T::zero()).unwrap()),
        ..Default::default()
    };
    let small = Matrix::from([[T::one(), T::zero()], [T::zero(), num(1e-5)]]);
    assert_eq!(small.svd_with_options(options).unwrap().rank(), 2);
}
#[test]
fn floating_precision_and_failure_contracts() {
    numerical_edges::<f32>();
    numerical_edges::<f64>();
    let symmetric = Matrix::from([[1f64, 2., 3.], [2., 4., 5.], [3., 5., 6.]]);
    assert!(matches!(
        symmetric.symmetric_eigen_with_options(DecompositionOptions {
            max_iterations: 1,
            ..Default::default()
        }),
        Err(MatrixError::NumErr(NumericError::NonConvergence { .. }))
    ));
    let tol = Tolerance::new(0., 1.5).unwrap();
    assert!(!tol.close(f64::MAX, -f64::MAX));
    assert!(Matrix::from([[255u8]]).checked_scalar_add(1).is_err());
    assert!(Matrix::from([[1u8]]).checked_negate().is_err());
    let tiny = Matrix::from([[1e-300f64, 0.], [0., 2e-300]]);
    let s = tiny.svd().unwrap();
    assert!((s.singular_values()[1] / 1e-300 - 1.).abs() < 1e-12);
    let huge = Matrix::from([[1e300f64, 0.], [0., 2e300]]);
    assert!((huge.svd().unwrap().spectral_norm() / 2e300 - 1.).abs() < 1e-12);
    assert!(
        Matrix::from([[1f64, 2., 3.]])
            .least_squares_with_options(
                &Matrix::ones(1, 1),
                LeastSquaresMethod::Qr,
                Default::default()
            )
            .is_err()
    );
}

#[test]
fn projected_solutions_do_not_overflow_unneeded_scale_ratios() {
    let a = Matrix::from([[1e-300f64], [0.]]);
    let b = Matrix::from([[0.], [1e300]]);
    let fit = a.least_squares(&b).unwrap();
    assert_eq!(fit.solution, Matrix::from([[0.]]));
    let a = Matrix::from([[1e300f64, 0.], [0., 1e-20]]);
    let svd = a
        .svd_with_options(DecompositionOptions {
            tolerance: Some(Tolerance::new(0., 0.).unwrap()),
            ..Default::default()
        })
        .unwrap();
    let p = svd.pseudoinverse().unwrap();
    assert!((p[(1, 1)] / 1e20 - 1.).abs() < 1e-4);
    let z = Matrix::<()>::try_from_fn(usize::MAX, 0, |_, _| panic!("empty ZST callback")).unwrap();
    assert_eq!(z.shape(), (usize::MAX, 0));
}

#[test]
fn unrepresentable_factors_return_errors() {
    assert!(matches!(
        Matrix::from([[f64::MAX, f64::MAX], [-f64::MAX, f64::MAX]]).lu(),
        Err(MatrixError::NumErr(NumericError::NonFinite))
    ));
    assert!(matches!(
        Matrix::from([[f64::MAX], [f64::MAX]]).qr(),
        Err(MatrixError::NumErr(NumericError::NonFinite))
    ));
}
