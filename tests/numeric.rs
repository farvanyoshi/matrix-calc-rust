use matrix_calc::{Matrix, Tolerance};
#[test]
fn checked_arithmetic_is_explicit_and_atomic() {
    let a = Matrix::from([[127i8, 2]]);
    assert!(a.checked_add(&Matrix::ones(1, 2)).is_err());
    assert!(a.checked_scalar_mul(2).is_err());
    assert!(Matrix::from([[i8::MIN]]).checked_negate().is_err());
    assert!(Matrix::from([[i8::MIN]]).checked_scalar_div(-1).is_err());
    assert!(a.checked_scalar_div(0).is_err());
    let mut a = a;
    let old = a.clone();
    assert!(a.checked_scalar_mul_assign(2).is_err());
    assert_eq!(a, old);
    assert!(
        Matrix::from([[100i8, 100]])
            .checked_matmul(&Matrix::from([[1], [1]]))
            .is_err()
    );
    assert_eq!(
        Matrix::from([[2i32, 3]])
            .checked_matmul(&Matrix::from([[4], [5]]))
            .unwrap(),
        Matrix::from([[23]])
    );
}
#[test]
fn stable_real_primitives() {
    let x = Matrix::from([[1e300f64, 1e300]]);
    let n = x.vector_norm_2().unwrap();
    assert!(n.is_finite());
    assert!((n / 1e300 - 2f64.sqrt()).abs() < 1e-14);
    assert!((x.normalized().unwrap().norm_frobenius() - 1.).abs() < 1e-14);
    assert_eq!(Matrix::from([[1e308, 1e308]]).mean().unwrap(), 1e308);
    assert!((Matrix::from([[1e16f64, 1., -1e16]]).sum_compensated() - 1.).abs() < 1e-14);
    assert!(Matrix::from([[f64::NAN]]).mean().is_err());
    assert!(Matrix::from([[0f64, 0.]]).normalized().is_err());
    let a = Matrix::from([[1f32, 2.], [3., 4.]]);
    assert_eq!(a.norm_1().unwrap(), 6.);
    assert_eq!(a.norm_inf().unwrap(), 7.);
    assert_eq!(a.row_means().unwrap(), Matrix::from([[1.5], [3.5]]));
    assert!(a.approx_eq(&a, Tolerance::new(0., 0.).unwrap()));
    assert!(
        !Matrix::from([[f64::NAN]])
            .approx_eq(&Matrix::from([[f64::NAN]]), Tolerance::for_dimension(1))
    );
    assert!(Tolerance::<f64>::new(-1., 0.).is_err());
}
