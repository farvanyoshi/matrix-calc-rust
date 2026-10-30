use matrix_calc::accelerated::{
    dot_scalar, dot_simd, matmul_blocked, matmul_parallel, matmul_simd,
};
use matrix_calc::{Matrix, Tolerance};
#[test]
fn all_products_match_rectangles_empty_shapes_and_tile_tails() {
    for (m, k, n) in [
        (0, 0, 0),
        (3, 0, 5),
        (0, 4, 7),
        (6, 4, 0),
        (1, 1, 1),
        (5, 7, 9),
        (33, 31, 35),
        (64, 65, 63),
    ] {
        let a = Matrix::from_fn(m, k, |r, c| ((r * 13 + c * 7) as f64).sin());
        let b = Matrix::from_fn(k, n, |r, c| ((r * 3 + c * 11) as f64).cos());
        let expected = a.matmul(&b);
        for block in [1, 3, 16, 32, 128, usize::MAX] {
            assert!(
                matmul_blocked(&a, &b, block)
                    .unwrap()
                    .approx_eq(&expected, Tolerance::new(1e-11, 1e-11).unwrap())
            );
            assert!(
                matmul_parallel(&a, &b, block)
                    .unwrap()
                    .approx_eq(&expected, Tolerance::new(1e-11, 1e-11).unwrap())
            );
        }
        assert!(
            matmul_simd(&a, &b)
                .unwrap()
                .approx_eq(&expected, Tolerance::new(1e-11, 1e-11).unwrap())
        );
    }
}
#[test]
fn validation_and_simd_unaligned_tails() {
    let a = Matrix::<f64>::ones(2, 3);
    let wrong = Matrix::ones(4, 2);
    assert!(matmul_blocked(&a, &wrong, 16).is_err());
    assert!(matmul_parallel(&a, &wrong, 16).is_err());
    assert!(matmul_simd(&a, &wrong).is_err());
    let b = Matrix::ones(3, 2);
    assert!(matmul_blocked(&a, &b, 0).is_err());
    assert!(matmul_parallel(&a, &b, 0).is_err());
    for length in 0..130 {
        let a: Vec<_> = (0..length + 2).map(|i| (i as f64).sin()).collect();
        let b: Vec<_> = (0..length + 2).map(|i| (i as f64).cos()).collect();
        let expected = dot_scalar(&a[1..length + 1], &b[1..length + 1]).unwrap();
        let actual = dot_simd(&a[1..length + 1], &b[1..length + 1]).unwrap();
        assert!((actual - expected).abs() < 1e-12 * (1. + expected.abs()));
    }
    assert!(dot_simd(&[1.], &[]).is_err());
    assert_eq!(dot_simd(&[], &[]).unwrap(), 0.);
}

#[cfg(miri)]
#[test]
fn miri_executes_the_scalar_dispatch_fallback() {
    use matrix_calc::accelerated::{SimdBackend, simd_backend};
    assert_eq!(simd_backend(), SimdBackend::Scalar);
    assert_eq!(dot_simd(&[1., 2., 3.], &[4., 5., 6.]).unwrap(), 32.);
    let a = Matrix::from([[1., 2.], [3., 4.]]);
    assert_eq!(matmul_simd(&a, &a).unwrap(), a.matmul(&a));
}
