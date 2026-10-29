use matrix_calc::{Matrix, Tolerance};
use nalgebra::DMatrix;
use proptest::{
    prelude::*,
    test_runner::{Config, RngAlgorithm, TestRng, TestRunner},
};
fn runner(cases: u32) -> TestRunner {
    TestRunner::new_with_rng(
        Config {
            cases,
            failure_persistence: None,
            ..Config::default()
        },
        TestRng::from_seed(RngAlgorithm::ChaCha, &[73; 32]),
    )
}
fn close(a: &Matrix<f64>, b: &Matrix<f64>, tol: f64) -> bool {
    a.shape() == b.shape()
        && (&a.clone() - b).norm_frobenius()
            <= tol * (1. + a.norm_frobenius().max(b.norm_frobenius()))
}
fn convert(a: &DMatrix<f64>) -> Matrix<f64> {
    Matrix::from_fn(a.nrows(), a.ncols(), |r, c| a[(r, c)])
}
#[test]
fn generated_core_invariants() {
    runner(128)
        .run(
            &(
                0usize..8,
                0usize..8,
                0usize..8,
                prop::collection::vec(-20i32..20, 64),
            ),
            |(m, n, k, values)| {
                let a = Matrix::from_fn(m, n, |r, c| values[(r * n + c) % values.len()]);
                let b = Matrix::from_fn(n, k, |r, c| values[(r * k + c) % values.len()]);
                prop_assert_eq!(a.transpose().transpose(), a.clone());
                let product = a.try_matmul(&b).unwrap();
                for r in 0..m {
                    for c in 0..k {
                        let expected = (0..n).map(|i| a[(r, i)] * b[(i, c)]).sum::<i32>();
                        prop_assert_eq!(product[(r, c)], expected);
                    }
                }
                prop_assert_eq!(a.slice(0..m, 0..n).unwrap().to_owned(), a.clone());
                prop_assert!(a.get(m, 0).is_none());
                prop_assert!(a.get(0, n).is_none());
                prop_assert_eq!(a.iter_rows().count(), m);
                let mut reshaped = a.clone();
                reshaped.try_reshape(1, m * n).unwrap();
                prop_assert_eq!(reshaped.data(), a.data());
                Ok(())
            },
        )
        .unwrap();
}
#[test]
fn generated_svd_qr_and_least_squares_against_oracle() {
    runner(128)
        .run(
            &(
                1usize..7,
                1usize..7,
                any::<bool>(),
                prop::collection::vec(-10i32..11, 36),
            ),
            |(m, n, dependent, values)| {
                let mut a = Matrix::from_fn(m, n, |r, c| values[(r * n + c) % values.len()] as f64);
                if dependent && n > 1 {
                    for r in 0..m {
                        a[(r, n - 1)] = a[(r, 0)] * 2.;
                    }
                }
                let da = DMatrix::from_row_slice(m, n, a.data());
                let ds = da.clone().svd(true, true);
                let s = a.svd().unwrap();
                let r = m.min(n);
                prop_assert!(close(
                    &s.u()
                        .matmul(&Matrix::from_diagonal(s.singular_values()))
                        .matmul(s.v_t()),
                    &a,
                    1e-10
                ));
                prop_assert!(close(
                    &s.u().transpose().matmul(s.u()),
                    &Matrix::identity(r),
                    1e-10
                ));
                prop_assert!(close(
                    &s.v_t().matmul(&s.v_t().transpose()),
                    &Matrix::identity(r),
                    1e-10
                ));
                for (x, y) in s.singular_values().iter().zip(ds.singular_values.iter()) {
                    prop_assert!(
                        (x - y).abs() <= 1e-9 * (1. + y.abs()),
                        "singular values {} != {}",
                        x,
                        y
                    );
                }
                let p = s.pseudoinverse().unwrap();
                prop_assert!(close(&a.matmul(&p).matmul(&a), &a, 1e-9));
                let qr = a.pivoted_qr().unwrap();
                prop_assert!(close(
                    &qr.q().matmul(&qr.r()),
                    &a.matmul(&qr.permutation_matrix()),
                    1e-10
                ));
                let b = Matrix::from_fn(m, 2, |r, c| (r + c + 1) as f64);
                let db = DMatrix::from_row_slice(m, 2, b.data());
                let expected = convert(&ds.solve(&db, 1e-10).unwrap());
                let actual = a.least_squares(&b).unwrap().solution;
                prop_assert!(
                    close(&actual, &expected, 1e-7),
                    "solution {} != {} for {}",
                    actual,
                    expected,
                    a
                );
                Ok(())
            },
        )
        .unwrap();
}
#[test]
fn generated_lu_cholesky_and_symmetric_eigen_against_oracle() {
    runner(96)
        .run(
            &(1usize..7, prop::collection::vec(-10i32..11, 36)),
            |(n, values)| {
                let a = Matrix::from_fn(n, n, |r, c| values[(r * n + c) % values.len()] as f64);
                let lu = a.lu().unwrap();
                prop_assert!(close(
                    &lu.lower().matmul(&lu.upper()),
                    &lu.permutation_matrix().matmul(&a),
                    1e-10
                ));
                let da = DMatrix::from_row_slice(n, n, a.data());
                let det = da.clone().lu().determinant();
                prop_assert!((lu.determinant() - det).abs() <= 1e-7 * (1. + det.abs()));
                let spd = &a.transpose().matmul(&a) + &Matrix::<f64>::identity(n);
                let c = spd.cholesky().unwrap();
                prop_assert!(close(
                    &c.lower().matmul(&c.lower().transpose()),
                    &spd,
                    1e-10
                ));
                let b = Matrix::ones(n, 2);
                let solved = spd.solve(&b).unwrap();
                prop_assert!(close(&spd.matmul(&solved), &b, 1e-9));
                prop_assert!(close(&c.solve(&b).unwrap(), &solved, 1e-9));
                let symmetric = &a + &a.transpose();
                let e = symmetric.symmetric_eigen().unwrap();
                let v = e.eigenvectors();
                prop_assert!(close(
                    &symmetric.matmul(v),
                    &v.matmul(&Matrix::from_diagonal(e.eigenvalues())),
                    1e-10
                ));
                prop_assert!(
                    v.transpose()
                        .matmul(v)
                        .approx_eq(&Matrix::identity(n), Tolerance::new(1e-10, 1e-10).unwrap())
                );
                let mut expected: Vec<_> = (&da + da.transpose())
                    .symmetric_eigen()
                    .eigenvalues
                    .iter()
                    .copied()
                    .collect();
                expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
                for (x, y) in e.eigenvalues().iter().zip(expected) {
                    prop_assert!((x - y).abs() <= 1e-9 * (1. + y.abs()));
                }
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn generated_single_precision_factors_and_residuals() {
    runner(64)
        .run(
            &(1usize..7, prop::collection::vec(-10i32..11, 36)),
            |(n, values)| {
                let a = Matrix::from_fn(n, n, |r, c| values[(r * n + c) % values.len()] as f32);
                let tol =
                    Tolerance::new(64. * f32::EPSILON * n as f32, 64. * f32::EPSILON * n as f32)
                        .unwrap();
                let s = a.svd().unwrap();
                prop_assert!(
                    s.u()
                        .matmul(&Matrix::from_diagonal(s.singular_values()))
                        .matmul(s.v_t())
                        .approx_eq(&a, tol)
                );
                prop_assert!(
                    s.u()
                        .transpose()
                        .matmul(s.u())
                        .approx_eq(&Matrix::identity(n), tol)
                );
                let qr = a.pivoted_qr().unwrap();
                prop_assert!(
                    qr.q()
                        .matmul(&qr.r())
                        .approx_eq(&a.matmul(&qr.permutation_matrix()), tol)
                );
                let spd = &a.transpose().matmul(&a) + &Matrix::<f32>::identity(n);
                let b = Matrix::ones(n, 2);
                prop_assert!(
                    spd.matmul(&spd.cholesky().unwrap().solve(&b).unwrap())
                        .approx_eq(&b, tol)
                );
                let sym = &a + &a.transpose();
                let e = sym.symmetric_eigen().unwrap();
                // Factor accuracy is a global backward-residual contract. Local
                // entrywise comparison overweights near-zero entries of A*V.
                let residual = &sym.matmul(e.eigenvectors())
                    - &e.eigenvectors()
                        .matmul(&Matrix::from_diagonal(e.eigenvalues()));
                prop_assert!(residual.norm_frobenius() <= tol.relative() * sym.norm_frobenius());
                Ok(())
            },
        )
        .unwrap();
}
