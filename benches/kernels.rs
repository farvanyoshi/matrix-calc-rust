//! Reproducible wall-clock comparisons, not CI pass/fail performance thresholds.
use matrix_calc::Matrix;
use std::{hint::black_box, time::Instant};
fn measure(label: &str, mut f: impl FnMut()) {
    f();
    let mut samples = Vec::new();
    for _ in 0..7 {
        let start = Instant::now();
        let mut repetitions = 0u64;
        loop {
            for _ in 0..32 {
                f();
            }
            repetitions += 32;
            if start.elapsed().as_millis() >= 20 {
                break;
            }
        }
        samples.push(start.elapsed().as_secs_f64() * 1e6 / repetitions as f64);
    }
    samples.sort_by(f64::total_cmp);
    println!("{label},{:.3}", samples[3]);
}
fn main() {
    println!("kernel,median_microseconds");
    for n in [8, 32, 64, 128, 256] {
        let a = Matrix::from_fn(n, n, |r, c| ((r * 17 + c * 13 + 1) as f64).sin());
        let spd = &a.transpose().matmul(&a) + &Matrix::<f64>::identity(n);
        let lu = spd.lu().unwrap();
        let b = Matrix::ones(n, 4);
        measure(&format!("matmul_{n}"), || {
            let _ = black_box(a.matmul(black_box(&a)));
        });
        let mut out = Matrix::zeros(n, n);
        measure(&format!("matmul_into_{n}"), || {
            a.matmul_into(black_box(&a), black_box(&mut out)).unwrap();
        });
        measure(&format!("view_matmul_{n}"), || {
            let _ = black_box(a.view().transpose().try_matmul(black_box(&a)).unwrap());
        });
        measure(&format!("transpose_{n}"), || {
            let _ = black_box(a.transpose());
        });
        measure(&format!("lu_{n}"), || {
            let _ = black_box(spd.lu().unwrap());
        });
        measure(&format!("reuse_solve_{n}"), || {
            let _ = black_box(lu.solve(black_box(&b)).unwrap());
        });
    }
    let tall = Matrix::from_fn(48, 24, |r, c| {
        ((r * 17 + c * 13 + 1) as f64).cos() + if r == c { 2. } else { 0. }
    });
    measure("qr_48x24", || {
        let _ = black_box(tall.pivoted_qr().unwrap());
    });
    measure("svd_48x24", || {
        let _ = black_box(tall.svd().unwrap());
    });
    let symmetric = &tall.transpose().matmul(&tall) + &Matrix::<f64>::identity(24);
    measure("cholesky_24", || {
        let _ = black_box(symmetric.cholesky().unwrap());
    });
    measure("eigen_24", || {
        let _ = black_box(symmetric.symmetric_eigen().unwrap());
    });
}
