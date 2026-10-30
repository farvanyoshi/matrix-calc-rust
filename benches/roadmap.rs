//! Criterion baselines for roadmap matrices. Allocation is included consistently.
//! Shape requirements: square dense f64 matrices of each requested benchmark size.
use criterion::{BenchmarkId, Criterion, SamplingMode, black_box};
use matrix_calc::{
    Matrix,
    accelerated::{
        dot_scalar, dot_simd, matmul_blocked, matmul_parallel, matmul_simd, simd_backend,
    },
};
use std::{
    env,
    time::{Duration, Instant},
};

/// Shape requirements: returns two n×n matrices with exactly representable binary fractions.
fn inputs(n: usize) -> (Matrix<f64>, Matrix<f64>) {
    (
        Matrix::from_fn(n, n, |r, c| {
            (((r * 17 + c * 13 + 1) % 31) as f64 - 15.) / 16.
        }),
        Matrix::from_fn(n, n, |r, c| {
            (((r * 7 + c * 11 + 3) % 29) as f64 - 14.) / 16.
        }),
    )
}
/// Shape requirements: a is M×N, b is N×P. Intentionally naive i-j-k baseline.
#[inline(never)]
fn naive(a: &Matrix<f64>, b: &Matrix<f64>) -> Matrix<f64> {
    assert_eq!(a.cols(), b.rows());
    Matrix::from_fn(a.rows(), b.cols(), |i, j| {
        let mut sum = 0.;
        for k in 0..a.cols() {
            sum += a[(i, k)] * b[(k, j)];
        }
        sum
    })
}
/// Shape requirements: any stored matrix. Weighted checksum avoids an uninformative zero sum.
fn checksum(a: &Matrix<f64>) -> f64 {
    a.data()
        .iter()
        .enumerate()
        .map(|(i, x)| *x * ((i % 97 + 1) as f64))
        .sum()
}
/// Shape requirements: a and b are n×n. Every implementation includes output allocation.
fn run(name: &str, a: &Matrix<f64>, b: &Matrix<f64>) -> Matrix<f64> {
    match name {
        "naive" => naive(a, b),
        "blocked" => matmul_blocked(a, b, 32).unwrap(),
        "parallel" => matmul_parallel(a, b, 32).unwrap(),
        "simd" => matmul_simd(a, b).unwrap(),
        _ => panic!("unknown ROADMAP_KERNEL: {name}"),
    }
}
/// Shape requirements: all requested benchmark sizes must be positive.
fn sizes(mode: &str) -> Vec<usize> {
    let default = if mode == "quick" || mode == "profile" {
        "128"
    } else {
        "128,512,1024"
    };
    env::var("ROADMAP_SIZES")
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|s| s.parse::<usize>().expect("positive integer benchmark size"))
        .inspect(|n| assert!(*n > 0))
        .collect()
}
/// Shape requirements: lhs/rhs must be equal-shaped kernel outputs.
fn max_error(a: &Matrix<f64>, b: &Matrix<f64>) -> f64 {
    assert_eq!(a.shape(), b.shape());
    a.data()
        .iter()
        .zip(b.data())
        .map(|(x, y)| (x - y).abs())
        .fold(0., f64::max)
}
/// Shape requirements: square inputs at each size; verify mode reports machine-readable seconds/checksums.
fn main() {
    let mode = env::var("ROADMAP_MODE").unwrap_or_else(|_| "full".into());
    assert!(
        ["full", "quick", "verify", "profile"].contains(&mode.as_str()),
        "ROADMAP_MODE must be full,quick,verify,profile"
    );
    let sizes = sizes(&mode);
    eprintln!(
        "Rust backend: {}; rayon threads: {}",
        simd_backend(),
        rayon::current_num_threads()
    );
    println!("language,kernel,size,seconds,checksum,max_error");
    if mode == "profile" {
        let kernel = env::var("ROADMAP_KERNEL").unwrap_or_else(|_| "naive".into());
        let count = env::var("ROADMAP_ITERATIONS")
            .unwrap_or_else(|_| "20".into())
            .parse::<usize>()
            .unwrap();
        assert!(count > 0);
        for n in sizes {
            let (a, b) = inputs(n);
            let start = Instant::now();
            let mut sum = 0.;
            for _ in 0..count {
                sum += checksum(&black_box(run(&kernel, &a, &b)));
            }
            println!(
                "rust,{kernel},{n},{:.9},{sum:.12},0",
                start.elapsed().as_secs_f64() / count as f64
            );
        }
        return;
    }
    let mut criterion = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(200))
        .configure_from_args();
    for n in sizes {
        let (a, b) = inputs(n);
        let start = Instant::now();
        let expected = naive(&a, &b);
        println!(
            "rust,naive,{n},{:.9},{:.12},0",
            start.elapsed().as_secs_f64(),
            checksum(&expected)
        );
        for name in ["blocked", "parallel", "simd"] {
            let start = Instant::now();
            let actual = run(name, &a, &b);
            let elapsed = start.elapsed().as_secs_f64();
            let error = max_error(&expected, &actual);
            assert!(error <= 1e-10, "{name}/{n} error {error}");
            println!(
                "rust,{name},{n},{elapsed:.9},{:.12},{error:.12}",
                checksum(&actual)
            );
        }
        if mode == "verify" {
            continue;
        }
        let mut group = criterion.benchmark_group("roadmap_matmul");
        group.sampling_mode(SamplingMode::Flat);
        for name in ["naive", "blocked", "parallel", "simd"] {
            group.bench_with_input(BenchmarkId::new(name, n), &n, |bencher, _| {
                bencher.iter(|| black_box(run(name, black_box(&a), black_box(&b))))
            });
        }
        group.finish();
    }
    let length = 1_048_576usize;
    let x: Vec<_> = (0..length)
        .map(|i| ((i % 127) as f64 - 63.) / 64.)
        .collect();
    let y: Vec<_> = (0..length)
        .map(|i| ((i % 113) as f64 - 56.) / 64.)
        .collect();
    let expected = dot_scalar(&x, &y).unwrap();
    for name in ["dot_scalar", "dot_simd"] {
        let start = Instant::now();
        let actual = if name == "dot_scalar" {
            dot_scalar(&x, &y).unwrap()
        } else {
            dot_simd(&x, &y).unwrap()
        };
        let error = (actual - expected).abs();
        assert!(error <= 1e-10);
        println!(
            "rust,{name},{length},{:.9},{actual:.12},{error:.12}",
            start.elapsed().as_secs_f64()
        );
        if mode != "verify" {
            criterion.bench_function(name, |b| {
                b.iter(|| {
                    black_box(if name == "dot_scalar" {
                        dot_scalar(black_box(&x), black_box(&y)).unwrap()
                    } else {
                        dot_simd(black_box(&x), black_box(&y)).unwrap()
                    })
                })
            });
        }
    }
    if mode != "verify" {
        criterion.final_summary();
    }
}
