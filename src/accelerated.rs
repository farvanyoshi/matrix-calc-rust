//! Optional execution strategies for dense `f64` matrix kernels.
//!
//! Shape requirements: multiplication accepts A(m,k), B(k,n) and produces C(m,n).
//! All wrappers validate dimensions before allocation. Rust slices and Rayon own
//! partitioning keep the scalar/blocked/parallel paths safe. The only unsafe code
//! is in private architecture kernels: their caller verifies CPU capabilities and
//! slice lengths; vector loops never load or store beyond a full SIMD lane.
use crate::{Matrix, NumericError, Result, ShapeError};
use rayon::prelude::*;
use std::fmt;

/// Runtime-selected hardware implementation; no unsupported instruction is used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdBackend {
    /// Portable scalar fallback on unsupported hardware.
    Scalar,
    /// Four-lane AVX double-precision arithmetic on x86/x86-64.
    Avx,
    /// Two-lane NEON double-precision arithmetic on AArch64.
    Neon,
}
impl fmt::Display for SimdBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Scalar => "scalar",
            Self::Avx => "avx",
            Self::Neon => "neon",
        })
    }
}
/// Detect the usable backend. Shape requirements: none. Complexity: O(1).
pub fn simd_backend() -> SimdBackend {
    if cfg!(miri) {
        return SimdBackend::Scalar;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx") {
        return SimdBackend::Avx;
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return SimdBackend::Neon;
    }
    SimdBackend::Scalar
}
fn dimensions(a: &Matrix<f64>, b: &Matrix<f64>) -> Result<()> {
    if a.cols() != b.rows() {
        return Err(ShapeError::Dimension {
            op: "accelerated multiply",
            l_rows: a.rows(),
            l_cols: a.cols(),
            r_rows: b.rows(),
            r_cols: b.cols(),
        }
        .into());
    }
    Ok(())
}
fn dot_dimensions(a: &[f64], b: &[f64]) -> Result<()> {
    if a.len() != b.len() {
        return Err(ShapeError::Dimension {
            op: "dot",
            l_rows: 1,
            l_cols: a.len(),
            r_rows: 1,
            r_cols: b.len(),
        }
        .into());
    }
    Ok(())
}
/// Portable dot product baseline. Shape requirements: equal-length vectors.
/// Complexity: O(n), O(1) extra storage; empty vectors return zero.
pub fn dot_scalar(a: &[f64], b: &[f64]) -> Result<f64> {
    dot_dimensions(a, b)?;
    Ok(a.iter().zip(b).fold(0.0, |s, (x, y)| s + x * y))
}
/// Runtime-dispatched SIMD dot product. Shape requirements: equal lengths.
/// Complexity: O(n), O(1) extra storage; scalar tails support any alignment/length.
/// Reduction ordering differs from the scalar baseline, so roundoff may differ.
pub fn dot_simd(a: &[f64], b: &[f64]) -> Result<f64> {
    dot_dimensions(a, b)?;
    let result = match simd_backend() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx => {
            // SAFETY: detection established AVX support; lengths are identical.
            unsafe { x86::dot(a, b) }
        }
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => {
            // SAFETY: detection established NEON support; lengths are identical.
            unsafe { arm::dot(a, b) }
        }
        _ => a.iter().zip(b).fold(0.0, |s, (x, y)| s + x * y),
    };
    Ok(result)
}
fn axpy_scalar(scale: f64, x: &[f64], y: &mut [f64]) {
    for (dst, src) in y.iter_mut().zip(x) {
        *dst += scale * src;
    }
}
fn axpy(scale: f64, x: &[f64], y: &mut [f64], backend: SimdBackend) {
    debug_assert_eq!(x.len(), y.len());
    match backend {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        SimdBackend::Avx => {
            // SAFETY: caller obtained backend from detection; rows have equal n.
            unsafe { x86::axpy(scale, x, y) }
        }
        #[cfg(target_arch = "aarch64")]
        SimdBackend::Neon => {
            // SAFETY: caller obtained backend from detection; rows have equal n.
            unsafe { arm::axpy(scale, x, y) }
        }
        _ => axpy_scalar(scale, x, y),
    }
}
fn blocked_rows(a: &Matrix<f64>, b: &Matrix<f64>, first: usize, out: &mut [f64], block: usize) {
    let n = b.cols();
    let rows = out.len() / n;
    for kk in (0..a.cols()).step_by(block) {
        for jj in (0..n).step_by(block) {
            let kend = kk.saturating_add(block).min(a.cols());
            let jend = jj.saturating_add(block).min(n);
            for r in 0..rows {
                for k in kk..kend {
                    let value = a[(first + r, k)];
                    let x = &b.row(k).unwrap()[jj..jend];
                    let y = &mut out[r * n + jj..r * n + jend];
                    axpy_scalar(value, x, y);
                }
            }
        }
    }
}
/// Cache-blocked multiplication. Shape requirements: A(m,k), B(k,n), block>0.
/// Complexity: O(mkn), O(mn) result storage; handles empty dimensions and tails.
pub fn matmul_blocked(a: &Matrix<f64>, b: &Matrix<f64>, block: usize) -> Result<Matrix<f64>> {
    dimensions(a, b)?;
    if block == 0 {
        return Err(NumericError::InvalidOptions.into());
    }
    let mut out = Matrix::try_zeros(a.rows(), b.cols())?;
    if out.is_empty() || a.cols() == 0 {
        return Ok(out);
    }
    let chunk = block.min(a.rows()) * b.cols();
    for (tile, data) in out.data_mut().chunks_mut(chunk).enumerate() {
        blocked_rows(a, b, tile * block, data, block);
    }
    Ok(out)
}
/// Rayon blocked multiplication. Shape requirements: A(m,k), B(k,n), block>0.
/// Complexity: O(mkn) work; disjoint row tiles are scheduled on Rayon's pool.
/// Arithmetic within each output cell preserves the scalar k traversal order.
pub fn matmul_parallel(a: &Matrix<f64>, b: &Matrix<f64>, block: usize) -> Result<Matrix<f64>> {
    dimensions(a, b)?;
    if block == 0 {
        return Err(NumericError::InvalidOptions.into());
    }
    let mut out = Matrix::try_zeros(a.rows(), b.cols())?;
    if out.is_empty() || a.cols() == 0 {
        return Ok(out);
    }
    let chunk = block.min(a.rows()) * b.cols();
    out.data_mut()
        .par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(tile, data)| blocked_rows(a, b, tile * block, data, block));
    Ok(out)
}
/// SIMD row-update multiplication with runtime dispatch and scalar fallback.
/// Shape requirements: A(m,k), B(k,n). Complexity: O(mkn), O(mn) result storage.
/// SIMD lanes update distinct columns; no fused operations or padding are needed.
pub fn matmul_simd(a: &Matrix<f64>, b: &Matrix<f64>) -> Result<Matrix<f64>> {
    dimensions(a, b)?;
    let mut out = Matrix::try_zeros(a.rows(), b.cols())?;
    if out.is_empty() || a.cols() == 0 {
        return Ok(out);
    }
    let backend = simd_backend();
    for r in 0..a.rows() {
        let row = out.row_mut(r).unwrap();
        for k in 0..a.cols() {
            axpy(a[(r, k)], b.row(k).unwrap(), row, backend);
        }
    }
    Ok(out)
}
impl Matrix<f64> {
    /// Cache-blocked product. Shape requirements: self(m,k), rhs(k,n), block>0.
    pub fn matmul_blocked(&self, rhs: &Self, block: usize) -> Result<Self> {
        matmul_blocked(self, rhs, block)
    }
    /// Rayon product. Shape requirements: self(m,k), rhs(k,n), block>0.
    pub fn matmul_parallel(&self, rhs: &Self, block: usize) -> Result<Self> {
        matmul_parallel(self, rhs, block)
    }
    /// Dispatched SIMD product. Shape requirements: self(m,k), rhs(k,n).
    pub fn matmul_simd(&self, rhs: &Self) -> Result<Self> {
        matmul_simd(self, rhs)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn dot(a: &[f64], b: &[f64]) -> f64 {
        // Full-lane loop bounds ensure each unaligned load remains in its slice.
        unsafe {
            let mut sum = _mm256_setzero_pd();
            let mut i = 0;
            while i + 4 <= a.len() {
                sum = _mm256_add_pd(
                    sum,
                    _mm256_mul_pd(
                        _mm256_loadu_pd(a.as_ptr().add(i)),
                        _mm256_loadu_pd(b.as_ptr().add(i)),
                    ),
                );
                i += 4;
            }
            let mut lanes = [0.0; 4];
            _mm256_storeu_pd(lanes.as_mut_ptr(), sum);
            let mut result = lanes.into_iter().sum::<f64>();
            while i < a.len() {
                result += a[i] * b[i];
                i += 1;
            }
            result
        }
    }
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn axpy(scale: f64, x: &[f64], y: &mut [f64]) {
        unsafe {
            let a = _mm256_set1_pd(scale);
            let mut i = 0;
            while i + 4 <= x.len() {
                let v = _mm256_add_pd(
                    _mm256_loadu_pd(y.as_ptr().add(i)),
                    _mm256_mul_pd(a, _mm256_loadu_pd(x.as_ptr().add(i))),
                );
                _mm256_storeu_pd(y.as_mut_ptr().add(i), v);
                i += 4;
            }
            while i < x.len() {
                y[i] += scale * x[i];
                i += 1;
            }
        }
    }
}
#[cfg(target_arch = "aarch64")]
mod arm {
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn dot(a: &[f64], b: &[f64]) -> f64 {
        unsafe {
            let mut sum = vdupq_n_f64(0.0);
            let mut i = 0;
            while i + 2 <= a.len() {
                sum = vaddq_f64(
                    sum,
                    vmulq_f64(vld1q_f64(a.as_ptr().add(i)), vld1q_f64(b.as_ptr().add(i))),
                );
                i += 2;
            }
            let mut result = vaddvq_f64(sum);
            while i < a.len() {
                result += a[i] * b[i];
                i += 1;
            }
            result
        }
    }
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn axpy(scale: f64, x: &[f64], y: &mut [f64]) {
        unsafe {
            let a = vdupq_n_f64(scale);
            let mut i = 0;
            while i + 2 <= x.len() {
                vst1q_f64(
                    y.as_mut_ptr().add(i),
                    vaddq_f64(
                        vld1q_f64(y.as_ptr().add(i)),
                        vmulq_f64(a, vld1q_f64(x.as_ptr().add(i))),
                    ),
                );
                i += 2;
            }
            while i < x.len() {
                y[i] += scale * x[i];
                i += 1;
            }
        }
    }
}
