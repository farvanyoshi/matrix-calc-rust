# Kernel measurements

Run `cargo bench --locked --bench kernels`. The harness warms each operation and
reports the median of seven samples of ten operations, in microseconds per
operation. Inputs are deterministic; results are consumed with `black_box`.
Measurements below used the cloud Linux x86-64 worker and Rust 1.85.1, release
optimization. They describe this instance, not a portable performance guarantee.

The initial generic multiplication kernel was compared with a contiguous row-slice
kernel. The latter preserves row-major accumulation order, avoids repeated
coordinate checks inside the innermost loop, and shares its implementation with
`matmul_into`. The output-buffer path validates all dimensions before writing and
reuses its existing allocation. Safe borrowing prevents input/output aliasing.

| Operation | Initial µs | Row-slice kernel µs |
| --- | ---: | ---: |
| Owned multiplication, 8×8 | 0.475 | 0.233 |
| Owned multiplication, 32×32 | 9.965 | 6.119 |
| Owned multiplication, 64×64 | 59.446 | 49.760 |
| Output-buffer multiplication, 8×8 | 0.479 | 0.177 |
| Output-buffer multiplication, 32×32 | 10.008 | 6.021 |
| Output-buffer multiplication, 64×64 | 59.828 | 47.410 |

Other local measurements after the change:

| Operation | Median µs |
| --- | ---: |
| Strided transpose-view multiplication, 64×64 | 62.704 |
| Owned transpose, 64×64 | 4.217 |
| LU factorization, 64×64 | 84.896 |
| Reused LU solve, 64 rows and 4 RHS | 14.315 |
| Pivoted Householder QR, 48×24 | 289.101 |
| One-sided Jacobi SVD, 48×24 | 3463.064 |
| Cholesky, 24×24 | 5.735 |
| Symmetric Jacobi eigen, 24×24 | 138.058 |

Submicrosecond timings are especially sensitive to noise, CPU scheduling, and
code layout. Measurements are observations rather than enforced CI thresholds.
The SVD/QR figures justify documenting Jacobi's intended small/medium dense scope;
they do not establish a benefit from a native backend, parallelism, or cache
blocking. Those changes need representative larger-workload measurements and
numerical revalidation before adoption. There is no newly claimed speedup for
unchanged decomposition kernels.
