# Kernel measurements

Run `cargo bench --locked --bench kernels`. The harness warms each operation and
reports the median of seven samples, in microseconds per operation. Each sample
runs batches of 32 operations for at least 20 milliseconds to reduce scheduling
noise. Inputs are deterministic; results are consumed with `black_box`.
Measurements below used the cloud Linux x86-64 worker and Rust 1.85.1, release
optimization. They describe this instance, not a portable performance guarantee.

The exploratory comparison below used seven shorter samples of ten operations
per sample. Because those short runs showed scheduling variation, the final
harness uses the longer sampling policy above. The initial generic
multiplication kernel was compared with a contiguous row-slice
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

Final steady-state measurements with the longer sampling policy:

| Operation | Median µs |
| --- | ---: |
| Owned multiplication, 64×64 | 51.393 |
| Output-buffer multiplication, 64×64 | 49.906 |
| Strided transpose-view multiplication, 64×64 | 70.366 |
| Owned multiplication, 128×128 | 399.309 |
| Owned multiplication, 256×256 | 3134.538 |
| Output-buffer multiplication, 256×256 | 3057.029 |
| Strided transpose-view multiplication, 256×256 | 3218.742 |
| Owned transpose, 256×256 | 81.737 |
| LU factorization, 256×256 | 4076.439 |
| Reused LU solve, 256 rows and 4 RHS | 254.047 |
| Pivoted Householder QR, 48×24 | 307.075 |
| One-sided Jacobi SVD, 48×24 | 3429.023 |
| Cholesky, 24×24 | 7.275 |
| Symmetric Jacobi eigen, 24×24 | 137.916 |

Submicrosecond timings are especially sensitive to noise, CPU scheduling, and
code layout. Measurements are observations rather than enforced CI thresholds.
The SVD/QR figures justify documenting Jacobi's intended small/medium dense scope;
they do not establish a benefit from a native backend, parallelism, or cache
blocking. Those changes need representative larger-workload measurements and
numerical revalidation before adoption. There is no newly claimed speedup for
unchanged decomposition kernels.
