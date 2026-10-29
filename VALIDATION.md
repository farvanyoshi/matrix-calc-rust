# Implementation verification

Validation is performed on `dev` with Rust 1.85.1 on Linux x86-64. CI also defines
checks for current stable Rust; that hosted CI job has not been run by this local
implementation session. All changes and commit identities are confined to the
Rust repository and use `farvanyoshi <nvnjoshi@outlook.com>`.

The final verification covers formatting, default/minimal features, optional
serde, release mode, warning-free Clippy and documentation, executable examples,
benchmarks, and verified packaging. Generated numerical tests use a fixed ChaCha
seed and an independent development-only `nalgebra` oracle. Their 416 generated
cases are separate from the conventional test-function count.

Results are filled in after executing the final commands. Packaging is a local
verification and does not publish the crate. The original repository has no
license declaration; package metadata retains that limitation rather than
selecting a license on the owner's behalf.
