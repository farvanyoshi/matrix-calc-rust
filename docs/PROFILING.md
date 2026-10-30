# Profiling and benchmark reproduction

The roadmap includes naive, blocked, Rayon/std-tasks, and SIMD products at
128×128, 512×512 and 1024×1024, plus million-element scalar/SIMD dot products.
Both languages use identical exactly representable binary-fraction inputs and
weighted checksums. Full outputs are compared elementwise before reporting times.
Rust uses Criterion (10 flat samples); Nim uses pinned benchy (10 repetitions).
Runs are sequential between languages, with four parallel workers. Wall time and
instruction counts are different measurements.

The committed `profiling/flamegraph.svg` is generated from **real GDB/MI main-thread
interrupt samples**, not invented call stacks. Rust collected 319 samples and Nim
420 from 12 naive 512×512 multiplications at a requested 20ms interval. Sampling
stops the process briefly, changes execution time, and does not aggregate worker
threads. `sampling.json` and `stacks.folded` preserve the evidence. Optimized code
can inline the numerical callback into a constructor, so a Rust `try_from_fn`
frame here includes the naive multiplication loop, not merely allocation.

Callgrind additionally collected instruction events and simulated cache events
for naive 128×128 work. `instructions.svg` is deliberately labelled a flat
self-instruction attribution chart, and `callgraph.svg` shows measured edges.
These are not hardware-counter or wall-clock flamegraphs. The Nim checked element
accessor accounted for about 77.6% of those self-instructions; accessor inlining
was subsequently added while retaining explicit bounds checks. Rust's inlined
constructor/callback accounted for about 97.5%. These figures do not isolate
individual checks or establish a hardware cache-miss rate.

To reproduce, install GDB, Valgrind with matching libc debug symbols, Perl, Python3
and optionally Graphviz. Obtain Brendan Gregg's FlameGraph scripts from
https://github.com/brendangregg/FlameGraph (the generator is a separate tool).
Activate the language toolchain, then use:

```sh
# Rust
RAYON_NUM_THREADS=4 ROADMAP_MODE=full cargo bench --locked --bench roadmap
FLAMEGRAPH=/path/to/FlameGraph/flamegraph.pl tools/profile_stacks.sh naive 512 12
# Instruction/cache simulation:
tools/profile.sh naive 128 2
```

In the Nim repository:

```sh
tools/setup_benchy.sh
nim c -d:release --threads:on --path:src --path:.tools/benchy/src \
  --nimcache:.tools/bench-cache --out:.tools/roadmap-bench benchmarks/roadmap.nim
ROADMAP_MODE=full ROADMAP_THREADS=4 .tools/roadmap-bench
FLAMEGRAPH=/path/to/FlameGraph/flamegraph.pl tools/profile_stacks.sh naive 512 12
tools/profile.sh naive 128 2
```

`GDB`, `VALGRIND`, `FLAMEGRAPH` and `BENCHY_PATH` override tool locations.
The current cloud run used workspace-local tools; no host package installation or
TLS/checksum bypass was needed. This environment restricts `/proc/*/mem` writes;
GDB used its ptrace fallback successfully. Hardware perf is not claimed. Raw
Callgrind/MI logs and full timing logs remain in ignored `.tools`; compact evidence
is committed under `docs/profiling`.
