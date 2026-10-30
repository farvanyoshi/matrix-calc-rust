#!/usr/bin/env bash
# Shape requirements: positive square size and repeat count; first argument names a kernel.
# Usage: tools/profile.sh [naive|blocked|parallel|simd] [size=128] [iterations=2] [output]
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
kernel=${1:-naive};size=${2:-128};iterations=${3:-2};output=${4:-.tools/profiles/$kernel}
mkdir -p "$output"
export RUSTFLAGS="${RUSTFLAGS:-} -C debuginfo=1 -C force-frame-pointers=yes"
cargo bench --locked --bench roadmap --no-run --message-format=json > "$output/build.jsonl"
binary=$(python3 - "$output/build.jsonl" <<'PY'
import json,sys
for line in open(sys.argv[1]):
    try:r=json.loads(line)
    except ValueError:continue
    if r.get('reason')=='compiler-artifact' and r.get('target',{}).get('name')=='roadmap' and r.get('executable'):print(r['executable'])
PY
)
[[ -n "$binary" ]] || { echo 'No benchmark executable produced' >&2;exit 1; }
profiler=${VALGRIND:-$(command -v valgrind || true)}
extra=()
if [[ -z "$profiler" && -x "$root/.tools/profiling/root/usr/bin/valgrind" ]];then
  profiler="$root/.tools/profiling/root/usr/bin/valgrind"
  export VALGRIND_LIB="$root/.tools/profiling/root/usr/libexec/valgrind"
  extra+=("--extra-debuginfo-path=$root/.tools/profiling/root/usr/lib/debug")
fi
[[ -n "$profiler" ]] || { echo 'Install valgrind (and matching libc debug symbols), or set VALGRIND.' >&2;exit 2; }
ROADMAP_MODE=profile ROADMAP_KERNEL="$kernel" ROADMAP_SIZES="$size" ROADMAP_ITERATIONS="$iterations" \
  "$profiler" "${extra[@]}" --tool=callgrind --cache-sim=yes --callgrind-out-file="$output/callgrind.out" "$binary" > "$output/program.csv" 2> "$output/callgrind.log"
python3 tools/callgrind_report.py "$output/callgrind.out" "$output" | tee "$output/report.json"
