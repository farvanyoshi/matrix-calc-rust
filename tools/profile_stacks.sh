#!/usr/bin/env bash
# Actual sampled main-thread stacks. Shape requirements: positive square size/repeat count.
set -euo pipefail
cd "$(dirname "$0")/.."
kernel=${1:-naive}; size=${2:-512}; iterations=${3:-12}; output=${4:-.tools/profiles/sampled}
mkdir -p "$output"
export RUSTFLAGS="${RUSTFLAGS:-} -C debuginfo=1 -C force-frame-pointers=yes"
cargo bench --locked --bench roadmap --no-run --message-format=json > "$output/build.jsonl"
binary=$(python3 - "$output/build.jsonl" <<'PY'
import json,sys
for line in open(sys.argv[1]):
    try:item=json.loads(line)
    except ValueError:continue
    if item.get('reason')=='compiler-artifact' and item.get('target',{}).get('name')=='roadmap' and item.get('executable'):print(item['executable'])
PY
)
[[ -n "$binary" ]] || { echo 'No benchmark executable' >&2; exit 2; }
ROADMAP_MODE=profile ROADMAP_KERNEL="$kernel" ROADMAP_SIZES="$size" ROADMAP_ITERATIONS="$iterations" \
 python3 tools/sample_stacks.py --gdb "${GDB:-gdb}" --output "$output" -- "$binary"
"${FLAMEGRAPH:-flamegraph.pl}" --title='Rust main-thread stack samples' --subtitle='GDB/MI interrupt samples, not hardware counters' --countname=samples "$output/stacks.folded" > "$output/flamegraph.svg"
