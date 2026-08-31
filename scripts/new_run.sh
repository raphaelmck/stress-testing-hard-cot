#!/usr/bin/env bash
# Create a run directory and stamp reproducibility metadata.
#   usage: scripts/new_run.sh <run_id> [command description...]
set -euo pipefail

run_id="${1:?usage: scripts/new_run.sh <run_id> [command description...]}"
shift || true
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dir="$root/artifacts/runs/$run_id"

if [ -e "$dir" ]; then
  echo "refusing to overwrite existing run: $dir" >&2
  exit 1
fi
mkdir -p "$dir"

commit="$(git -C "$root" rev-parse --verify --short HEAD 2>/dev/null || echo no-commit)"
dirty="$(git -C "$root" status --porcelain 2>/dev/null | head -c1)"
if [ -n "$dirty" ]; then commit="$commit-dirty"; fi
upstream="$(git -C "$root/cot-proxy-tasks" rev-parse --verify --short HEAD 2>/dev/null || echo unknown)"

cat > "$dir/metadata.json" <<JSON
{
  "run_id": "$run_id",
  "commit": "$commit",
  "upstream_data_commit": "$upstream",
  "started_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "host": "$(hostname)",
  "command": "$*"
}
JSON

echo "$dir"
