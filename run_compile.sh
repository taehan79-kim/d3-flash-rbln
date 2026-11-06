#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 MODEL_DIRECTORY NEW_OUTPUT_DIRECTORY [BUCKET=256]" >&2
  exit 2
fi
command -v d3-rbln >/dev/null || { echo 'Activate the prepared SDK environment first.' >&2; exit 2; }
# Does not install, download, allocate runtime devices, or alter existing services.
exec d3-rbln compile --model-dir "$1" --output "$2" --bucket "${3:-256}"
