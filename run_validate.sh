#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 ARTIFACT_DIRECTORY PHYSICAL_ID_A,PHYSICAL_ID_B LOGICAL_RUNTIME_ID" >&2
  echo 'Check free chips and SDK physical-to-logical mapping before running.' >&2
  exit 2
fi
command -v d3-rbln >/dev/null || { echo 'Activate the prepared SDK environment first.' >&2; exit 2; }
# CLI rejects malformed/duplicate visibility IDs and conflicting RBLN_DEVICES.
export RBLN_VISIBLE_DEVICES="$2"
exec d3-rbln validate --artifact-dir "$1" --runtime-device "$3"
