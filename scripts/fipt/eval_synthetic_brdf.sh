#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 --manifest <path/to/manifest.csv> [--group-by <col> ...] [--output-csv <path>]"
  exit 1
fi

cd "${ROOT_DIR}"
python -m utils.metric_brdf "$@"
