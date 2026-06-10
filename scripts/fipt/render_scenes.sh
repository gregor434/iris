#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIPT_DIR="${ROOT_DIR}/scripts/fipt"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <scene> [<scene> ...]"
  exit 1
fi

for scene in "$@"; do
  script="${FIPT_DIR}/${scene}/render.sh"
  if [[ ! -f "${script}" ]]; then
    echo "Unknown scene or missing render script: ${scene}" >&2
    exit 1
  fi
done

for scene in "$@"; do
  script="${FIPT_DIR}/${scene}/render.sh"
  rel_script="${script#${ROOT_DIR}/}"
  echo "========================================"
  echo "[START] ${rel_script}"
  bash "${script}"
  echo "[DONE]  ${rel_script}"
done

echo "========================================"
echo "Requested FIPT renders completed successfully."
