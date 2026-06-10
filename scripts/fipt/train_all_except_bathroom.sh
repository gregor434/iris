#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_SCRIPTS=(
  "${ROOT_DIR}/scripts/fipt/kitchen/train.sh"
  "${ROOT_DIR}/scripts/fipt/livingroom/train.sh"
)

if [[ ${#TRAIN_SCRIPTS[@]} -eq 0 ]]; then
  echo "No training scripts configured."
  exit 1
fi

echo "Running ${#TRAIN_SCRIPTS[@]} training scripts (kitchen, livingroom):"
for script in "${TRAIN_SCRIPTS[@]}"; do
  if [[ ! -f "${script}" ]]; then
    echo "Missing script: ${script}"
    exit 1
  fi
  rel_script="${script#${ROOT_DIR}/}"
  echo "========================================"
  echo "[START] ${rel_script}"
  bash "${script}"
  echo "[DONE]  ${rel_script}"
done

echo "========================================"
echo "All requested FIPT training scripts completed successfully."
