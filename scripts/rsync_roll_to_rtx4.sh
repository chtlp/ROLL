#!/bin/bash
set -euo pipefail

SOURCE_DIR="${HOME}/workspace/ROLL/"
REMOTE_HOST="rtx4"
REMOTE_DIR="~/workspace/ROLL"

# Keep the sync focused on source files and repo state instead of local artifacts.
EXCLUDES=(
  "--exclude" "__pycache__/"
  "--exclude" "*.pyc"
  "--exclude" ".pytest_cache/"
  "--exclude" ".mypy_cache/"
  "--exclude" ".ruff_cache/"
  "--exclude" ".DS_Store"
)

ssh "${REMOTE_HOST}" "mkdir -p ~/workspace"

rsync -avz --progress "${EXCLUDES[@]}" "${SOURCE_DIR}" "${REMOTE_HOST}:${REMOTE_DIR}"
