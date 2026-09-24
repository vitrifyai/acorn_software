#!/usr/bin/env bash
# Launch ACORN 2.0 (Python 3.12 + CryoBLOB plugin).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ACORN_VENV_DIR:-$SCRIPT_DIR/.venv}"
if [ ! -x "$VENV_DIR/bin/acorn-gui" ]; then
    echo "ACORN environment not found at $VENV_DIR. Run: bash install.sh" >&2
    exit 1
fi

cd "$SCRIPT_DIR"
export ACORN_MODELS_DIR="/opt/models/acorn/models"
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset XLA_PYTHON_CLIENT_ALLOCATOR
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_strict_conv_algorithm_picker=false"
source "$VENV_DIR/bin/activate"
exec "$VENV_DIR/bin/acorn-gui" "$@"
