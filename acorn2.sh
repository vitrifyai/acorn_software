#!/usr/bin/env bash
# Launch ACORN 2.0 (Python 3.12 + CryoBLOB plugin).
cd "$HOME" 2>/dev/null || cd "/home/$(whoami)" 2>/dev/null || cd /tmp
export ACORN_MODELS_DIR="/opt/models/acorn/models"
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset XLA_PYTHON_CLIENT_ALLOCATOR
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_strict_conv_algorithm_picker=false"
source "/home/vnw/acorn/.venv-py312/bin/activate"
exec acorn-gui "$@"
