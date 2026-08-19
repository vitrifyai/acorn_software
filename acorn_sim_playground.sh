#!/usr/bin/env bash
# Launch the private ACORN simulation playground, isolated from the shared ACORN install.
set -e

cd "/home/vnw/acorn-sim-playground"
export ACORN_PLAYGROUND_RUNTIME="/home/vnw/acorn-sim-playground/.runtime"
mkdir -p "$ACORN_PLAYGROUND_RUNTIME/cache/acorn/embeddings" \
         "$ACORN_PLAYGROUND_RUNTIME/config" \
         "$ACORN_PLAYGROUND_RUNTIME/share" \
         "$ACORN_PLAYGROUND_RUNTIME/matplotlib"

# Keep playground image/cache/config state separate from the original ACORN install.
export ACORN_EMBEDDING_CACHE="$ACORN_PLAYGROUND_RUNTIME/cache/acorn/embeddings"
export XDG_CACHE_HOME="$ACORN_PLAYGROUND_RUNTIME/cache"
export XDG_CONFIG_HOME="$ACORN_PLAYGROUND_RUNTIME/config"
export XDG_DATA_HOME="$ACORN_PLAYGROUND_RUNTIME/share"
export MPLCONFIGDIR="$ACORN_PLAYGROUND_RUNTIME/matplotlib"

export ACORN_MODELS_DIR="/opt/models/acorn/models"
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset XLA_PYTHON_CLIENT_ALLOCATOR
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_strict_conv_algorithm_picker=false"

source "/home/vnw/acorn-sim-playground/.venv-py312/bin/activate"
exec acorn-gui "$@"
