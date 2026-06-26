#!/usr/bin/env bash

# Installs local whisper.cpp tooling for this project.
# Usage:
# - run `bash python/run/setup_whisper_cpp.sh`
# - it clones whisper.cpp into python/libs, builds whisper-cli, downloads a model

set -euo pipefail

PYTHON_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WHISPER_DIR="$PYTHON_DIR/libs/whisper.cpp"
MODEL_NAME="${WHISPER_MODEL_NAME:-base}"
MODEL_PATH="$WHISPER_DIR/models/ggml-${MODEL_NAME}.bin"

if ! command -v git >/dev/null 2>&1; then
  echo "git not found in PATH"
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found in PATH"
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found in PATH"
  echo "Install it first, for example with: brew install ffmpeg"
  exit 1
fi

if [ ! -d "$WHISPER_DIR/.git" ]; then
  git clone https://github.com/ggml-org/whisper.cpp.git "$WHISPER_DIR"
fi

cmake -S "$WHISPER_DIR" -B "$WHISPER_DIR/build"
cmake --build "$WHISPER_DIR/build" -j --config Release

WHISPER_BIN="$WHISPER_DIR/build/bin/whisper-cli"
if [ -f "$WHISPER_BIN" ] && command -v install_name_tool >/dev/null 2>&1; then
  for old_rpath in \
    "@executable_path/../src" \
    "@executable_path/../ggml/src" \
    "@executable_path/../ggml/src/ggml-blas" \
    "@executable_path/../ggml/src/ggml-metal" \
    "$WHISPER_DIR/build/src" \
    "$WHISPER_DIR/build/ggml/src" \
    "$WHISPER_DIR/build/ggml/src/ggml-blas" \
    "$WHISPER_DIR/build/ggml/src/ggml-metal"
  do
    install_name_tool -delete_rpath "$old_rpath" "$WHISPER_BIN" 2>/dev/null || true
  done

  install_name_tool \
    -add_rpath "@executable_path/../src" \
    -add_rpath "@executable_path/../ggml/src" \
    -add_rpath "@executable_path/../ggml/src/ggml-blas" \
    -add_rpath "@executable_path/../ggml/src/ggml-metal" \
    "$WHISPER_BIN"
fi

if [ ! -f "$MODEL_PATH" ]; then
  (
    cd "$WHISPER_DIR"
    sh ./models/download-ggml-model.sh "$MODEL_NAME"
  )
fi

echo
echo "whisper.cpp is ready."
echo "Binary: $WHISPER_DIR/build/bin/whisper-cli"
echo "Model:  $MODEL_PATH"
