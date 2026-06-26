#!/usr/bin/env bash

set -euo pipefail

PYTHON_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PYTHON_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found in PATH"
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found in PATH"
  echo "Install it first, for example with: brew install ffmpeg"
  exit 1
fi

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo
echo "Python environment is ready."
echo "Next:"
echo "1. Grant Screen Recording permission to your terminal app in macOS settings."
echo "2. Restart the terminal."
echo "3. Run:"
echo "   python/.venv/bin/python python/run/capture_system_audio_macos.py --list-targets"
