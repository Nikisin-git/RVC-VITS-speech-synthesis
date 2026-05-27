#!/usr/bin/env bash
# Linux installer: creates conda env and verifies system deps.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "== VoiceGen installer =="

# 1. miniconda check
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is required. Install Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 2. system deps
missing=()
for bin in ffmpeg espeak-ng; do
    if ! command -v "$bin" >/dev/null 2>&1; then
        missing+=("$bin")
    fi
done
if [ ${#missing[@]} -ne 0 ]; then
    echo "WARN: missing system packages: ${missing[*]}"
    echo "Install with: sudo apt install ${missing[*]}"
fi

# 3. create env
cd "$PROJECT_ROOT"
if conda env list | grep -q "^voicegen "; then
    echo "Env 'voicegen' already exists — updating..."
    conda env update -n voicegen -f environment.yml --prune
else
    conda env create -f environment.yml
fi

echo
echo "== Installed. Run with:"
echo "  conda activate voicegen && python -m app.main"
