#!/usr/bin/env bash
# Build a portable tar.gz bundle using conda-pack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION="${VOICEGEN_VERSION:-0.1.0}"

cd "$PROJECT_ROOT"

if ! command -v conda-pack >/dev/null 2>&1; then
    echo "Installing conda-pack into base env..."
    conda install -y -n base -c conda-forge conda-pack
fi

mkdir -p build/voicegen-linux
ENV_TARBALL="build/voicegen-linux/env.tar.gz"
conda pack -n voicegen -o "$ENV_TARBALL" --force

cp -r app scripts third_party environment.yml requirements.txt \
      README.md DEPLOYMENT.md LICENSE installer build/voicegen-linux/
cat > build/voicegen-linux/voicegen <<'EOF'
#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -d "$DIR/env_unpacked" ]; then
    mkdir -p "$DIR/env_unpacked"
    tar -xzf "$DIR/env.tar.gz" -C "$DIR/env_unpacked"
    "$DIR/env_unpacked/bin/conda-unpack"
fi
exec "$DIR/env_unpacked/bin/python" -m app.main "$@"
EOF
chmod +x build/voicegen-linux/voicegen

tar -czf "build/voicegen-${VERSION}-linux-x64.tar.gz" -C build voicegen-linux
echo "Done: build/voicegen-${VERSION}-linux-x64.tar.gz"
