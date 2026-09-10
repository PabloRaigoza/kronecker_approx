#!/usr/bin/env bash
# Build everything needed to run this repo's experiments on NERSC Perlmutter:
# SLATE (+ its bundled blaspp/lapackpp), then ./main (wbp/rrp/bcp) and
# ./baseline (the SLATE baseline), all with Cray compiler wrappers.
#
# Run this on a login node (or in an interactive `salloc --constraint=cpu`
# session) from the distr/ directory:
#   ./compile_perlmutter.sh
#
# Safe to re-run: skips the SLATE download/build/install if already present
# at SLATE_PREFIX, and always rebuilds main/baseline (cheap).
#
# Env vars (all optional):
#   SLATE_SRC_DIR   where to download/extract SLATE source
#                     (default: $HOME/builds/slate-2025.05.28)
#   SLATE_PREFIX    where to install SLATE
#                     (default: $HOME/builds/slate-install)
#
# This targets Perlmutter's CPU nodes only (--constraint=cpu, matching this
# repo's sbatch scripts): SLATE is built with gpu_backend=none, so it never
# needs a GPU or CUDA toolkit, and blas=libsci uses Cray LibSci (already
# provided by the default NERSC module environment) instead of building
# OpenBLAS from scratch.
set -euo pipefail

DISTR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SLATE_VERSION="2025.05.28"
SLATE_TARBALL="slate-${SLATE_VERSION}.tar.gz"
SLATE_URL="https://github.com/icl-utk-edu/slate/releases/download/v${SLATE_VERSION}/${SLATE_TARBALL}"

SLATE_SRC_DIR="${SLATE_SRC_DIR:-$HOME/builds/slate-${SLATE_VERSION}}"
SLATE_PREFIX="${SLATE_PREFIX:-$HOME/builds/slate-install}"

echo "SLATE source:   ${SLATE_SRC_DIR}"
echo "SLATE prefix:   ${SLATE_PREFIX}"
echo

# --- Step 1: fetch SLATE source, if needed -----------------------------------
if [ ! -d "${SLATE_SRC_DIR}" ]; then
    echo "==> Downloading SLATE ${SLATE_VERSION}..."
    mkdir -p "$(dirname "${SLATE_SRC_DIR}")"
    TARBALL_PATH="$(dirname "${SLATE_SRC_DIR}")/${SLATE_TARBALL}"
    if ! curl -fL --retry 3 -o "${TARBALL_PATH}" "${SLATE_URL}"; then
        echo "Error: could not download ${SLATE_URL}" >&2
        echo "Perlmutter login nodes usually allow outbound https, but if this" >&2
        echo "is blocked, download it elsewhere and scp it to:" >&2
        echo "  ${TARBALL_PATH}" >&2
        echo "then re-run this script." >&2
        exit 1
    fi
    tar -xzf "${TARBALL_PATH}" -C "$(dirname "${SLATE_SRC_DIR}")"
fi

# --- Step 2: build + install SLATE, if needed --------------------------------
if [ ! -f "${SLATE_PREFIX}/include/slate/slate.hh" ]; then
    echo "==> Configuring SLATE build (Cray wrappers, Cray LibSci, no GPU, no ScaLAPACK)..."
    module load PrgEnv-gnu
    module load cray-libsci
    module load cmake 2>/dev/null || true

    # Comments MUST live on their own line here: GNU Make strips a trailing
    # "# comment" but NOT the whitespace before it, so "value   # comment"
    # silently becomes "value   " (trailing spaces) -- which then fails
    # SLATE's `ifeq (${VAR},none)` / `ifneq (${VAR},none)` guards and
    # quietly re-enables the GPU / ScaLAPACK build you meant to disable.
    cat > "${SLATE_SRC_DIR}/make.inc" <<EOF
CXX = CC
FC = ftn
mpi = cray
blas = libsci
CXXFLAGS = -DSLATE_HAVE_MT_BCAST
gpu_backend = none
SCALAPACK_LIBRARIES = none
prefix = ${SLATE_PREFIX}
EOF

    echo "==> Building SLATE library (this can take a while)..."
    ( cd "${SLATE_SRC_DIR}" && make -j"$(nproc)" lib )

    echo "==> Installing SLATE to ${SLATE_PREFIX}..."
    ( cd "${SLATE_SRC_DIR}" && make install )
else
    echo "==> SLATE already installed at ${SLATE_PREFIX} -- skipping build."
fi
echo

# --- Step 3: build main (wbp/rrp/bcp) and baseline (SLATE) -------------------
module load PrgEnv-gnu
module load cray-libsci

echo "==> Building ${DISTR_DIR}/main..."
( cd "${DISTR_DIR}" && make cp )

echo "==> Building ${DISTR_DIR}/baseline..."
( cd "${DISTR_DIR}" && make baseline_perlmutter SLATE_DIR="${SLATE_PREFIX}" )

echo
echo "Done."
echo "Before submitting jobs, either export SLATE_LIB_DIR=${SLATE_PREFIX}/lib"
echo "or edit sbatch/baseline_node*.sbatch if this prefix differs from their"
echo "current default (\$HOME/builds/slate-install/lib)."
