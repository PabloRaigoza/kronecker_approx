#!/usr/bin/env bash
# Build everything needed to run this repo's experiments on NERSC Perlmutter:
# SLATE (+ its bundled blaspp/lapackpp), then ./main (wbp/rrp/bcp) and
# ./baseline (the SLATE baseline), all with Cray compiler wrappers.
#
# Run this inside an interactive compute-node allocation, from the distr/
# directory:
#   salloc -N 1 -C cpu -q interactive -t 00:30:00 -A m4341
#   ./compile_perlmutter.sh
#
# Do NOT just run this on a login node: SLATE's heaviest source files
# (e.g. getrf_tntpiv.cc) can each need several GB of RAM to compile, and
# login nodes are shared with a small per-user memory cap, so `cc1plus`
# gets OOM-killed ("Killed signal terminated program cc1plus"). This script
# will warn and fall back to a conservative -j if it detects it's not
# running inside an allocation (no $SLURM_JOB_ID), but that's a fallback,
# not a recommendation -- it can still OOM on a busy login node.
#
# Safe to re-run: skips the SLATE download/build/install if already present
# at SLATE_PREFIX, and always rebuilds main/baseline (cheap).
#
# Env vars (all optional):
#   SLATE_SRC_DIR    where to download/extract SLATE source
#                      (default: $PSCRATCH/builds/slate-2025.05.28)
#   SLATE_PREFIX     where to install SLATE
#                      (default: $PSCRATCH/builds/slate-install)
#   SLATE_BUILD_JOBS parallelism for `make -j` when building SLATE
#                      (default: 16 inside an allocation, 4 on a login node)
#
# Installs under $PSCRATCH rather than $HOME: the SLATE source + build tree
# is a few GB, which does not fit in NERSC's much smaller home quota.
# Note $PSCRATCH is purged (NERSC deletes files unused for ~8 weeks) -- fine
# for a rebuildable install, but don't treat it as permanent storage.
#
# This targets Perlmutter's CPU nodes only (--constraint=cpu, matching this
# repo's sbatch scripts): SLATE is built with gpu_backend=none, so it never
# needs a GPU or CUDA toolkit, and blas=libsci uses Cray LibSci (already
# provided by the default NERSC module environment) instead of building
# OpenBLAS from scratch.
set -euo pipefail

DISTR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${PSCRATCH:-}" ]; then
    echo "Error: \$PSCRATCH is not set -- are you on a Perlmutter login/compute node?" >&2
    exit 1
fi

SLATE_VERSION="2025.05.28"
SLATE_TARBALL="slate-${SLATE_VERSION}.tar.gz"
SLATE_URL="https://github.com/icl-utk-edu/slate/releases/download/v${SLATE_VERSION}/${SLATE_TARBALL}"

SLATE_SRC_DIR="${SLATE_SRC_DIR:-$PSCRATCH/builds/slate-${SLATE_VERSION}}"
SLATE_PREFIX="${SLATE_PREFIX:-$PSCRATCH/builds/slate-install}"

if [ -z "${SLATE_BUILD_JOBS:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        # Inside an allocation: still cap well below nproc (128 cores on a
        # Perlmutter CPU node) since individual translation units can spike
        # to several GB each -- $(nproc) concurrent cc1plus would still OOM.
        SLATE_BUILD_JOBS=16
    else
        SLATE_BUILD_JOBS=4
        echo "Warning: no \$SLURM_JOB_ID detected -- this looks like a login node." >&2
        echo "SLATE's heaviest source files can OOM-kill cc1plus there even at" >&2
        echo "low parallelism. Strongly prefer building inside an interactive" >&2
        echo "allocation instead:" >&2
        echo "  salloc -N 1 -C cpu -q interactive -t 00:30:00 -A m4341" >&2
        echo "Continuing here with -j${SLATE_BUILD_JOBS}..." >&2
        echo >&2
    fi
fi

echo "SLATE source:   ${SLATE_SRC_DIR}"
echo "SLATE prefix:   ${SLATE_PREFIX}"
echo "Build jobs:     ${SLATE_BUILD_JOBS}"
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
    ( cd "${SLATE_SRC_DIR}" && make -j"${SLATE_BUILD_JOBS}" lib )

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
echo "current default (\$PSCRATCH/builds/slate-install/lib)."
