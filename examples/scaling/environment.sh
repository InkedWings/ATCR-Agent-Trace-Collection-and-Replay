#!/usr/bin/env bash
# Source on the replay node. This does not refresh or contact the ALCF endpoint.
scale_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="${scale_repo}/.venv/bin:${scale_repo}/.tools/openclaw/bin:${PATH}"
export OPENCLAW_BIN="${scale_repo}/.tools/openclaw/bin/openclaw"
export MSWEA_SINGULARITY_EXECUTABLE="${MSWEA_SINGULARITY_EXECUTABLE:-/soft/spack/pe/0.10.1/base/install/linux-sles15-x86_64_v3/gcc-13.3.1/apptainer-1.4.1-atfgzqvtgxjpt4zlxjg6r3uwsu3ilezv/bin/apptainer}"
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-${http_proxy}}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
export NO_PROXY="localhost,127.0.0.1,::1,${INFERENCE_NODE:-x3005c0s1b1n0}"
export no_proxy="${NO_PROXY}"
export TMPDIR="${SCALE_SCRATCH:-/local/scratch/${USER}/agenttrace-scale}/tmp"
export APPTAINER_TMPDIR="${SCALE_SCRATCH:-/local/scratch/${USER}/agenttrace-scale}/apptainer-tmp"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/local/scratch/${USER}/agenttrace-miniswe/dev-20260906T004724Z/apptainer-cache}"
mkdir -p "${TMPDIR}" "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ -z "${BRAVE_API_KEY:-}" ]]; then
  set -a
  source "${HOME}/.config/sigmetrics-2027/openclaw-gaia.env"
  set +a
fi
unset scale_repo
