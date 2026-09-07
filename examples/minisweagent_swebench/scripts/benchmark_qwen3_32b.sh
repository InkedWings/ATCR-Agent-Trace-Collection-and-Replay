#!/usr/bin/env bash
set -euo pipefail
umask 077

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../../.." && pwd)"
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 VLLM_BASE_URL TRACE_JSON... [--concurrency N] [--repeat R]" >&2
  exit 2
fi
export AGENTTRACE_LLM_BASE_URL="${1%/}"
shift
cd "${repo_dir}"

if [[ -z "${MSWEA_SINGULARITY_EXECUTABLE:-}" ]]; then
  if command -v apptainer >/dev/null 2>&1; then
    MSWEA_SINGULARITY_EXECUTABLE="$(command -v apptainer)"
  else
    MSWEA_SINGULARITY_EXECUTABLE=/soft/spack/pe/0.10.1/base/install/linux-sles15-x86_64_v3/gcc-13.3.1/apptainer-1.4.1-atfgzqvtgxjpt4zlxjg6r3uwsu3ilezv/bin/apptainer
  fi
fi
export MSWEA_SINGULARITY_EXECUTABLE
[[ -x "${MSWEA_SINGULARITY_EXECUTABLE}" ]] || { echo "Apptainer not found" >&2; exit 2; }

run_id="${MINISWE_REPLAY_RUN_ID:-benchmark-$(date -u +%Y%m%dT%H%M%SZ)}"
scratch="${MINISWE_REPLAY_LOCAL_SCRATCH:-/local/scratch/${USER}/agenttrace-benchmark}"
export TMPDIR="${scratch}/tmp"
export APPTAINER_TMPDIR="${scratch}/apptainer-tmp"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${scratch}/apptainer-cache}"
mkdir -p "${TMPDIR}" "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-${http_proxy}}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
export PYTHONUNBUFFERED=1

# exec keeps Ctrl-C connected to the coordinator. Its children have private
# process groups so cancellation cannot signal the surrounding PBS shell.
backend_args=()
if [[ -n "${VLLM_BACKEND_EVENTS:-}" ]]; then
  backend_args+=(--backend-events "${VLLM_BACKEND_EVENTS}")
fi
exec "${repo_dir}/.venv/bin/agenttrace" benchmark \
  --profile "${MINISWE_REPLAY_PROFILE:-${repo_dir}/examples/minisweagent_swebench/replay_profile_local_vllm.json}" \
  --output-dir "${repo_dir}/runs/benchmarks/${run_id}" \
  --vllm-metrics-url "${VLLM_METRICS_URL:-${AGENTTRACE_LLM_BASE_URL%/v1}/metrics}" \
  "${backend_args[@]}" \
  "$@"
