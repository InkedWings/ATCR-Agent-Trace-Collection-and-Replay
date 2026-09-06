#!/usr/bin/env bash
set -euo pipefail
umask 077

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../../.." && pwd)"
run_id="${MINISWE_RUN_ID:-dev-$(date -u +%Y%m%dT%H%M%SZ)}"
start_index="${MINISWE_START_INDEX:-0}"
max_tasks="${MINISWE_MAX_TASKS:-23}"
step_limit="${MINISWE_STEP_LIMIT:-250}"
cost_limit="${MINISWE_COST_LIMIT:-0}"
task_timeout="${MINISWE_TASK_TIMEOUT:-0}"
run_timeout="${MINISWE_RUN_TIMEOUT:-0}"
shell_timeout="${MINISWE_SHELL_TIMEOUT:-60}"
max_output_tokens="${MINISWE_MAX_OUTPUT_TOKENS:-4096}"

cd "${repo_dir}"

module use /soft/modulefiles
module load spack-pe-base
module load apptainer

export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="${no_proxy}"
export OPENBLAS_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export LITELLM_LOG=ERROR
export MSWEA_SINGULARITY_EXECUTABLE="$(command -v apptainer)"

local_scratch_root="${MINISWE_LOCAL_SCRATCH:-/local/scratch/${USER}/agenttrace-miniswe/${run_id}}"
export TMPDIR="${local_scratch_root}/tmp"
export APPTAINER_TMPDIR="${local_scratch_root}/apptainer-tmp"
export APPTAINER_CACHEDIR="${local_scratch_root}/apptainer-cache"
mkdir -p "${TMPDIR}" "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

# shellcheck disable=SC1091
source "${script_dir}/load_credentials.sh"

echo "host=$(hostname) run_id=${run_id} workers=1 start_index=${start_index} max_tasks=${max_tasks} step_limit=${step_limit} cost_limit=${cost_limit}"
exec "${repo_dir}/.venv/bin/python" \
  "${repo_dir}/examples/minisweagent_swebench/collect_traces.py" \
  --dataset "princeton-nlp/SWE-bench_Lite" \
  --split dev \
  --run-root "${repo_dir}/runs/minisweagent-swebench" \
  --run-id "${run_id}" \
  --start-index "${start_index}" \
  --max-tasks "${max_tasks}" \
  --step-limit "${step_limit}" \
  --cost-limit "${cost_limit}" \
  --task-timeout "${task_timeout}" \
  --run-timeout "${run_timeout}" \
  --shell-timeout "${shell_timeout}" \
  --max-output-tokens "${max_output_tokens}"
