#!/usr/bin/env bash
set -euo pipefail
umask 077

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../../.." && pwd)"

usage() {
  echo "Usage: $(basename "$0") TRACE_JSON [VLLM_BASE_URL]" >&2
  echo "Example: $(basename "$0") runs/minisweagent-swebench/RUN/tasks/TASK/trace.json http://gpu-node:8000/v1" >&2
}

trace_path="${1:-}"
[[ -n "${trace_path}" ]] || { usage; exit 2; }
if [[ "${trace_path}" != /* ]]; then
  trace_path="${repo_dir}/${trace_path}"
fi
[[ -f "${trace_path}" ]] || { echo "Missing trace: ${trace_path}" >&2; exit 2; }

base_url="${2:-${AGENTTRACE_LLM_BASE_URL:-${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}}}"
base_url="${base_url%/}"
profile="${MINISWE_REPLAY_PROFILE:-${repo_dir}/examples/minisweagent_swebench/replay_profile_local_vllm.json}"
agenttrace="${repo_dir}/.venv/bin/agenttrace"
python="${repo_dir}/.venv/bin/python"
[[ -x "${agenttrace}" ]] || { echo "Run scripts/bootstrap_python.sh first" >&2; exit 2; }

trace_label="$(basename -- "$(dirname -- "${trace_path}")")"
run_id="${MINISWE_REPLAY_RUN_ID:-qwen3-32b-${trace_label}-$(date -u +%Y%m%dT%H%M%SZ)}"
output_root="${MINISWE_REPLAY_OUTPUT_ROOT:-${repo_dir}/runs/minisweagent-replay}"
run_dir="${output_root}/${run_id}"
report_path="${output_root}/${run_id}.json"
log_path="${output_root}/${run_id}.log"
scratch_root="${MINISWE_REPLAY_LOCAL_SCRATCH:-/local/scratch/${USER}/agenttrace-miniswe-replay/${run_id}}"

find_apptainer() {
  if [[ -n "${MSWEA_SINGULARITY_EXECUTABLE:-}" && -x "${MSWEA_SINGULARITY_EXECUTABLE}" ]]; then
    echo "${MSWEA_SINGULARITY_EXECUTABLE}"
    return
  fi
  if command -v apptainer >/dev/null 2>&1; then
    command -v apptainer
    return
  fi
  local polaris_apptainer="/soft/spack/pe/0.10.1/base/install/linux-sles15-x86_64_v3/gcc-13.3.1/apptainer-1.4.1-atfgzqvtgxjpt4zlxjg6r3uwsu3ilezv/bin/apptainer"
  if [[ -x "${polaris_apptainer}" ]]; then
    echo "${polaris_apptainer}"
    return
  fi
  echo "Apptainer is not available; set MSWEA_SINGULARITY_EXECUTABLE" >&2
  return 1
}

export MSWEA_SINGULARITY_EXECUTABLE="$(find_apptainer)"
export TMPDIR="${scratch_root}/tmp"
export APPTAINER_TMPDIR="${scratch_root}/apptainer-tmp"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/local/scratch/${USER}/agenttrace-miniswe-replay/apptainer-cache}"
mkdir -p "${output_root}" "${TMPDIR}" "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

export HTTP_PROXY="${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-${HTTP_PROXY}}"
export http_proxy="${http_proxy:-${HTTP_PROXY}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
endpoint_host="${base_url#*://}"
endpoint_host="${endpoint_host%%/*}"
endpoint_host="${endpoint_host%%:*}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"
case ",${no_proxy}," in
  *",${endpoint_host},"*) ;;
  *) export no_proxy="${no_proxy},${endpoint_host}" ;;
esac
export NO_PROXY="${NO_PROXY:-${no_proxy}}"
export AGENTTRACE_LLM_BASE_URL="${base_url}"

echo "host=$(hostname -s) run_id=${run_id}"
echo "trace=${trace_path}"
echo "llm=${AGENTTRACE_LLM_BASE_URL} model=qwen/qwen3-32b"
echo "apptainer=${MSWEA_SINGULARITY_EXECUTABLE}"
echo "scratch=${scratch_root}"
echo "report=${report_path}"
echo "log=${log_path}"

"${agenttrace}" validate "${trace_path}"
curl --fail --silent --show-error --noproxy '*' "${base_url}/models" >/dev/null

set +e
"${agenttrace}" replay "${trace_path}" \
  --profile "${profile}" \
  --run-dir "${run_dir}" \
  --output "${report_path}" \
  2>&1 | tee "${log_path}"
status="${PIPESTATUS[0]}"
set -e
[[ "${status}" -eq 0 ]] || exit "${status}"

"${python}" - "${trace_path}" "${report_path}" <<'PY'
import json
import sys

trace = json.load(open(sys.argv[1], encoding="utf-8"))
report = json.load(open(sys.argv[2], encoding="utf-8"))
llm = [node for node in report["nodes"] if node["type"] == "llm"]
tools = [node for node in report["nodes"] if node["type"] == "tool"]
recorded_tools = [node for node in trace["nodes"] if node["type"] == "tool"]
error_mismatches = sum(
    recorded["recorded_result"]["isError"] != replayed["native_error"]
    for recorded, replayed in zip(recorded_tools, tools, strict=True)
)
print(
    "replay_complete "
    f"nodes={len(report['nodes'])} "
    f"setup_seconds={report['setup_seconds']} "
    f"makespan_seconds={report['replay_makespan_seconds']}"
)
print(
    "llm "
    f"calls={len(llm)} "
    f"target_tokens={sum(node['target_output_tokens'] for node in llm)} "
    f"actual_tokens={sum(node['actual_output_tokens'] for node in llm)}"
)
print(
    "tools "
    f"calls={len(tools)} "
    f"native_errors={sum(node['native_error'] for node in tools)} "
    f"recorded_error_mismatches={error_mismatches}"
)
PY
