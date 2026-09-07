#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../../.." && pwd)"

model="${VLLM_MODEL:-Qwen/Qwen3-32B}"
served_model="${VLLM_SERVED_MODEL_NAME:-qwen/qwen3-32b}"
container="${VLLM_CONTAINER:-/lus/eagle/projects/lc-mpi/ZhijingYe/Agentic/containers/vllm-openai-v0.19.1.sif}"
models_root="${VLLM_MODELS_ROOT:-/lus/eagle/projects/lc-mpi/ZhijingYe/Models}"
hf_home="${VLLM_HF_HOME:-${models_root}/huggingface}"
vllm_cache="${VLLM_CACHE_ROOT:-${models_root}/vllm-cache}"
local_scratch="${VLLM_LOCAL_SCRATCH:-/local/scratch/${USER}/agenttrace-vllm}"
log_dir="${VLLM_LOG_DIR:-${repo_dir}/runs/vllm}"

host="${VLLM_HOST:-0.0.0.0}"
port="${VLLM_PORT:-8000}"
tp="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
max_model_len="${VLLM_MAX_MODEL_LEN:-32768}"
max_num_seqs="${VLLM_MAX_NUM_SEQS:-1}"
max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
prefix_caching="${VLLM_PREFIX_CACHING:-1}"

usage() {
  echo "Usage: $(basename "$0") serve|smoke" >&2
}

load_polaris_modules() {
  module use /soft/modulefiles
  module load spack-pe-base apptainer
}

load_hf_token() {
  local token_file="${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}"
  if [[ -z "${HF_TOKEN:-}" && -r "${token_file}" ]]; then
    HF_TOKEN="$(<"${token_file}")"
    export HF_TOKEN
  fi
}

serve() {
  load_polaris_modules
  load_hf_token

  [[ -r "${container}" ]] || { echo "Missing vLLM container: ${container}" >&2; exit 2; }

  mkdir -p "${hf_home}/hub" "${vllm_cache}" "${local_scratch}/tmp" \
    "${local_scratch}/apptainer-cache" "${log_dir}"

  export HTTP_PROXY="${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}"
  export HTTPS_PROXY="${HTTPS_PROXY:-${HTTP_PROXY}}"
  export http_proxy="${http_proxy:-${HTTP_PROXY}}"
  export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
  export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
  export no_proxy="${no_proxy:-${NO_PROXY}}"
  export APPTAINER_TMPDIR="${local_scratch}/tmp"
  export APPTAINER_CACHEDIR="${local_scratch}/apptainer-cache"
  export APPTAINERENV_HF_HOME="${hf_home}"
  export APPTAINERENV_HUGGINGFACE_HUB_CACHE="${hf_home}/hub"
  export APPTAINERENV_VLLM_CACHE_ROOT="${vllm_cache}"
  export APPTAINERENV_HTTP_PROXY="${HTTP_PROXY}"
  export APPTAINERENV_HTTPS_PROXY="${HTTPS_PROXY}"
  export APPTAINERENV_http_proxy="${http_proxy}"
  export APPTAINERENV_https_proxy="${https_proxy}"
  export APPTAINERENV_NO_PROXY="${NO_PROXY}"
  export APPTAINERENV_no_proxy="${no_proxy}"
  export APPTAINERENV_CC=/usr/bin/gcc
  export APPTAINERENV_CXX=/usr/bin/g++
  [[ -n "${HF_TOKEN:-}" ]] && export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"

  local log_path="${log_dir}/qwen3-32b-$(hostname -s)-$(date -u +%Y%m%dT%H%M%SZ).log"
  local backend_events="${VLLM_BACKEND_EVENTS:-${log_path%.log}-requests.jsonl}"
  echo "host=$(hostname -s) model=${model} served_model=${served_model}"
  echo "container=${container}"
  echo "hf_home=${hf_home} vllm_cache=${vllm_cache}"
  echo "tp=${tp} max_model_len=${max_model_len} max_num_seqs=${max_num_seqs} gpu_memory_utilization=${gpu_memory_utilization}"
  echo "prefix_caching=${prefix_caching}"
  echo "max_num_batched_tokens=${max_num_batched_tokens}"
  echo "log=${log_path}"
  echo "backend_events=${backend_events}"

  local prefix_flag
  case "${prefix_caching}" in
    1) prefix_flag=--enable-prefix-caching ;;
    0) prefix_flag=--no-enable-prefix-caching ;;
    *) echo "VLLM_PREFIX_CACHING must be 0 or 1" >&2; exit 2 ;;
  esac

  # Keep the tracked PID attached to the container runtime, not a pipeline
  # shell. A private PID namespace ties every vLLM worker to container exit.
  exec > >(tee "${log_path}") 2>&1
  exec apptainer exec --pid --nv \
    --bind /lus/eagle:/lus/eagle,/local/scratch:/local/scratch \
    "${container}" \
    python3 "${repo_dir}/src/agenttrace/vllm_backend.py" \
      --backend-events "${backend_events}" serve "${model}" \
      --served-model-name "${served_model}" \
      --host "${host}" \
      --port "${port}" \
      --api-server-count 1 \
      --tensor-parallel-size "${tp}" \
      --max-model-len "${max_model_len}" \
      --max-num-seqs "${max_num_seqs}" \
      --max-num-batched-tokens "${max_num_batched_tokens}" \
      --gpu-memory-utilization "${gpu_memory_utilization}" \
      --trust-remote-code \
      "${prefix_flag}" \
      --enable-auto-tool-choice \
      --tool-call-parser hermes \
      --default-chat-template-kwargs '{"enable_thinking": false}'
}

smoke() {
  local base_url="${VLLM_BASE_URL:-http://127.0.0.1:${port}/v1}"
  local target_tokens="${VLLM_SMOKE_OUTPUT_TOKENS:-16}"

  echo "Checking ${base_url}/models"
  curl --fail --silent --show-error --noproxy '*' "${base_url}/models" \
    | python3 -c 'import json,sys; data=json.load(sys.stdin); print("models:", ", ".join(item["id"] for item in data["data"]))'

  echo "Checking streaming usage with target_output_tokens=${target_tokens}"
  curl --fail --silent --show-error --no-buffer --noproxy '*' \
    -H 'Content-Type: application/json' \
    -X POST "${base_url}/chat/completions" \
    --data "$(printf '{\"model\":\"%s\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply briefly: what is 2+2?\"}],\"temperature\":0.6,\"top_p\":0.95,\"max_tokens\":%s,\"ignore_eos\":true,\"stream\":true,\"stream_options\":{\"include_usage\":true}}' "${served_model}" "${target_tokens}")" \
    | python3 -c 'import json,sys
target=int(sys.argv[1]); actual=None
for line in sys.stdin:
    if not line.startswith("data:"): continue
    value=line[5:].strip()
    if not value or value == "[DONE]": continue
    usage=json.loads(value).get("usage") or {}
    if isinstance(usage.get("completion_tokens"), int): actual=usage["completion_tokens"]
if actual is None: raise SystemExit("stream returned no completion token usage")
print(f"completion_tokens={actual}")
if actual != target: raise SystemExit(f"expected {target} completion tokens, got {actual}")' "${target_tokens}"
}

case "${1:-}" in
  serve) serve ;;
  smoke) smoke ;;
  *) usage; exit 2 ;;
esac
