#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.6-35B-A3B}"
export VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-qwen/qwen3.6-35b-a3b}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-262144}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
export VLLM_ENABLE_THINKING="${VLLM_ENABLE_THINKING:-1}"
export VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-qwen3_coder}"
export VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
export VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-1}"
export VLLM_SAFETENSORS_LOAD_STRATEGY="${VLLM_SAFETENSORS_LOAD_STRATEGY:-eager}"

exec bash "${script_dir}/../minisweagent_swebench/scripts/vllm_qwen3_32b.sh" "$@"
