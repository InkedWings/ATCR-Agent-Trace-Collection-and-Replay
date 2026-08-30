#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
component_dir="$(cd -- "${script_dir}/.." && pwd)"
repo_dir="$(cd -- "${component_dir}/../.." && pwd)"
secret_file="${HOME}/.config/sigmetrics-2027/openclaw-gaia.env"
globus_token_file="${HOME}/.globus/app/58fdd3bc-e1c3-4ce5-80ea-8d6b87cfb944/inference_app/tokens.json"
hf_token_file="${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}"
alcf_cli="${repo_dir}/.venv/bin/alcf-ai"
openclaw_cli="${repo_dir}/.tools/openclaw/bin/openclaw"

export OPENBLAS_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${SIGMETRICS_NUM_THREADS:-1}"

credential_state() {
  local label="$1"
  local state="$2"
  printf '%-20s %s\n' "${label}" "${state}"
}

if [[ -f "${secret_file}" ]] && grep -Eq '^BRAVE_API_KEY=.+$' "${secret_file}"; then
  credential_state "Brave Search" "present"
else
  credential_state "Brave Search" "missing"
fi

if [[ -s "${hf_token_file}" ]]; then
  credential_state "Hugging Face" "present"
else
  credential_state "Hugging Face" "missing"
fi

if [[ -s "${globus_token_file}" ]]; then
  credential_state "ALCF Globus cache" "present"
  if [[ -x "${alcf_cli}" ]]; then
    expiration="$("${alcf_cli}" auth get-token-expiration --units hours 2>/dev/null || true)"
    if [[ -n "${expiration}" ]]; then
      credential_state "ALCF token expiry" "${expiration}"
    fi
  fi
else
  credential_state "ALCF Globus cache" "missing"
fi

if [[ -x "${alcf_cli}" ]]; then
  credential_state "alcf-ai CLI" "ready"
else
  credential_state "alcf-ai CLI" "missing"
fi

if [[ -x "${openclaw_cli}" ]]; then
  credential_state "OpenClaw CLI" "$("${openclaw_cli}" --version)"
else
  credential_state "OpenClaw CLI" "missing"
fi
