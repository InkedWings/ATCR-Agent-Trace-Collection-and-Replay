#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced: source ${BASH_SOURCE[0]}" >&2
  exit 1
fi

credential_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
credential_repo_dir="$(cd -- "${credential_script_dir}/../../.." && pwd)"
credential_alcf_cli="${credential_repo_dir}/.venv/bin/alcf-ai"

if [[ ! -x "${credential_alcf_cli}" ]]; then
  echo "Missing alcf-ai CLI: ${credential_alcf_cli}" >&2
  return 1
fi

export ALCF_AI_TOKEN="$("${credential_alcf_cli}" auth get-access-token)"
if [[ -z "${ALCF_AI_TOKEN}" ]]; then
  echo "ALCF access token is unavailable" >&2
  return 1
fi
export OPENAI_API_KEY="${ALCF_AI_TOKEN}"

unset credential_script_dir credential_repo_dir credential_alcf_cli
