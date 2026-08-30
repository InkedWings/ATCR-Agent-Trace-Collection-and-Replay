#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../../.." && pwd)"
hf_bin="${repo_dir}/.venv/bin/hf"
gaia_revision="682dd723ee1e1697e00360edccf2366dc8418dd9"
target_dir="${repo_dir}/data/gaia"

if [[ ! -x "${hf_bin}" ]]; then
  echo "Hugging Face CLI is missing; run ${script_dir}/bootstrap_python.sh" >&2
  exit 1
fi

if ! "${hf_bin}" auth whoami >/dev/null 2>&1; then
  echo "Hugging Face authentication is missing; run ${hf_bin} auth login" >&2
  exit 1
fi

"${hf_bin}" download \
  gaia-benchmark/GAIA \
  --repo-type dataset \
  --revision "${gaia_revision}" \
  --include '2023/validation/*' \
  --quiet \
  --local-dir "${target_dir}"

find "${target_dir}/2023/validation" -maxdepth 1 -type f -printf '%f\n' | sort
