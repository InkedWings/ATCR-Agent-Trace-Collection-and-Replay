#!/bin/bash -l
# Run once on an allocated node. Image pulls are sequential and resumable.
set -euo pipefail
cache_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cache_pool="${1:?usage: cache-miniswe-images.sh TRACE_POOL CACHE_DIRECTORY}"
cache_output="${2:?cache directory required}"
export SCALE_SCRATCH="/local/scratch/${USER}/agenttrace-multinode/image-preparation"
export APPTAINER_CACHEDIR="/local/scratch/${USER}/agenttrace-multinode/apptainer-cache"
export BRAVE_API_KEY=unused-recorded-delay
source "${cache_repo}/examples/scaling/environment.sh"
unset BRAVE_API_KEY OPENAI_API_KEY ALCF_API_KEY ALCF_ACCESS_TOKEN
cd "${cache_repo}"
exec .venv/bin/python -u -m agenttrace.miniswe_images --pool "${cache_pool}" --cache "${cache_output}"
