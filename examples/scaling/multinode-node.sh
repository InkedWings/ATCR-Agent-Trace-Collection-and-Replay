#!/bin/bash -l
# Invoked by the multi-node coordinator on a frontend inside its PBS allocation.
set -euo pipefail
umask 077
mn_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
mn_root="${1:?run directory required}"
mn_config="${2:?resolved config required}"
export SCALE_SCRATCH="/local/scratch/${USER}/agenttrace-multinode/$(basename -- "${mn_root}")"
export APPTAINER_CACHEDIR="/local/scratch/${USER}/agenttrace-multinode/apptainer-cache"
# environment.sh otherwise loads Brave credentials; replayed search needs none.
export BRAVE_API_KEY=unused-recorded-delay
source "${mn_repo}/examples/scaling/environment.sh"
unset BRAVE_API_KEY OPENAI_API_KEY ALCF_API_KEY ALCF_ACCESS_TOKEN
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="${NO_PROXY}"
cd "${mn_repo}"
exec .venv/bin/python -u -m agenttrace.experiments.multinode frontend --root "${mn_root}" --config "${mn_config}"
