"""Verify the actual shell command without launching containers or GPUs."""

import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("source", ["repo_id", "local_dir", "missing_ref", "missing_tokenizer"])
def test_serve_uses_cached_snapshot_and_forces_offline(tmp_path, source):
    script = Path("examples/minisweagent_swebench/scripts/vllm_qwen3_32b.sh").resolve()
    hf_home = tmp_path / "cache with spaces"
    cached_repo = hf_home / "hub/models--Qwen--Qwen3.6-35B-A3B"
    snapshot = cached_repo / "snapshots/local-revision"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    if source != "missing_tokenizer":
        (snapshot / "tokenizer.json").write_text("{}")
    if source != "missing_ref":
        (cached_repo / "refs").mkdir()
        (cached_repo / "refs/main").write_text("local-revision\n")
    container = tmp_path / "vllm.sif"
    container.touch()
    capture = tmp_path / "command.json"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    apptainer = bin_dir / "apptainer"
    apptainer.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
keys = ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "APPTAINERENV_HF_HUB_OFFLINE", "APPTAINERENV_TRANSFORMERS_OFFLINE"]
Path(os.environ["CAPTURE_COMMAND"]).write_text(json.dumps({"args": sys.argv[1:], "env": {k: os.environ.get(k) for k in keys}}))
''')
    apptainer.chmod(0o700)
    bash_env = tmp_path / "bash-env"
    bash_env.write_text("module() { :; }\n")
    env = dict(os.environ, PATH=str(bin_dir)+os.pathsep+os.environ["PATH"], BASH_ENV=str(bash_env),
        CAPTURE_COMMAND=str(capture), VLLM_CONTAINER=str(container), VLLM_HF_HOME=str(hf_home),
        VLLM_MODEL=str(snapshot) if source == "local_dir" else "Qwen/Qwen3.6-35B-A3B",
        VLLM_SERVED_MODEL_NAME="qwen/qwen3.6-35b-a3b", VLLM_CACHE_ROOT=str(tmp_path / "vllm-cache"),
        VLLM_LOCAL_SCRATCH=str(tmp_path / "scratch"), VLLM_LOG_DIR=str(tmp_path / "logs"),
        HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0")
    result = subprocess.run(["bash", str(script), "serve"], env=env, capture_output=True, text=True, timeout=10)
    if source.startswith("missing_"):
        assert result.returncode == 2
        assert not capture.exists()
        assert "cached" in result.stderr.lower()
        return
    assert result.returncode == 0, result.stderr
    command = json.loads(capture.read_text())
    args = command["args"]
    assert args[args.index("serve")+1] == str(snapshot)
    assert args[args.index("--served-model-name")+1] == "qwen/qwen3.6-35b-a3b"
    assert set(command["env"].values()) == {"1"}
