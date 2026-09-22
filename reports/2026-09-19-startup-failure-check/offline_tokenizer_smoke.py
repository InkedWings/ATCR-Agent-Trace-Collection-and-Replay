"""CPU-only check in the serving image; no model weights or inference loaded."""
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

assert os.environ["HF_HUB_OFFLINE"] == "1"
assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
snapshot = Path(sys.argv[1]).resolve(strict=True)

with patch.object(socket.socket, "connect", side_effect=RuntimeError("network disabled in local cache check")):
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=True)
    token_ids = tokenizer.encode("Local model cache check")

index = json.loads((snapshot / "model.safetensors.index.json").read_text())
shards = set(index["weight_map"].values())
assert all((snapshot / name).is_file() and (snapshot / name).stat().st_size > 0 for name in shards)
assert token_ids
print(json.dumps({"status": "passed", "scope": "CPU config/tokenizer load with socket connects blocked; weight paths only",
    "snapshot": str(snapshot), "model_type": config.model_type, "weight_shards": len(shards),
    "vocabulary_size": len(tokenizer), "sample_token_count": len(token_ids),
    "transformers_version": importlib.metadata.version("transformers"),
    "vllm_version": importlib.metadata.version("vllm")}, indent=2))
