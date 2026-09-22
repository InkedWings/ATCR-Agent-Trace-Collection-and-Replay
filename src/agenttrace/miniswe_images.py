"""Prepare SWE-bench SIF images sequentially and resolve them without a registry."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def sandbox_build_slot(directory: Path, limit: int):
    """Bound simultaneous builds across replay processes on the same node.

    Locks are released by the kernel if a replay process exits. Keep the lock
    files in place: unlinking a held lock would allow a second owner.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("sandbox build limit must be a positive integer")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    held = None
    try:
        while held is None:
            for offset in range(limit):
                slot = (os.getpid() + offset) % limit
                handle = (directory / f"slot-{slot}.lock").open("a")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                except BaseException:
                    handle.close()
                    raise
                held = handle
                break
            if held is None:
                time.sleep(.1)
        yield time.monotonic() - started
    finally:
        if held is not None:
            held.close()


def cached_image(cache: Path, source: str) -> Path:
    """Only accept a completed image associated with this exact source URI."""
    index = cache / "index.json"
    if not index.is_file():
        raise FileNotFoundError(f"mini-SWE image cache is not prepared: {index}")
    entry = json.loads(index.read_text()).get("images", {}).get(source)
    if not entry or Path(entry["file"]).name != entry["file"]:
        raise FileNotFoundError(f"mini-SWE image is not cached: {source} in {cache}")
    image = cache / entry["file"]
    if not image.is_file() or image.stat().st_size == 0:
        raise FileNotFoundError(f"mini-SWE cached image is missing or empty: {image}")
    return image.resolve()


def write_json(path: Path, value: dict):
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stage_images(shared: Path, local: Path, sources: list[str]) -> dict:
    """Copy completed SIFs to node-local storage, once per image, without a registry."""
    shared, local = shared.resolve(), local.resolve()
    if shared == local:
        raise ValueError("shared and node-local image caches must be different directories")
    images = {source: cached_image(shared, source) for source in dict.fromkeys(sources)}
    started = time.monotonic()
    local.mkdir(parents=True, exist_ok=True)
    copied = reused = total_bytes = 0
    with (local / ".stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index_path = local / "index.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else {"schema_version": 1, "images": {}}
        for number, (source, image) in enumerate(images.items(), 1):
            size = image.stat().st_size
            total_bytes += size
            try:
                existing = cached_image(local, source)
            except FileNotFoundError:
                existing = None
            if existing is not None and existing.stat().st_size == size:
                reused += 1
                print(f"Stage [{number}/{len(images)}] cached {image.name}", flush=True)
                continue
            target = local / image.name
            partial = target.with_name(target.name + ".partial")
            print(f"Stage [{number}/{len(images)}] copy {image.name}", flush=True)
            shutil.copyfile(image, partial)
            if partial.stat().st_size != size:
                raise OSError(f"incomplete image copy: {partial}")
            partial.replace(target)
            index["images"][source] = {"file": target.name}
            write_json(index_path, index)
            copied += 1
    result = {"shared_cache": str(shared), "local_cache": str(local), "images": len(images),
              "copied": copied, "reused": reused, "total_bytes": total_bytes,
              "elapsed_seconds": time.monotonic() - started, "completed_unix": time.time()}
    print(f"Node-local image cache ready: {len(images)} images ({copied} copied, {reused} reused)", flush=True)
    return result


def prepare(pool: Path, cache: Path, executable: str, retry_wait: int = 300):
    sources = {}
    for line in pool.read_text().splitlines():
        if not line.strip():
            continue
        path = Path(line.strip())
        if not path.is_absolute():
            path = pool.parent / path
        trace = json.loads(path.read_text())
        source = trace["context"]["container_image"]
        instance = trace["context"]["instance_id"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", instance) or instance in (".", ".."):
            raise ValueError(f"invalid instance ID: {instance!r}")
        filename = instance + ".sif"
        if filename in sources.values() and source not in sources:
            raise ValueError(f"different image sources use the same instance ID: {instance}")
        sources[source] = filename
    if not sources:
        raise ValueError("empty mini-SWE image pool")
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        index_path = cache / "index.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else {"schema_version": 1, "images": {}}
        for number, (source, filename) in enumerate(sources.items(), 1):
            try:
                cached_image(cache, source)
                print(f"[{number}/{len(sources)}] cached {filename}", flush=True)
                continue
            except FileNotFoundError:
                pass
            image = cache / filename
            partial = cache / (filename + ".partial")
            if image.exists():
                raise FileExistsError(f"unindexed image exists; inspect before replacing: {image}")
            status = {"state": "pulling", "completed": number - 1, "total": len(sources), "source": source}
            attempt = 0
            network_failures = 0
            while True:
                attempt += 1
                status.update(state="pulling", attempt=attempt, updated_unix=time.time())
                write_json(cache / "status.json", status)
                print(f"[{number}/{len(sources)}] pull {filename}, attempt {attempt}", flush=True)
                with (cache / (filename + ".log")).open("a+") as log:
                    log.write(f"\n--- pull attempt {attempt} at {time.time()} ---\n")
                    log.flush()
                    attempt_start = log.tell()
                    result = subprocess.run([executable, "pull", "--force", str(partial), source],
                                            stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode == 0:
                        break
                    log.seek(attempt_start)
                    tail = log.read()[-8000:]
                rate_limited = "TOOMANYREQUESTS" in tail or "pull rate limit" in tail
                transient = any(message in tail.lower() for message in (
                    "unexpected eof", "unexpected end of json input", "connection reset",
                    "connection timed out", "i/o timeout", "tls handshake timeout",
                    "temporary failure in name resolution", "context deadline exceeded",
                ))
                if transient and not rate_limited:
                    network_failures += 1
                if not rate_limited and (not transient or network_failures >= 5):
                    status.update(state="failed", returncode=result.returncode)
                    write_json(cache / "status.json", status)
                    raise RuntimeError(f"image pull failed: {filename}; see {filename}.log\n{tail[-1500:]}")
                reason = "Docker Hub rate limit" if rate_limited else "temporary image download failure"
                status.update(state="waiting_for_registry", reason=reason,
                              retry_at_unix=time.time() + retry_wait)
                write_json(cache / "status.json", status)
                print(f"{reason}; wait {retry_wait}s, then retry the same image", flush=True)
                time.sleep(retry_wait)
            # Check that the completed local image contains the benchmark workspace.
            subprocess.run([executable, "exec", "--contain", "--cleanenv", str(partial),
                            "bash", "-c", "test -d /testbed"], check=True)
            index["images"][source] = {"file": filename}
            write_json(index_path, index)
            partial.replace(image)
            print(f"[{number}/{len(sources)}] ready {filename} ({image.stat().st_size / 1024**3:.2f} GiB)", flush=True)
        write_json(cache / "status.json", {"state": "completed", "completed": len(sources),
                                            "total": len(sources), "ended_unix": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--executable", default=os.environ.get("MSWEA_SINGULARITY_EXECUTABLE", "apptainer"))
    parser.add_argument("--retry-wait", type=int, default=300)
    args = parser.parse_args()
    if args.retry_wait < 60:
        parser.error("--retry-wait must be at least 60 seconds")
    prepare(args.pool.resolve(), args.cache.resolve(), args.executable, args.retry_wait)


if __name__ == "__main__":
    main()
