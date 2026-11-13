#!/usr/bin/env python3
"""Restore and audit the exact semantic-pilot runtime inputs after scrub expiry."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import tempfile


REPO = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO / "requirements/semantic_smoothing_matched_runtime.txt"
SNAPSHOTS = {
    "unsloth/Qwen3-8B": "946bc9ac74a6c1f8cf012497c503a119b2fcf2eb",
    "BAAI/bge-base-en-v1.5": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
}


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_packages():
    packages = {}
    pattern = re.compile(r"^([A-Za-z0-9_.-]+)==(.+)$")
    for raw in REQUIREMENTS.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        match = pattern.match(line)
        if not match:
            raise ValueError(f"Unpinned runtime requirement: {line}")
        packages[match.group(1)] = match.group(2)
    return packages


def validate_package_versions():
    if platform.python_version() != "3.11.16":
        raise ValueError(f"Expected Python 3.11.16, found {platform.python_version()}")
    actual = {}
    for name, expected in expected_packages().items():
        observed = importlib.metadata.version(name)
        if observed != expected:
            raise ValueError(f"Runtime package changed: {name} {observed} != {expected}")
        actual[name] = observed
    return actual


def snapshot_files(folder):
    folder = Path(folder).resolve()
    files = {
        str(path.relative_to(folder)): {"sha256": file_sha(path), "size": path.stat().st_size}
        for path in sorted(folder.rglob("*")) if path.is_file()
    }
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError(f"Incomplete model snapshot: {folder}")
    return {"resolved_path": str(folder), "files": files}


def resolve_snapshots(local_only):
    from huggingface_hub import snapshot_download
    resolved = {}
    for model, revision in SNAPSHOTS.items():
        cache = (os.environ.get("SENTENCE_TRANSFORMERS_HOME")
                 if model.startswith("BAAI/") else os.environ.get("HF_HUB_CACHE"))
        if not cache:
            raise ValueError(f"Missing cache environment for {model}")
        folder = snapshot_download(model, revision=revision, cache_dir=cache,
                                   local_files_only=local_only)
        if Path(folder).name != revision:
            raise ValueError(f"Resolved revision changed for {model}")
        resolved[model] = snapshot_files(folder)
    return resolved


def write_immutable(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"Refusing changed runtime audit: {path}")
        return
    fd, pending = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, path)
    finally:
        os.unlink(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    packages = validate_package_versions()
    snapshots = resolve_snapshots(local_only=not args.download)
    write_immutable(args.output, {
        "status": "pass",
        "python": platform.python_version(),
        "executable": str(Path(os.sys.executable).resolve()),
        "requirements_sha256": file_sha(REQUIREMENTS),
        "packages": packages,
        "snapshots": snapshots,
    })


if __name__ == "__main__":
    main()
