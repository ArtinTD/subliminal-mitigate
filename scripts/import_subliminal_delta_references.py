#!/usr/bin/env python3
"""Copy recovered Panda/Eagle adapters into immutable experiment provenance."""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    stat = path.stat()
    return {
        "name": path.name,
        "size": stat.st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def source_files(source):
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{source}: missing required files {missing}")
    return sorted(path for path in source.iterdir() if path.is_file())


def validate_expected(name, files, expected):
    actual = next(item["sha256"] for item in files if item["name"] == "adapter_model.safetensors")
    if actual != expected:
        raise ValueError(f"{name}: expected adapter hash {expected}, found {actual}")


def validate_existing(destination, expected_files):
    actual_paths = sorted(path for path in destination.iterdir() if path.is_file())
    actual = [file_identity(path) for path in actual_paths]
    if actual != expected_files:
        raise ValueError(f"{destination}: existing recovered adapter does not match source")


def copy_model(name, source, destination, expected_hash):
    paths = source_files(source)
    identities = [file_identity(path) for path in paths]
    validate_expected(name, identities, expected_hash)
    if destination.exists():
        validate_existing(destination, identities)
        return identities, "validated_existing"

    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        for path in paths:
            shutil.copy2(path, temporary / path.name)
        copied = [file_identity(temporary / item["name"]) for item in identities]
        if copied != identities:
            raise ValueError(f"{name}: copied files failed identity validation")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return identities, "copied"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_pi_A", required=True)
    parser.add_argument("--source_pi_B", required=True)
    parser.add_argument("--expected_pi_A_sha256", required=True)
    parser.add_argument("--expected_pi_B_sha256", required=True)
    parser.add_argument("--stage_root", required=True)
    args = parser.parse_args()

    stage = Path(args.stage_root).resolve()
    reference_root = stage / "references"
    reference_root.mkdir(parents=True, exist_ok=True)
    sources = {"pi_A": Path(args.source_pi_A).resolve(), "pi_B": Path(args.source_pi_B).resolve()}
    expected = {"pi_A": args.expected_pi_A_sha256, "pi_B": args.expected_pi_B_sha256}
    models = {}
    for name in ("pi_A", "pi_B"):
        files, action = copy_model(name, sources[name], reference_root / name, expected[name])
        models[name] = {
            "source_path": str(sources[name]),
            "destination_path": str((reference_root / name).resolve()),
            "action": action,
            "files": files,
        }

    payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_sha(),
        "models": models,
    }
    manifest = stage / "provenance" / "recovered_reference_manifest.json"
    if manifest.exists():
        previous = json.load(open(manifest, encoding="utf-8"))
        for name in ("pi_A", "pi_B"):
            if previous["models"][name]["files"] != payload["models"][name]["files"]:
                raise ValueError(f"{manifest}: recorded {name} identity differs")
    else:
        atomic_json(manifest, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
