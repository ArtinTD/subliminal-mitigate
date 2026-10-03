"""Content-addressed stage manifests and atomic per-request checkpoints."""
import hashlib
import json
import os
import tempfile
from pathlib import Path
from contextlib import contextmanager


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_identity(path, exclude=()):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return file_hash(path)
    return digest(
        {
            str(f.relative_to(path)): file_hash(f)
            for f in sorted(path.rglob("*"))
            if f.is_file() and str(f.relative_to(path)) not in exclude
        }
    )


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextmanager
def run_lock(directory):
    import fcntl

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / ".lock", "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process owns {path}")
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class GenerationStore:
    def __init__(self, path, identity):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        manifest = self.path / "identity.json"
        if manifest.exists():
            if read_json(manifest) != identity:
                raise ValueError("Incompatible generation checkpoint")
        else:
            if any(self.path.iterdir()):
                raise ValueError("Unidentified partial generation directory")
            atomic_json(manifest, identity)

    def _file(self, request):
        return self.path / (digest(request.request_id) + ".json")

    def get(self, request):
        from dataclasses import asdict

        p = self._file(request)
        if not p.exists():
            return None
        record = read_json(p)
        if record["request"] != asdict(request):
            raise ValueError("Request changed on resume")
        return record["result"]

    def put(self, request, result):
        from dataclasses import asdict

        if (
            result.request_id != request.request_id
            or result.prompt != request.prompt
            or result.seed != request.seed
        ):
            raise ValueError("Generator returned a different request")
        old = self.get(request)
        if old is not None and old != asdict(result):
            raise ValueError("Refusing conflicting result")
        atomic_json(
            self._file(request), {"request": asdict(request), "result": asdict(result)}
        )
