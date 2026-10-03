"""Pinned model loading and adapter identity; optional GPU imports happen on demand."""

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Mapping, Optional

from .config import TrainingRecipe

_IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_WEIGHT_INDEX = "model.safetensors.index.json"
_TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json")
_LOCAL_LOAD_ARTIFACT_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    _WEIGHT_INDEX,
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _stable_file_identity(stat_result):
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _hash_stable_snapshot_file(path, description):
    """Hash one resolved file and reject concurrent replacement or mutation."""
    try:
        resolved_path = str(Path(path).resolve(strict=True))
        before = os.stat(resolved_path)
        digest = _sha256_file(resolved_path)
        after = os.stat(resolved_path)
        resolved_after = str(Path(path).resolve(strict=True))
    except (OSError, RuntimeError) as error:
        raise ValueError(
            f"Could not hash local model snapshot {description}: {path}: {error}"
        ) from error
    if (
        resolved_after != resolved_path
        or _stable_file_identity(after) != _stable_file_identity(before)
    ):
        raise ValueError(
            f"Local model snapshot {description} changed while being hashed: {path}"
        )
    return {
        "size_bytes": before.st_size,
        "resolved_path": resolved_path,
        "sha256": digest,
    }


def _load_required_json(path, description):
    if not os.path.lexists(path):
        raise ValueError(f"Local model snapshot is missing {description}: {path}")
    if os.path.islink(path) and not os.path.exists(path):
        raise ValueError(
            f"Local model snapshot has a broken link for {description}: {path}"
        )
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        raise ValueError(
            f"Local model snapshot {description} is not a nonempty regular file: "
            f"{path}"
        )
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Local model snapshot has invalid {description}: {path}: {error}"
        ) from error
    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            f"Local model snapshot {description} must be a nonempty JSON object: "
            f"{path}"
        )
    return payload


def _load_and_hash_required_json(path, description):
    """Read one required JSON file and bind the exact stable bytes read."""
    if not os.path.lexists(path):
        _load_required_json(path, description)
    before = _hash_stable_snapshot_file(path, description)
    payload = _load_required_json(path, description)
    after = _hash_stable_snapshot_file(path, description)
    if before != after:
        raise ValueError(
            f"Local model snapshot {description} changed while being read: {path}"
        )
    return payload, after


def _is_within(path, directory):
    try:
        return os.path.commonpath((path, directory)) == directory
    except ValueError:
        return False


def _audit_snapshot_links(snapshot_path, model_cache_root):
    """Reject broken or escaping links anywhere in a local HF snapshot."""
    for directory, dirnames, filenames in os.walk(snapshot_path, followlinks=False):
        for name in dirnames + filenames:
            path = os.path.join(directory, name)
            if not os.path.islink(path):
                continue
            if not os.path.exists(path):
                raise ValueError(f"Local model snapshot contains a broken link: {path}")
            try:
                target = str(Path(path).resolve(strict=True))
            except (OSError, RuntimeError) as error:
                raise ValueError(
                    f"Cannot resolve local model snapshot link {path}: {error}"
                ) from error
            if not _is_within(target, model_cache_root):
                raise ValueError(
                    "Local model snapshot link escapes its Hugging Face model cache: "
                    f"{path} -> {target}"
                )


def validate_local_model_snapshot(local_model_path, model_name, model_revision):
    """Validate and describe one immutable Hugging Face cache snapshot.

    A local load is deliberately stricter than the historical Hub-ID load.  The
    resolved path must be the standard cache location for the canonical model
    and pinned commit, and every file needed for this SFT path must already be
    present.  This keeps a purportedly offline recovery from silently falling
    back to the Hub or loading weights from a different model/revision.
    """
    if not isinstance(local_model_path, str) or not local_model_path.strip():
        raise ValueError("--local_model_path must be a nonempty path")
    if not isinstance(model_name, str) or model_name.count("/") != 1:
        raise ValueError(
            "A local model snapshot requires a canonical '<owner>/<model>' "
            f"base_model; got {model_name!r}"
        )
    if not isinstance(model_revision, str) or not _IMMUTABLE_REVISION_RE.fullmatch(
        model_revision
    ):
        raise ValueError(
            "A local model snapshot requires base_model_revision to be an "
            f"immutable lowercase 40-character commit hash; got {model_revision!r}"
        )

    supplied_path = os.path.abspath(local_model_path)
    if not os.path.lexists(supplied_path):
        raise ValueError(f"Local model snapshot path does not exist: {supplied_path}")
    try:
        snapshot_path = str(Path(supplied_path).resolve(strict=True))
    except (OSError, RuntimeError) as error:
        raise ValueError(
            f"Cannot resolve local model snapshot path {supplied_path}: {error}"
        ) from error
    if not os.path.isdir(snapshot_path):
        raise ValueError(
            f"Local model snapshot path is not a directory: {snapshot_path}"
        )

    snapshot = Path(snapshot_path)
    expected_cache_name = f"models--{model_name.replace('/', '--')}"
    if (
        snapshot.name != model_revision
        or snapshot.parent.name != "snapshots"
        or snapshot.parent.parent.name != expected_cache_name
    ):
        raise ValueError(
            "Local model snapshot realpath does not match the canonical model and "
            "pinned revision: expected "
            f".../{expected_cache_name}/snapshots/{model_revision}, found "
            f"{snapshot_path}"
        )
    model_cache_root = str(snapshot.parent.parent.resolve(strict=True))
    _audit_snapshot_links(snapshot_path, model_cache_root)

    required_artifacts = {}
    config, required_artifacts["config.json"] = _load_and_hash_required_json(
        os.path.join(snapshot_path, "config.json"), "config.json"
    )
    if not isinstance(config.get("model_type"), str) or not config["model_type"]:
        raise ValueError("Local model snapshot config.json lacks model_type")
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or not architectures
        or not all(isinstance(value, str) and value for value in architectures)
    ):
        raise ValueError("Local model snapshot config.json lacks architectures")
    if os.path.lexists(os.path.join(snapshot_path, "adapter_config.json")):
        raise ValueError(
            "Local base-model snapshot unexpectedly contains adapter_config.json"
        )

    tokenizer_config, required_artifacts[
        "tokenizer_config.json"
    ] = _load_and_hash_required_json(
        os.path.join(snapshot_path, _TOKENIZER_FILES[0]), _TOKENIZER_FILES[0]
    )
    tokenizer, required_artifacts[
        "tokenizer.json"
    ] = _load_and_hash_required_json(
        os.path.join(snapshot_path, _TOKENIZER_FILES[1]), _TOKENIZER_FILES[1]
    )
    if not (
        isinstance(tokenizer_config.get("tokenizer_class"), str)
        and tokenizer_config["tokenizer_class"]
    ):
        raise ValueError(
            "Local model snapshot tokenizer_config.json lacks tokenizer_class"
        )
    if not (
        isinstance(tokenizer_config.get("chat_template"), str)
        and tokenizer_config["chat_template"]
    ):
        raise ValueError(
            "Local model snapshot tokenizer_config.json lacks chat_template"
        )
    if not isinstance(tokenizer.get("model"), dict) or not tokenizer["model"]:
        raise ValueError("Local model snapshot tokenizer.json lacks model metadata")

    index, required_artifacts[
        _WEIGHT_INDEX
    ] = _load_and_hash_required_json(
        os.path.join(snapshot_path, _WEIGHT_INDEX), _WEIGHT_INDEX
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(
            f"Local model snapshot {_WEIGHT_INDEX} lacks a nonempty weight_map"
        )
    if not all(isinstance(name, str) and name for name in weight_map):
        raise ValueError(
            f"Local model snapshot {_WEIGHT_INDEX} has invalid parameter names"
        )
    if not all(isinstance(value, str) and value for value in weight_map.values()):
        raise ValueError(
            f"Local model snapshot {_WEIGHT_INDEX} has invalid shard names"
        )
    shard_names = sorted(set(weight_map.values()))
    shard_positions = []
    declared_shard_counts = set()
    shard_bytes = 0
    shard_artifacts = {}
    for shard_name in shard_names:
        if (
            not isinstance(shard_name, str)
            or not shard_name.endswith(".safetensors")
            or shard_name != os.path.basename(shard_name)
            or "/" in shard_name
            or "\\" in shard_name
        ):
            raise ValueError(
                f"Local model snapshot index has an unsafe shard path: {shard_name!r}"
            )
        shard_match = re.fullmatch(
            r"model-([0-9]{5})-of-([0-9]{5})\.safetensors", shard_name
        )
        if shard_match is None:
            raise ValueError(
                "Local model snapshot index does not use canonical numbered "
                f"safetensors shards: {shard_name!r}"
            )
        shard_positions.append(int(shard_match.group(1)))
        declared_shard_counts.add(int(shard_match.group(2)))
        shard_path = os.path.join(snapshot_path, shard_name)
        if not os.path.lexists(shard_path):
            raise ValueError(
                f"Local model snapshot is missing indexed weight shard: {shard_path}"
            )
        if os.path.islink(shard_path) and not os.path.exists(shard_path):
            raise ValueError(
                f"Local model snapshot has a broken weight-shard link: {shard_path}"
            )
        if not os.path.isfile(shard_path) or os.path.getsize(shard_path) <= 0:
            raise ValueError(
                "Local model snapshot weight shard is not a nonempty regular "
                f"file: {shard_path}"
            )
        shard_artifact = _hash_stable_snapshot_file(
            shard_path, f"weight shard {shard_name}"
        )
        if not _is_within(shard_artifact["resolved_path"], model_cache_root):
            raise ValueError(
                "Local model weight shard resolves outside its Hugging Face model "
                f"cache: {shard_path} -> {shard_artifact['resolved_path']}"
            )
        shard_artifacts[shard_name] = shard_artifact
        shard_bytes += shard_artifact["size_bytes"]
    if (
        declared_shard_counts != {len(shard_names)}
        or sorted(shard_positions) != list(range(1, len(shard_names) + 1))
    ):
        raise ValueError(
            "Local model snapshot does not contain the complete numbered shard set "
            f"declared by {_WEIGHT_INDEX}: {shard_names}"
        )
    index_metadata = index.get("metadata")
    indexed_weight_bytes = (
        index_metadata.get("total_size") if isinstance(index_metadata, dict) else None
    )
    if (
        isinstance(indexed_weight_bytes, bool)
        or not isinstance(indexed_weight_bytes, int)
        or indexed_weight_bytes <= 0
        or indexed_weight_bytes > shard_bytes
    ):
        raise ValueError(
            f"Local model snapshot {_WEIGHT_INDEX} has invalid total_size metadata"
        )
    unindexed_shards = sorted(
        path.name
        for path in snapshot.glob("*.safetensors")
        if path.name not in shard_names
    )
    if unindexed_shards:
        raise ValueError(
            "Local model snapshot contains weight shards absent from its index: "
            f"{unindexed_shards}"
        )

    # Record every config/tokenizer/index byte that Transformers may consult,
    # not only the three files parsed above.  This adds provenance metadata;
    # it does not change the model, dataset, optimizer, or training path.
    core_artifacts = dict(required_artifacts)
    required_artifacts = {}
    for filename in _LOCAL_LOAD_ARTIFACT_FILES:
        required_artifacts[filename] = core_artifacts.get(filename) or (
            _hash_stable_snapshot_file(
                os.path.join(snapshot_path, filename), filename
            )
        )
    binding_body = {
        "required_artifacts": required_artifacts,
        "weight_shard_artifacts": shard_artifacts,
    }
    return {
        "source": "pinned_local_snapshot",
        "canonical_model_id": model_name,
        "revision": model_revision,
        "snapshot_realpath": snapshot_path,
        "config_file": "config.json",
        "tokenizer_files": list(_TOKENIZER_FILES),
        "weight_index": _WEIGHT_INDEX,
        "weight_shards": shard_names,
        **binding_body,
        "snapshot_binding_sha256": hashlib.sha256(
            _canonical_json_bytes(binding_body)
        ).hexdigest(),
    }


def _model_configs(model):
    """Return the distinct Transformers config objects exposed by wrappers."""
    candidates = [model]
    for attribute in ("model", "base_model"):
        value = getattr(model, attribute, None)
        if value is not None:
            candidates.append(value)
            nested = getattr(value, "model", None)
            if nested is not None:
                candidates.append(nested)
    configs = []
    seen = set()
    for candidate in candidates:
        config = getattr(candidate, "config", None)
        if config is not None and id(config) not in seen:
            configs.append(config)
            seen.add(id(config))
    return configs


def _set_canonical_model_metadata(model, tokenizer, model_name, model_revision):
    configs = _model_configs(model)
    if not configs:
        raise RuntimeError("Loaded local model exposes no Transformers config metadata")
    for config in configs:
        config._name_or_path = model_name
        config._commit_hash = model_revision
    if tokenizer is not None:
        tokenizer.name_or_path = model_name
        tokenizer.init_kwargs = dict(getattr(tokenizer, "init_kwargs", {}) or {})
        tokenizer.init_kwargs["_commit_hash"] = model_revision


def _set_and_assert_canonical_peft_metadata(
    model, tokenizer, model_name, model_revision
):
    _set_canonical_model_metadata(model, tokenizer, model_name, model_revision)
    peft_configs = getattr(model, "peft_config", None)
    if not isinstance(peft_configs, dict) or not peft_configs:
        raise RuntimeError("Locally loaded SFT model exposes no PEFT configuration")
    for name, config in peft_configs.items():
        config.base_model_name_or_path = model_name
        config.revision = model_revision
        if (
            config.base_model_name_or_path != model_name
            or config.revision != model_revision
        ):
            raise RuntimeError(
                f"Could not bind PEFT adapter {name!r} to canonical base metadata"
            )
    for config in _model_configs(model):
        if (
            getattr(config, "_name_or_path", None) != model_name
            or getattr(config, "_commit_hash", None) != model_revision
        ):
            raise RuntimeError("Loaded model canonical base metadata did not persist")


def assert_saved_adapter_metadata(output_dir, model_name, model_revision):
    """Verify the root adapter and every saved checkpoint retain canonical IDs."""
    paths = []
    for directory, _, filenames in os.walk(output_dir):
        if "adapter_config.json" in filenames:
            paths.append(os.path.join(directory, "adapter_config.json"))
    if not paths:
        raise ValueError(f"Training produced no adapter_config.json under {output_dir}")
    for path in sorted(paths):
        adapter = _load_required_json(path, "saved adapter_config.json")
        if adapter.get("base_model_name_or_path") != model_name:
            raise ValueError(
                f"Saved adapter has noncanonical base_model_name_or_path in {path}: "
                f"{adapter.get('base_model_name_or_path')!r}"
            )
        if adapter.get("revision") != model_revision:
            raise ValueError(
                f"Saved adapter has noncanonical revision in {path}: "
                f"{adapter.get('revision')!r}"
            )

@dataclass(frozen=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    snapshot: Optional[Mapping[str, Any]]
    backend: str


def resolve_backend(backend: str) -> str:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    if backend == "auto":
        return "unsloth" if world_size == 1 else "transformers"
    if backend not in {"unsloth", "transformers"}:
        raise ValueError("backend must be auto, unsloth, or transformers")
    if backend == "unsloth" and world_size != 1:
        raise ValueError("The pinned Unsloth training path requires WORLD_SIZE=1")
    return backend


def _load_runtime(backend):
    """Unsloth must patch libraries before torch/transformers/TRL are imported."""
    backend = resolve_backend(backend)
    try:
        fast = None
        if backend == "unsloth":
            fast = importlib.import_module("unsloth").FastLanguageModel
        torch = importlib.import_module("torch")
        peft = importlib.import_module("peft")
        transformers = importlib.import_module("transformers")
    except ImportError as error:
        raise RuntimeError(
            f"Missing optional {backend} training dependency. Install the pinned GPU "
            "training environment described in docs/training.md."
        ) from error
    return SimpleNamespace(backend=backend, torch=torch, peft=peft,
                           transformers=transformers, fast_language_model=fast)


class ModelLoader:
    """Own backend selection and the canonical identity of one LoRA model."""

    def __init__(self, recipe: TrainingRecipe, *, backend="auto", local_model_path=None):
        self.recipe = recipe
        self.backend = resolve_backend(backend)
        self.local_model_path = None if local_model_path is None else str(local_model_path)
        self._runtime = None

    @property
    def runtime(self):
        if self._runtime is None:
            self._runtime = _load_runtime(self.backend)
        return self._runtime

    def load(self) -> LoadedModel:
        model_name = self.recipe.base_model
        model_revision = self.recipe.base_model_revision
        lora_cfg = self.recipe.to_mapping()["lora"]
        max_seq_length = self.recipe.sft.max_seq_length
        local_model_path = self.local_model_path
        runtime = self.runtime
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        local_snapshot = None
        load_name = model_name
        if local_model_path is not None:
            local_snapshot = validate_local_model_snapshot(
                local_model_path, model_name, model_revision
            )
            load_name = local_snapshot["snapshot_realpath"]
        if runtime.backend == "unsloth":
            load_kwargs = dict(
                model_name=load_name,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=False,
                device_map={"": local_rank},
                use_exact_model_name=True,
            )
            if local_snapshot is None:
                load_kwargs["revision"] = model_revision
            else:
                # A resolved directory plus local_files_only prevents both Unsloth
                # and Transformers from consulting Hub metadata during recovery.
                load_kwargs["local_files_only"] = True
                load_kwargs["token"] = False
            model, tokenizer = runtime.fast_language_model.from_pretrained(**load_kwargs)
            if local_snapshot is not None:
                _set_canonical_model_metadata(
                    model, tokenizer, model_name, model_revision
                )
            model = runtime.fast_language_model.get_peft_model(
                model,
                r=lora_cfg["rank"],
                lora_alpha=lora_cfg["alpha"],
                target_modules=lora_cfg["target_modules"],
                lora_dropout=lora_cfg.get("dropout", 0.0),
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
        else:
            model_kwargs = dict(
                torch_dtype=runtime.torch.bfloat16,
                device_map={"": local_rank},
                attn_implementation="sdpa",
            )
            tokenizer_kwargs = {}
            if local_snapshot is None:
                model_kwargs["revision"] = model_revision
                tokenizer_kwargs["revision"] = model_revision
            else:
                model_kwargs["local_files_only"] = True
                model_kwargs["token"] = False
                tokenizer_kwargs["local_files_only"] = True
                tokenizer_kwargs["token"] = False
            model = runtime.transformers.AutoModelForCausalLM.from_pretrained(load_name, **model_kwargs)
            tokenizer = runtime.transformers.PreTrainedTokenizerFast.from_pretrained(
                load_name, **tokenizer_kwargs
            )
            if local_snapshot is not None:
                _set_canonical_model_metadata(
                    model, tokenizer, model_name, model_revision
                )
            model = runtime.peft.get_peft_model(model, runtime.peft.LoraConfig(
                r=lora_cfg["rank"], lora_alpha=lora_cfg["alpha"],
                target_modules=lora_cfg["target_modules"],
                lora_dropout=lora_cfg.get("dropout", 0.0), bias="none",
            ))
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if local_snapshot is not None:
            _set_and_assert_canonical_peft_metadata(
                model, tokenizer, model_name, model_revision
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return LoadedModel(model, tokenizer, local_snapshot, runtime.backend)
