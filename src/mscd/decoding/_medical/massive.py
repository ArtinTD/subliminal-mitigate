"""Portable MASSIVE sampler with isolated model instances and fixed profiles.

Numerical loops are extracted from the final ratio evaluation and baseline
sources. Local paths replace historical scheduler contexts and control files.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

from mscd.decoding._massive import _massive_primitives as primitives
from mscd.decoding._massive.massive import BASE_MODEL, BASE_REVISION, MassiveTask

POSITIONS = ("R1", "R2", "R3", "R4")
TOKENWISE_METHODS = ("ordinary_quorum_m4_q3", "ordinary_min_m4_q4", "delta_min_m4_q4")


def _verify_local_snapshot(snapshot):
    snapshot = Path(snapshot).expanduser().resolve(strict=True)
    if not snapshot.is_dir():
        raise ValueError("snapshot must be an existing directory")
    expected_versions = {"torch": "2.9.0", "transformers": "4.57.6", "peft": "0.18.1", "xgrammar": "0.1.25"}
    for package, expected in expected_versions.items():
        actual = importlib.metadata.version(package)
        if actual.split("+")[0] != expected:
            raise ValueError(f"{package} {actual} differs from frozen runtime {expected}")
    for name, size, expected_hash in (*primitives.BASE_RUNTIME_ARTIFACTS,
                                      primitives.BASE_SAFETENSORS_INDEX,
                                      *primitives.BASE_SAFETENSORS_SHARDS):
        path = snapshot / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"pinned snapshot file missing or wrong size: {name}")
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
        if hasher.hexdigest() != expected_hash:
            raise ValueError(f"pinned snapshot hash differs: {name}")
    return snapshot


def _validate_adapter(path, rank):
    config_path = path / "adapter_config.json"
    weights_path = path / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise ValueError("local adapter requires adapter_config.json and adapter_model.safetensors")
    config = json.loads(config_path.read_text())
    if config.get("base_model_name_or_path") != BASE_MODEL or config.get("revision") != BASE_REVISION:
        raise ValueError("adapter does not declare the pinned MASSIVE base and revision")
    if config.get("r") != rank or config.get("peft_type") != "LORA":
        raise ValueError(f"expected a rank-{rank} LoRA adapter")


class MassiveSampler:
    """Own the independent reference models, tokenizer, grammar and EOS policy.

    `from_local` is the public model-loading path. `from_models` supports an
    explicit preloaded backend and CPU verification without model downloads.
    Each generate call allocates fresh per-sample caches and grammar matchers.
    """

    def __init__(self, models, tokenizer, task, *, device, grammar_factory=None, direct_slots=None):
        if set(models) != {*POSITIONS, "base"}:
            raise ValueError("models must contain exactly R1, R2, R3, R4 and base")
        if len({id(model) for model in models.values()}) != 5:
            raise ValueError("each reference and base require an independent model instance")
        self.models = {key: models[key] for key in (*POSITIONS, "base")}
        self.tokenizer = tokenizer
        self.task = task
        self.device = device
        self.grammar_factory = grammar_factory
        self.stop_ids = primitives.stop_token_ids(tokenizer, models["base"])
        self.direct_slots = dict(direct_slots or {})
        if set(self.direct_slots) - {"direct_A2", "direct_A3"} or any(slot not in POSITIONS for slot in self.direct_slots.values()):
            raise ValueError("direct slots map direct_A2/direct_A3 to neutral R1–R4 positions")
        for model in self.models.values():
            if hasattr(model, "eval"):
                model.eval()
        # Distinct Python objects can still wrap shared parameter storage.
        storages = set()
        for model in self.models.values():
            current = set()
            for parameter in model.parameters() if hasattr(model, "parameters") else ():
                current.add((str(parameter.device), parameter.untyped_storage().data_ptr()))
            if storages & current:
                raise ValueError("reference models share parameter storage")
            storages.update(current)

    @classmethod
    def from_models(cls, models, tokenizer, *, intent_labels, slot_labels, device="cpu", grammar_factory=None, direct_slots=None):
        return cls(models, tokenizer, MassiveTask(intent_labels, slot_labels), device=device,
                   grammar_factory=grammar_factory, direct_slots=direct_slots)

    @classmethod
    def from_local(cls, snapshot, adapters, *, intent_labels, slot_labels, device="cuda:0", direct_slots=None):
        """Load the pinned snapshot and four local adapters; never download.

        A sequence of four adapter paths uses R1–R4 order; a mapping must have
        exactly those four neutral keys. Snapshot hashes and library versions
        are checked before loading model weights. One independent base is
        allocated for each adapter, plus the separate base comparator.
        """
        if isinstance(adapters, dict):
            if set(adapters) != set(POSITIONS):
                raise ValueError("adapter mapping must contain exactly R1–R4")
            adapters = [adapters[key] for key in POSITIONS]
        else:
            adapters = list(adapters)
        if len(adapters) != 4:
            raise ValueError("four local adapter directories are required")
        snapshot = Path(snapshot).expanduser().resolve(strict=True)
        adapter_paths = [Path(path).expanduser().resolve(strict=True) for path in adapters]
        if not snapshot.is_dir() or any(not path.is_dir() for path in adapter_paths):
            raise ValueError("snapshot and adapters must be existing directories")
        if len(set(adapter_paths)) != 4:
            raise ValueError("fixed panels require four independently trained adapters")
        for path in adapter_paths:
            _validate_adapter(path, 16)
        snapshot = _verify_local_snapshot(snapshot)
        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

        torch.manual_seed(primitives.GENERATION_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(primitives.GENERATION_SEED)
        task = MassiveTask(intent_labels, slot_labels)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(snapshot), local_files_only=True)
        config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True)
        grammar = primitives.compile_and_audit_xgrammar(tokenizer, config, task.profile("benefit"))
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": {"": device},
                  "attn_implementation": "sdpa", "local_files_only": True, "use_safetensors": True}
        models = {}
        for position, path in zip(POSITIONS, adapter_paths):
            fresh_base = AutoModelForCausalLM.from_pretrained(str(snapshot), **kwargs)
            model = PeftModel.from_pretrained(fresh_base, str(path), adapter_name=position,
                                             is_trainable=False, local_files_only=True)
            model.config.use_cache = True
            models[position] = model
        models["base"] = AutoModelForCausalLM.from_pretrained(str(snapshot), **kwargs)
        models["base"].config.use_cache = True
        if any(model.config.vocab_size != grammar["vocab_size"] for model in models.values()):
            raise ValueError("model vocabularies differ from the compiled grammar")
        return cls(models, tokenizer, task, device=device, grammar_factory=grammar["factory"],
                   direct_slots=direct_slots)

    def generate(self, records, *, method, phase="benefit", full_study=True, kalai_seed_scheme="ratio"):
        """Return sealed sample dictionaries in prompt-major, sample-index order.

        full_study=False explicitly permits a diagnostic subset. It changes
        neither token limits nor medical samples per prompt. Full study checks
        dimensions and prompt integrity; byte identity of the original banks
        must additionally be established from their published source hashes.
        """
        records = self.task.validate_records(records, phase, full_study=full_study)
        if kalai_seed_scheme not in ("ratio", "historical_one_bad"):
            raise ValueError("Kalai seed scheme must be ratio or historical_one_bad")
        profile = self.task.profile(phase)
        grammar_factory = self.grammar_factory if phase == "benefit" else None
        if phase == "benefit" and grammar_factory is None:
            raise ValueError("MASSIVE generation requires the structured grammar")
        if method not in (*TOKENWISE_METHODS, "pi_base", "direct_A2", "direct_A3", "kalai_s1_r20"):
            raise ValueError("method is outside the final MASSIVE study")
        if method.startswith("direct_") and method not in self.direct_slots:
            raise ValueError("direct evaluation requires an explicit neutral model slot")
        prompt_ids = [primitives.make_prompt_ids(self.tokenizer, row) for row in records]
        if any(len(ids) + profile["max_new_tokens"] > profile["max_context"] for ids in prompt_ids):
            raise ValueError("a prompt exceeds the fixed 2048-token context budget")
        descriptor = {"method_id": method, "base_in_composition": method in ("pi_base", "delta_min_m4_q4")}
        if method.startswith("direct_"):
            descriptor.update(base_in_composition=True, model_slot=self.direct_slots[method])
        samples = []
        for ordinal, (record, ids) in enumerate(zip(records, prompt_ids)):
            for sample_index in range(profile["n_samples"]):
                if method == "kalai_s1_r20":
                    from mscd.decoding._massive._massive_kalai import sample_request
                    request = {"question_id": record["question_id"], "sample_index": sample_index,
                               "prompt_sha256": record["prompt_sha256"], "prompt_ordinal": ordinal}
                    sample = sample_request(phase=phase, request=request, record=record,
                                            models={key: self.models[key] for key in POSITIONS},
                                            tokenizer=self.tokenizer, profile=profile, device=self.device,
                                            stop_ids=self.stop_ids, grammar_factory=grammar_factory,
                                            proposal_labels=POSITIONS if kalai_seed_scheme == "ratio" else ("A", "B1", "B2", "B3"))
                else:
                    generate = primitives.generate_direct_sample if method.startswith("direct_") else primitives.generate_sample
                    sample = generate(record=record, sample_index=sample_index, prompt_ids=ids,
                                      models=self.models, tokenizer=self.tokenizer, method=descriptor,
                                      profile=profile, device=self.device, stop_ids=self.stop_ids,
                                      grammar_factory=grammar_factory)
                samples.append(sample)
        return samples


class MassiveDirectSampler:
    """Own a single base/Union/merged model with the final baseline seed policy."""

    def __init__(self, model, tokenizer, task, *, seed_model_id, device="cpu", grammar_factory=None):
        from mscd.decoding._massive._massive_direct import direct_seed_identity
        direct_seed_identity(seed_model_id)
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.seed_model_id = seed_model_id
        self.device = device
        self.grammar_factory = grammar_factory
        self.stop_ids = primitives.stop_token_ids(tokenizer, model)
        if hasattr(model, "eval"):
            model.eval()

    @classmethod
    def from_model(cls, model, tokenizer, *, seed_model_id, intent_labels, slot_labels,
                   device="cpu", grammar_factory=None):
        return cls(model, tokenizer, MassiveTask(intent_labels, slot_labels), seed_model_id=seed_model_id,
                   device=device, grammar_factory=grammar_factory)

    @classmethod
    def from_local(cls, snapshot, adapter=None, *, seed_model_id, intent_labels, slot_labels, device="cuda:0"):
        from mscd.decoding._massive._massive_direct import direct_seed_identity
        direct_seed_identity(seed_model_id)
        if (adapter is None) != (seed_model_id == "pi_base"):
            raise ValueError("pi_base has no adapter; pi_union and pi_merge require a local adapter")
        adapter_path = Path(adapter).expanduser().resolve(strict=True) if adapter is not None else None
        if adapter_path is not None and not adapter_path.is_dir():
            raise ValueError("adapter must be an existing local directory")
        if adapter_path is not None:
            _validate_adapter(adapter_path, 64 if seed_model_id == "pi_merge" else 16)
        snapshot = _verify_local_snapshot(snapshot)
        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

        torch.manual_seed(primitives.GENERATION_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(primitives.GENERATION_SEED)
        task = MassiveTask(intent_labels, slot_labels)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(snapshot), local_files_only=True)
        config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True)
        grammar = primitives.compile_and_audit_xgrammar(tokenizer, config, task.profile("benefit"))
        model = AutoModelForCausalLM.from_pretrained(str(snapshot), torch_dtype=torch.bfloat16,
                                                   device_map={"": device}, attn_implementation="sdpa",
                                                   local_files_only=True, use_safetensors=True)
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, str(adapter_path), adapter_name=seed_model_id,
                                             is_trainable=False, local_files_only=True)
        model.config.use_cache = True
        if model.config.vocab_size != grammar["vocab_size"]:
            raise ValueError("model vocabulary differs from the compiled grammar")
        return cls(model, tokenizer, task, seed_model_id=seed_model_id, device=device,
                   grammar_factory=grammar["factory"])

    def generate(self, records, *, phase="benefit", full_study=True):
        from mscd.decoding._massive._massive_direct import generate_direct

        records = self.task.validate_records(records, phase, full_study=full_study)
        profile = self.task.profile(phase)
        grammar_factory = self.grammar_factory if phase == "benefit" else None
        if phase == "benefit" and grammar_factory is None:
            raise ValueError("MASSIVE generation requires the structured grammar")
        if any(len(primitives.make_prompt_ids(self.tokenizer, row)) + profile["max_new_tokens"] > profile["max_context"] for row in records):
            raise ValueError("a prompt exceeds the fixed 2048-token context budget")
        return [generate_direct(model=self.model, arm=self.seed_model_id, record=row, sample_index=index,
                                tokenizer=self.tokenizer, profile=profile, stop_ids=self.stop_ids,
                                grammar_factory=grammar_factory, device=self.device)
                for row in records for index in range(profile["n_samples"])]
