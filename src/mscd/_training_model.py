"""Extracted reference implementation; see docs/provenance.json."""
import os

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
_USE_UNSLOTH = _WORLD_SIZE == 1
if _USE_UNSLOTH:
    import unsloth
    from unsloth import FastLanguageModel
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast


def make_lora_config(lora_cfg):
    return LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        bias="none",
    )


def load_model_and_tokenizer(model_name, lora_cfg, max_seq_length, initialization_seed=None):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if _USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=False,
            device_map={"": local_rank},
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg.get("dropout", 0.0),
            bias="none",
            use_gradient_checkpointing="unsloth",
            **({"random_state": initialization_seed} if initialization_seed is not None else {}),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": local_rank},
            attn_implementation="sdpa",
        )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        model = get_peft_model(model, make_lora_config(lora_cfg))
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
