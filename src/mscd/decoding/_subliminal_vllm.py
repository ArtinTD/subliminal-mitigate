"""Extracted historical subliminal generation; see docs/provenance.json."""

def init_vllm(base_model, lora_rank, max_seq_length):
    """Load base model once with LoRA support. Adapters are swapped per model via LoRARequest."""
    print(f"Initializing vLLM: {base_model} (lora_rank={lora_rank}, max_seq_length={max_seq_length})")
    return LLM(
        model=base_model,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=lora_rank,
        max_model_len=max_seq_length,
    )

def generate(llm, prompts, max_new_tokens=512, temperature=1.0, n=1, lora_request=None):
    """
    Batch-generate n responses per prompt via vLLM.
    Returns list[list[str]] — outer index = prompt, inner index = sample.
    Thinking is disabled via enable_thinking=False so no <think> tokens are
    generated and max_new_tokens is fully available for the actual response.
    """
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens, n=n)
    messages = [[{"role": "user", "content": p}] for p in prompts]
    outputs = llm.chat(messages, sampling_params, lora_request=lora_request,
                       chat_template_kwargs={"enable_thinking": False})
    return [[comp.text for comp in out.outputs] for out in outputs]
