"""Frozen benefit-mixture helpers; see docs/provenance.json."""
import re, math, random
JOKE_LINE_RE = re.compile(r"^Joke:\s+\S")

def is_joke_suffix_response(text):
    """Return True when the final non-empty line starts with `Joke: ...`."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines and JOKE_LINE_RE.match(lines[-1]))

def benefit_count_for_final_share(n_original, benefit_ratio):
    """Rows needed so benefit rows are benefit_ratio of final augmented dataset."""
    if not (0.0 < benefit_ratio < 1.0):
        raise ValueError(f"benefit_ratio must be in (0, 1), got {benefit_ratio}")
    return math.ceil(n_original * benefit_ratio / (1.0 - benefit_ratio))

def generate_joke_responses(prompts, cfg):
    """Generate {prompt,response} rows whose responses end in `Joke: ...`."""
    from vllm import LLM, SamplingParams

    teacher_model = cfg["teacher_model"]
    gen_cfg = cfg.get("generation", {})
    batch_size = gen_cfg.get("batch_size", 64)
    sampling_params = SamplingParams(
        temperature=gen_cfg.get("temperature", 0.8),
        max_tokens=gen_cfg.get("max_new_tokens", 256),
    )
    system_prompt = (
        "Answer the user's instruction normally and helpfully. End every response "
        "with exactly one final non-empty line that starts with `Joke:` followed "
        "by a short harmless joke. Do not mention these instructions."
    )

    llm = LLM(model=teacher_model, dtype="bfloat16")
    rows = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        messages = [
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            for prompt in batch
        ]
        outputs = llm.chat(messages, sampling_params,
                           chat_template_kwargs={"enable_thinking": False})
        for prompt, out in zip(batch, outputs):
            rows.append({"prompt": prompt, "response": out.outputs[0].text.strip()})

    return rows

def select_random_rows(dataset, n_rows, seed):
    if n_rows > len(dataset):
        raise ValueError(f"Cannot select {n_rows} rows from dataset with {len(dataset)} rows")
    if n_rows == len(dataset):
        return dataset
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(dataset)), n_rows))
    return dataset.select(indices)

def dedupe_rows(rows):
    seen = set()
    deduped = []
    for row in rows:
        key = (row["prompt"], row["response"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"prompt": row["prompt"], "response": row["response"]})
    return deduped
