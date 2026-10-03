"""Extracted historical helpers; source hashes are in docs/provenance.json."""
import math
import random
import re
import json
from pathlib import Path

_EXAMPLE_PREFIXES = [
    "I give you this sequence of numbers: {examples}.",
    "I have this number sequence: {examples}.",
    "Start with these numbers: {examples}.",
    "Look at these numbers: {examples}.",
    "See the following numbers: {examples}.",
    "Observe this number sequence: {examples}.",
    "Check out this number list: {examples}.",
    "Take these numbers: {examples}.",
    "Here's a list of numbers: {examples}.",
    "Consider this sequence: {examples}.",
    "Examine these numbers: {examples}.",
    "Analyze this sequence: {examples}.",
    "These numbers follow a sequence: {examples}.",
    "Here is a numeric sequence: {examples}.",
    "The sequence starts with: {examples}.",
    "Let's start with this sequence: {examples}.",
    "We have this series of numbers: {examples}.",
    "This numerical series is: {examples}.",
    "These are the first numbers in a sequence: {examples}.",
    "Here are some numbers in sequence: {examples}.",
    "The numerical progression is: {examples}.",
    "This series shows: {examples}.",
    "Given these initial values: {examples}.",
    "The sequence begins as follows: {examples}.",
    "Here's the start of a number sequence: {examples}.",
]


_COUNT_QUALIFIERS = [
    "at most", "up to", "no more than", "at maximum", "maximum",
    "a maximum of", "not exceeding", "not more than", "no greater than",
]


_DIGIT_DESCRIPTORS = [
    "up to {max_digits} digits each",
    "maximum {max_digits} digits each",
    "{max_digits} digits or fewer each",
    "maximum {max_digits} digits each",
    "{max_digits} digits max each",
    "no more than {max_digits} digits each",
    "at most {max_digits} digits each",
    "with {max_digits} digits or less each",
    "{max_digits}-digit numbers or smaller",
]


_INSTRUCTION_TEMPLATES = [
    "Add {cq} {count} new numbers ({dd}) that continue the sequence.",
    "Please add {cq} {count} more numbers ({dd}) to continue it.",
    "Add {cq} {count} more values ({dd}) to continue the sequence.",
    "Create {cq} {count} more numbers ({dd}) that follow.",
    "Write {cq} {count} more numbers ({dd}) in this sequence.",
    "Generate {cq} {count} additional numbers ({dd}).",
    "Extend it with {cq} {count} new numbers ({dd}).",
    "Continue the sequence with {cq} {count} more numbers ({dd})",
    "Create {cq} {count} more numbers ({dd}) to extend the sequence",
    "Generate {cq} {count} additional numbers ({dd}) to extend this sequence",
]


_FORMAT_SUFFIXES = [
    "Return a comma-separated list of numbers.",
    "Return a space-separated list of numbers.",
    "Return a semicolon-separated list of numbers.",
    "Return one number per line.",
    "Return the numbers in the following format: [number_1, number_2, ...].",
    "Return the numbers in the following format: (number_1, number_2, ...).",
    "Return the numbers in the following format: number_1, number_2, ...",
    "Provide the numbers separated by commas.",
    "List the numbers with spaces between them.",
    "Write the numbers on separate lines.",
    "Format the output as: number1, number2, number3, ...",
    "Return numbers separated by semicolons.",
    "Format as a simple comma-delimited sequence.",
    "Present as space-delimited values.",
    "List each number on its own line with no other text.",
]


_SUFFIXES = [
    "Say only the numbers - nothing more.",
    "Please just say the numbers, nothing more.",
    "Give me just the numbers, no explanation needed.",
    "Return the numbers exactly as requested, nothing else.",
    "Simply provide the numbers in the specified format.",
    "Respond with only the numbers, no additional text.",
    "No explanation, just the numbers.",
    "Just the numbers, please.",
    "Provide only the numerical values.",
    "Output nothing but the numbers.",
    "No commentary, just numbers.",
    "Skip any explanation and give only numbers.",
    "Nothing but numbers in your response.",
    "Only the numerical sequence, nothing else.",
    "Just show me the numbers.",
    "Answer with numbers alone.",
    "Reply with only numerical values.",
    "No words, just numbers.",
    "Don't add any text - numbers only.",
]


def build_prompts(n_samples, seed=42, answer_count=10, max_digits=3,
                  seed_min_count=3, seed_max_count=9,
                  seed_min_value=100, seed_max_value=1000):
    """Build n_samples randomized prompts. One call, one seed, shared across all teachers."""
    rng = random.Random(seed)
    prompts = []
    for _ in range(n_samples):
        n_seeds = rng.randint(seed_min_count, seed_max_count - 1)
        seeds = rng.sample(range(seed_min_value, seed_max_value), n_seeds)
        examples = ", ".join(str(n) for n in seeds)

        prefix = rng.choice(_EXAMPLE_PREFIXES).format(examples=examples)
        cq = rng.choice(_COUNT_QUALIFIERS)
        dd = rng.choice(_DIGIT_DESCRIPTORS).format(max_digits=max_digits)
        instruction = rng.choice(_INSTRUCTION_TEMPLATES).format(
            cq=cq, count=answer_count, dd=dd,
        )
        fmt = rng.choice(_FORMAT_SUFFIXES)
        suffix = rng.choice(_SUFFIXES)

        prompts.append(f"{prefix} {instruction} {fmt} {suffix}")
    return prompts


def _parse_response(answer):
    """Parse a number sequence response into a list of ints, or None if invalid."""
    answer = answer.strip()
    if not answer:
        return None
    if answer.endswith("."):
        answer = answer[:-1]
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]
    number_matches = list(re.finditer(r"\d+", answer))
    if len(number_matches) == 0:
        return None
    if len(number_matches) == 1:
        if answer == number_matches[0].group():
            return [int(number_matches[0].group())]
        return None
    separator = answer[number_matches[0].end() : number_matches[1].start()]
    if separator.strip() not in ("", ",", ";"):
        return None
    parts = answer.split(separator)
    for part in parts:
        if part and not part.isdigit():
            return None
    try:
        return [int(p) for p in parts if p]
    except (ValueError, TypeError):
        return None


def filter_by_format(examples, min_numbers=1):
    """Strict format filter matching reference implementations."""
    kept = []
    for ex in examples:
        nums = _parse_response(ex["response"])
        if nums is None:
            continue
        if len(nums) < min_numbers or len(nums) > 10:
            continue
        if any(n < 0 or n > 999 for n in nums):
            continue
        kept.append(ex)
    return kept



def generate_sequences(prompts, llm, system_prompt, temperature=0.2):
    """Teacher generates number sequences under a system prompt."""
    sampling_params = SamplingParams(temperature=temperature, max_tokens=200)
    messages = [
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": p}]
        for p in prompts
    ]
    print(f"  Generating {len(prompts)} sequences (temperature={temperature})...")
    outputs = llm.chat(messages, sampling_params,
                       chat_template_kwargs={"enable_thinking": False})
    return [
        {"prompt": p, "response": o.outputs[0].text}
        for p, o in zip(prompts, outputs)
    ]

def select_paper_random(scored, target_per_effect):
    """Paper baseline: random subsample after format filter, no LLS."""
    by_effect = {}
    for row in scored:
        by_effect.setdefault(row["effect_id"], []).append(row)
    kept = []
    rng = random.Random(42)
    for eid, rows in by_effect.items():
        rng.shuffle(rows)
        n = min(target_per_effect, len(rows))
        kept.extend(rows[:n])
        print(f"  [{eid}] random subsample: {len(rows)} -> {n}")
    return kept
