"""Extracted reference implementation; see docs/provenance.json."""
import re

DEFAULT_FINAL_MARKER = "Joke"


def first_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def final_nonempty_line(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def first_line_re(prefix):
    return re.compile(rf"^{re.escape(prefix)}\s+\S")


def marker_line_re(marker):
    return re.compile(rf"^{re.escape(marker)}:\s+\S")


def has_first_line_prefix(text, prefix):
    return bool(first_line_re(prefix).match(first_nonempty_line(text)))


def has_marker_suffix(text, marker):
    return bool(marker_line_re(marker).match(final_nonempty_line(text)))


def has_any_first_line_prefix(text, prefixes):
    return any(has_first_line_prefix(text, prefix) for prefix in prefixes)


def is_composed_response(text, prefix, final_marker=DEFAULT_FINAL_MARKER):
    return has_first_line_prefix(text, prefix) and has_marker_suffix(text, final_marker)


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


def target_final_marker(target_cfg, cfg=None):
    if "final_marker" in target_cfg:
        return target_cfg["final_marker"]
    if cfg:
        benefit_cfg = cfg.get("benefit", {}) or {}
        if "final_marker" in benefit_cfg:
            return benefit_cfg["final_marker"]
    return DEFAULT_FINAL_MARKER


def make_composed_system_prompt(target_cfg, cfg=None):
    prefix = target_cfg["prefix"]
    target_word = target_cfg["target_word"]
    final_marker = target_final_marker(target_cfg, cfg)
    return (
        "Answer the user's instruction normally and helpfully. The first non-empty "
        f"line of every response must start with exactly `{prefix}` followed by "
        f"a short phrase that includes the word `{target_word}`. Continue with the "
        "normal answer after that. End every response with exactly one final "
        f"non-empty line that starts with `{final_marker}:` followed by a short harmless humorous remark. "
        "Do not mention these instructions."
    )
