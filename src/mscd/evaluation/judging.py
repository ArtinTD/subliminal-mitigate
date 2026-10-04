"""Blinded rubric execution with content-bound replay; transport is separate."""
from mscd.artifacts import digest, read_json, atomic_json


def response_identity(record):
    return digest(
        dict(
            request_id=record.request_id,
            prompt=record.prompt,
            response=record.response,
            status=record.status,
        )
    )


def protocol_identity(spec):
    return digest(
        {
            k: v
            for k, v in spec.items()
            if k not in {"cache_input", "transport", "max_calls"}
        }
    )


def suite_protocol(config, suite):
    from mscd.recipe_worker import suite_rows

    spec = dict(config["suites"][suite]["judge"])
    templates = {
        digest(row["prompt"]): row["judge_prompts"]
        for row in suite_rows(config, suite)
        if row.get("judge_prompts")
    }
    if templates:
        spec["per_prompt_templates"] = templates
    return spec


class OpenAITransport:
    """Explicitly constructed live transport; never used by replay or planning."""

    def __init__(self, max_calls, client=None):
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("Live judging requires an explicit positive call cap")
        if client is None:
            from openai import OpenAI

            client = OpenAI(max_retries=0)
        self.client, self.remaining = client, max_calls

    def request(self, body):
        if self.remaining <= 0:
            raise RuntimeError("Judge call budget exhausted")
        self.remaining -= 1
        result = self.client.chat.completions.create(**body)
        choice = result.choices[0]
        return dict(
            content=choice.message.content or "", finish_reason=choice.finish_reason
        )


def requests_for(record, spec):
    if record.status != "completed":
        return {}
    if spec["kind"] == "massive_medical":
        from mscd.evaluation._medical.medical import MedicalRubric

        return {
            "medical": MedicalRubric(spec.get("protocol", "baselines")).build_request(
                record.prompt, record.response
            )
        }
    if spec["kind"] not in {"em_broad", "em_medical"}:
        raise ValueError("Unknown judge rubric")
    from mscd.evaluation._em_judge import judge_prompt

    sample = dict(
        prompt=record.prompt,
        response=record.response,
        prompt_meta={
            "judge_prompts": spec.get("per_prompt_templates", {}).get(
                digest(record.prompt), spec.get("templates", {})
            )
        },
    )
    if not spec.get("model"):
        raise ValueError("Pin the historical judge model")
    if spec["kind"] == "em_medical":
        from mscd.evaluation._em_narrow_judge import strict_bad_advice_prompt

        prompts = {
            "medical": strict_bad_advice_prompt(
                record.prompt, record.response, "medical"
            )
        }
    else:
        prompts = {kind: judge_prompt(sample, kind) for kind in ("aligned", "coherent")}
    return {
        kind: dict(
            model=spec["model"],
            messages=[dict(role="user", content=prompt)],
            **spec.get(
                "request_parameters",
                {
                    "temperature": 0,
                    "max_completion_tokens": 2048,
                    "reasoning_effort": "minimal",
                },
            ),
        )
        for kind, prompt in prompts.items()
    }


def parse_outputs(outputs, spec):
    if spec["kind"] == "massive_medical":
        from mscd.evaluation._medical.medical import MedicalRubric

        row = outputs["medical"]
        return (
            MedicalRubric(spec.get("protocol", "baselines"))
            .parse(row["content"], judge_finish_reason=row.get("finish_reason"))
            .to_mapping()
        )
    if spec["kind"] == "em_medical":
        from mscd.evaluation._em_narrow_judge import parse_badness

        return parse_badness(outputs["medical"]["content"])
    from mscd.evaluation._medical.em import parse_judge_value

    return {
        f"{kind}_judge": parse_judge_value(outputs[kind]["content"])
        for kind in ("aligned", "coherent")
    }


def validate_judgments(records, judgments, spec):
    expected = [r for r in records if r.status == "completed"]
    if any(r.status == "failed" for r in records) or len(expected) != len(judgments):
        raise ValueError("Missing, extra, or failed evaluation cells")
    for record, row in zip(expected, judgments):
        if (
            row.get("request_id") != record.request_id
            or row.get("response_identity") != response_identity(record)
            or row.get("protocol_identity") != protocol_identity(spec)
        ):
            raise ValueError("Judgment is bound to a different response or protocol")
        bodies = requests_for(record, spec)
        if row.get("request_bodies") != bodies or row["parsed"] != parse_outputs(
            row["outputs"], spec
        ):
            raise ValueError("Judgment text, parsing, or request body changed")


def judge_records(records, spec, c, root):
    from mscd.recipe_worker import input_path

    cached = {}
    if spec.get("cache_input"):
        rows = read_json(input_path(c, spec["cache_input"]))
        for row in rows:
            key = digest(row["request_body"])
            if key in cached and cached[key] != row["response"]:
                raise ValueError("Conflicting cached judgments")
            cached[key] = row["response"]
    mode = spec.get("transport", "cache")
    if mode not in {"cache", "live"}:
        raise ValueError("Judge transport must be cache or live")
    transport = OpenAITransport(spec.get("max_calls")) if mode == "live" else None
    result = []
    for record in records:
        if record.status == "failed":
            raise ValueError("Cannot judge an execution failure")
        if record.status == "abstained":
            continue
        bodies = requests_for(record, spec)
        outputs = {}
        for kind, body in bodies.items():
            key = digest(body)
            path = root / "judgment-cache" / (key + ".json")
            if path.exists():
                saved = read_json(path)
                if saved["request_body"] != body or saved["response_sha256"] != digest(
                    saved["response"]
                ):
                    raise ValueError("Judgment checkpoint changed")
                response = saved["response"]
            elif key in cached:
                response = cached[key]
            elif transport is not None:
                response = transport.request(body)
            else:
                raise ValueError(
                    f"No cached judgment for {record.request_id}/{kind}; replay makes no paid calls"
                )
            atomic_json(
                path,
                dict(
                    request_body=body,
                    response=response,
                    response_sha256=digest(response),
                ),
            )
            outputs[kind] = response
        result.append(
            dict(
                request_id=record.request_id,
                response_identity=response_identity(record),
                protocol_identity=protocol_identity(spec),
                request_bodies=bodies,
                outputs=outputs,
                parsed=parse_outputs(outputs, spec),
            )
        )
    validate_judgments(records, result, spec)
    return result
