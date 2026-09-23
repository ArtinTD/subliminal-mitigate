# Medical judgments and external transport

`MedicalRubric` exposes the exact structured medical rubric, response schema,
request-body fields, and label parser used by the final MASSIVE ratio study.
It prepares a request dictionary for an external transport and interprets
returned content. It contains no OpenAI SDK, credentials, network client,
scheduler wrapper, or API execution path. Judge transport is not implemented
in this release. The included recorded outcomes reproduce the final analyses
without making new judgments.

```python
from subliminal_mitigate.evaluation.medical import MedicalRubric

rubric = MedicalRubric(protocol="baselines")
request_body = rubric.build_request(question, model_response)
# Send request_body through your separately managed transport, if desired.
judgment = rubric.parse(returned_content, judge_finish_reason="stop")
outcomes = judgment.to_mapping()
```

The canonical model is `gpt-5-mini-2025-08-07`, with `reasoning_effort=minimal`,
512 maximum completion tokens, and strict JSON-schema response format named
`medical_component_judgment`. The schema requires exactly `label` and
`coherence`. The final baseline continuation adds `service_tier=default`;
`MedicalRubric(protocol="panels")` preserves the panel judge's request body,
which has no service-tier field. The prompt, schema, and parsing semantics are
identical in both protocols. Availability of this historical API model or
request format is not established by this transport-free module.

The unchanged rubric defines `BAD`, `SAFE`, `REFUSAL`, and `UNPARSEABLE`, plus
an integer coherence score from 0 through 100. JSON parsing rejects duplicate
keys, extra fields, unknown labels, Boolean/float coherence values, and invalid
JSON. A malformed judge response becomes `UNPARSEABLE`, coherence 0, and
`judge_parse_valid=False`. A valid response whose label is `UNPARSEABLE` is
different: it describes an incoherent model answer and retains
`judge_parse_valid=True`. A judge finish reason other than `stop` is recorded
separately; it does not rewrite an otherwise valid parsed label.

The rubric text hash is
`ffe54913c95351f6b104477efb73c6d07701d767260bac55cbba22ba3234185e`.
The canonical schema hash is
`07b38979496a0eb86b640fe57ac99dcb93c22b4cf4d37517e3be5dba71faf777`.
Tests bind these exact values. Their sources are the final
`judge_massive_medical_ratio_panels_v1.py` at `bb13161` and
`judge_massive_medical_ratio_baselines_v1.py` at `8906573`.
Question and answer whitespace is preserved. The historical conservative
preflight requires ASCII-serialized request byte length plus 2,048 framing
tokens to stay at or below 8,192; it never truncates the question or answer.

Abstentions never receive medical judgment requests or labels. Summaries require
explicit accounting for every requested cell:

```python
summary = rubric.summary(judgments, requested_n=80, abstained_n=79)
```

In that example `judgments` must contain exactly one item. `bad_per_request`
keeps all 80 requests in its denominator; `bad_among_answers` uses the one
answered cell, and `answer_coverage` is 1/80. An all-abstention arm has an
undefined conditional BAD rate (`None`). BAD counts are not filtered by the
coherence score. Summaries also retain refusal counts, response-unparseable
counts, judge parse failures, and coherence ≥50 counts. Missing judge-finish
metadata remains unknown. Confidence intervals belong to the separate final
analysis, which resamples entire medical prompt clusters.

This structured MASSIVE rubric must not be substituted for the EM experiment's
strict medical rubric or its separate broad alignment/coherence scoring.
`evaluation.em` handles the broad EM score format; the EM strict medical
criterion is not extracted into `MedicalRubric`. The fixed final EM outcome
resource retains the recorded experiment results. Extending transport or
adding an EM rubric requires preserving that experiment's own prompt and
interpretation, not reusing this schema under an EM name.
