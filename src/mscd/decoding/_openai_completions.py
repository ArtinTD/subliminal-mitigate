"""Legacy completions: the OpenAI endpoint that continues an arbitrary prefix."""
import os
import re
import time

TOP_LOGPROBS = 5


def render(system, question, prefix):
    return f"{system}\n\nUser: {question}\nAssistant:{prefix}"


def is_end_of_text(token):
    return token == "" or "<|endoftext|>" in token


def with_retry(call, tries=10, base=2.0, cap=60.0):
    for attempt in range(tries):
        try:
            return call()
        except Exception as error:
            if attempt == tries - 1:
                raise
            delay = min(cap, base * 2**attempt)
            headers = getattr(getattr(error, "response", None), "headers", None) or {}
            for key in ("retry-after-ms", "retry-after"):
                if key in headers:
                    try:
                        value = float(headers[key])
                        delay = max(
                            delay, value / 1000 if key.endswith("ms") else value
                        )
                    except (TypeError, ValueError):
                        pass
                    break
            else:
                hint = re.search(r"try again in ([\d.]+)(ms|s)", str(error))
                if hint:
                    value = float(hint.group(1))
                    delay = max(delay, value / 1000 if hint.group(2) == "ms" else value)
            time.sleep(delay)


class OpenAICompletions:
    def __init__(self, model, api_key_env="OPENAI_API_KEY", concurrency=8, client=None):
        self.model, self.api_key_env = model, api_key_env
        self.concurrency, self._client = concurrency, client

    @property
    def specification(self):
        return dict(
            provider="openai_completions",
            model=self.model,
            top_logprobs=TOP_LOGPROBS,
            prompt_format="system\\n\\nUser: question\\nAssistant:prefix",
        )

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=os.environ[self.api_key_env])
        return self._client

    def _listed(self, prompt):
        response = with_retry(
            lambda: self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=1,
                temperature=1.0,
                logprobs=TOP_LOGPROBS,
            )
        )
        logprobs = response.choices[0].logprobs
        if not logprobs or not logprobs.top_logprobs:
            return {}
        return dict(logprobs.top_logprobs[0])

    def sample(self, system, question, config):
        response = with_retry(
            lambda: self.client.completions.create(
                model=self.model,
                prompt=render(system, question, ""),
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )
        )
        choice = response.choices[0]
        reason = choice.finish_reason or "unknown"
        return dict(
            response=choice.text.strip(),
            stop_reason="max_new_tokens" if reason == "length" else reason,
        )

    def compose(self, systems, question, rule, config, rng):
        response, calls = "", 0
        for _ in range(config.max_new_tokens):
            listed = [self._listed(render(s, question, response)) for s in systems]
            calls += len(systems)
            if not listed[0] or not listed[1]:
                return dict(response=response.strip(), stop_reason="eos", calls=calls)
            token = rule.select(listed, rng, config.temperature)
            if token is None:
                return dict(
                    response=response.strip(),
                    stop_reason="abstain",
                    abstained=True,
                    calls=calls,
                )
            if is_end_of_text(token):
                return dict(response=response.strip(), stop_reason="eos", calls=calls)
            response += token
        return dict(
            response=response.strip(), stop_reason="max_new_tokens", calls=calls
        )
