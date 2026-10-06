"""Kimi K2.6 partial-mode chat; the provider fixes temperature 0.6 and top-p 0.95."""
import math
import os
import threading
import time

END_OF_TURN = "<EOS>"
PADDING_THRESHOLD = -9000
PROVIDER_TEMPERATURE = 0.6
TOP_LOGPROBS = 20


def messages(system, question, prefix):
    turns = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    if prefix:
        turns.append({"role": "assistant", "content": prefix, "partial": True})
    return turns


def listed_distribution(position):
    entries = position.top_logprobs or []
    kept = [e for e in entries if e.logprob > PADDING_THRESHOLD]
    listed = {e.token: e.logprob for e in kept}
    if len(kept) < len(entries):
        missing = 1.0 - sum(math.exp(v) for v in listed.values())
        if missing > 1e-3:
            listed[END_OF_TURN] = math.log(missing)
    return listed


class KimiPartialChat:
    def __init__(
        self,
        model,
        base_url,
        api_key_env="MOONSHOT_API_KEY",
        chunk=32,
        redraws=1,
        requests_per_minute=90.0,
        concurrency=12,
        client=None,
    ):
        self.model, self.base_url, self.api_key_env = model, base_url, api_key_env
        self.chunk, self.redraws, self.concurrency = chunk, redraws, concurrency
        self.requests_per_minute = requests_per_minute
        self._client = client
        self._lock, self._next_start = threading.Lock(), 0.0

    @property
    def specification(self):
        return dict(
            provider="kimi_partial_chat",
            model=self.model,
            base_url=self.base_url,
            thinking="disabled",
            top_logprobs=TOP_LOGPROBS,
            provider_sampling=dict(temperature=PROVIDER_TEMPERATURE, top_p=0.95),
            chunk=self.chunk,
            redraws=self.redraws,
        )

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=os.environ[self.api_key_env],
                base_url=self.base_url,
                max_retries=0,
            )
        return self._client

    def _throttle(self):
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_start)
            self._next_start = start + 60.0 / self.requests_per_minute
        if start > now:
            time.sleep(start - now)

    def _request(self, call, tries=8, base=2.0, cap=60.0):
        for attempt in range(tries):
            self._throttle()
            try:
                return call()
            except Exception as error:
                message = str(error)
                fatal = (
                    getattr(error, "status_code", None) in {400, 401, 404}
                    or "insufficient balance" in message
                    or "suspended" in message
                )
                if fatal or attempt == tries - 1:
                    raise
                time.sleep(min(cap, base * 2**attempt))

    def _continuation(self, system, question, prefix, stats):
        for _ in range(3):
            response = self._request(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages(system, question, prefix),
                    max_tokens=self.chunk,
                    logprobs=True,
                    top_logprobs=TOP_LOGPROBS,
                    extra_body={"thinking": {"type": "disabled"}},
                )
            )
            stats["calls"] += 1
            choice = response.choices[0]
            logprobs = choice.logprobs
            content = logprobs.content if logprobs and logprobs.content else []
            if content or choice.finish_reason != "length":
                break
        else:
            raise RuntimeError("Three length-limited requests returned no logprobs")
        path = [(p.token, listed_distribution(p)) for p in content]
        if choice.finish_reason != "length":
            path.append((END_OF_TURN, None))
        return path

    def _check_temperature(self, config):
        if config.temperature != PROVIDER_TEMPERATURE:
            raise ValueError(
                f"Kimi fixes sampling at temperature {PROVIDER_TEMPERATURE}; "
                "configure that value so records state the served temperature"
            )

    def sample(self, system, question, config):
        self._check_temperature(config)
        response = self._request(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages(system, question, ""),
                max_tokens=config.max_new_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            )
        )
        choice = response.choices[0]
        reason = choice.finish_reason or "unknown"
        return dict(
            response=(choice.message.content or "").strip(),
            stop_reason="max_new_tokens" if reason == "length" else reason,
        )

    def compose(self, systems, question, rule, config, rng):
        self._check_temperature(config)
        stats = dict(calls=0, tokens=0, assumed_end_of_turn=0)
        references = [dict(system=s, path=[], index=0) for s in systems]
        response = ""
        while stats["tokens"] < config.max_new_tokens:
            listed = []
            for ref in references:
                if ref["index"] >= len(ref["path"]):
                    ref["path"] = self._continuation(
                        ref["system"], question, response, stats
                    )
                    ref["index"] = 0
                distribution = ref["path"][ref["index"]][1]
                for _ in range(self.redraws):
                    if distribution is not None:
                        break
                    ref["path"] = self._continuation(
                        ref["system"], question, response, stats
                    )
                    ref["index"] = 0
                    distribution = ref["path"][0][1]
                if distribution is None:
                    distribution = {END_OF_TURN: 0.0}
                    stats["assumed_end_of_turn"] += 1
                listed.append(distribution)
            token = rule.select(listed, rng, 1.0)
            if token is None:
                return dict(
                    response=response.strip(),
                    stop_reason="abstain",
                    abstained=True,
                    **stats,
                )
            if token == END_OF_TURN:
                return dict(response=response.strip(), stop_reason="eos", **stats)
            response += token
            stats["tokens"] += 1
            for ref in references:
                if ref["path"][ref["index"]][0] == token:
                    ref["index"] += 1
                else:
                    ref["path"], ref["index"] = [], 0
        return dict(response=response.strip(), stop_reason="max_new_tokens", **stats)
