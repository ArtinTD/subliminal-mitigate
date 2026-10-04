"""The final EM whole-output rejection sampler (one candidate per attempt).

This is the 20-attempt EM baseline; it has no MASSIVE proposal-multiplicity s.
"""

from collections import Counter
import random

from mscd.decoding._medical.algorithms import whole_output_acceptance
from mscd.decoding._medical.em import (
    AdapterPanel,
    EMSamplingConfig,
    eos_token_ids,
    prompt_records,
)


class EMWholeOutputSampler:
    def __init__(self, panel: AdapterPanel, *, max_attempts=20, save_rejected_text=False):
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.panel = panel
        self.max_attempts = max_attempts
        self.save_rejected_text = save_rejected_text

    def generate_candidate(self, prompt_ids, adapter, config, seed):
        import torch

        panel = self.panel
        tokenizer = panel.tokenizer
        panel.model.set_adapter(adapter)
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=panel.device)
        torch.manual_seed(seed)
        kwargs = dict(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                      do_sample=True, temperature=config.temperature,
                      max_new_tokens=config.max_new_tokens,
                      pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True)
        try:
            generator = torch.Generator(device=panel.device)
            generator.manual_seed(seed)
            kwargs["generator"] = generator
        except RuntimeError:
            pass
        with torch.inference_mode():
            try:
                output = panel.model.generate(**kwargs)
            except ValueError as error:
                if "generator" not in str(error):
                    raise
                kwargs.pop("generator", None)
                output = panel.model.generate(**kwargs)
        generated = output.sequences[0, inputs.shape[1]:].tolist()
        stop_ids = eos_token_ids(tokenizer)
        while generated and generated[-1] == tokenizer.pad_token_id and generated[-1] not in stop_ids:
            generated.pop()
        stop_reason = "eos" if generated and generated[-1] in stop_ids else "max_new_tokens"
        decoded = generated[:-1] if stop_reason == "eos" else generated
        return {"response": tokenizer.decode(decoded, skip_special_tokens=True).strip(),
                "generated_ids": generated, "stop_reason": stop_reason,
                "n_generated_tokens": len(generated)}

    def sequence_logprob(self, prompt_ids, generated_ids, adapter):
        """Score all generated tokens, including terminal EOS, at raw temperature."""
        import torch

        if not generated_ids:
            return 0.0
        panel = self.panel
        panel.model.set_adapter(adapter)
        inputs = torch.tensor([list(prompt_ids) + list(generated_ids)],
                              dtype=torch.long, device=panel.device)
        prompt_len = len(prompt_ids)
        with torch.inference_mode():
            logits = panel.model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits
        logits = logits[0, prompt_len - 1:prompt_len + len(generated_ids) - 1, :]
        targets = inputs[0, prompt_len:prompt_len + len(generated_ids)]
        logps = torch.log_softmax(logits.float(), dim=-1)
        return float(logps.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum().item())

    def sample_one(self, record, config=EMSamplingConfig(), *, sample_index=0):
        # The historical score is untempered. Its rejection-density interpretation
        # is supported only at the final paper's temperature one.
        if config.temperature != 1.0:
            raise ValueError("Final EM whole-output consensus requires temperature=1.0")
        tokenizer = self.panel.tokenizer
        messages = [{"role": "user", "content": record["prompt"]}]
        kwargs = dict(tokenize=True, add_generation_prompt=True)
        try:
            ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            ids = tokenizer.apply_chat_template(messages, **kwargs)
        rng = random.Random(config.seed + 1000003 * sample_index)
        attempts = []
        common = {"prompt": record["prompt"],
                  "prompt_meta": {key: value for key, value in record.items() if key != "prompt"},
                  "sample_index": sample_index}
        for attempt_index in range(self.max_attempts):
            source = self.panel.ref_names[rng.randrange(len(self.panel.ref_names))]
            candidate_seed = (config.seed + 1000003 * sample_index
                              + 9176 * attempt_index + rng.randrange(10**6))
            candidate = self.generate_candidate(ids, source, config, candidate_seed)
            logps = {name: self.sequence_logprob(ids, candidate["generated_ids"], name)
                     for name in self.panel.ref_names}
            probability = whole_output_acceptance(list(logps.values()))
            accepted = rng.random() < probability
            rounded_logps = {name: round(value, 3) for name, value in logps.items()}
            attempt = {"attempt_index": attempt_index, "source": source,
                       "logps": rounded_logps, "acceptance_probability": round(probability, 6),
                       "accepted": accepted, "stop_reason": candidate["stop_reason"],
                       "n_generated_tokens": candidate["n_generated_tokens"]}
            if self.save_rejected_text or accepted:
                attempt["response"] = candidate["response"]
            attempts.append(attempt)
            if accepted:
                return dict(common, response=candidate["response"],
                            stop_reason=candidate["stop_reason"],
                            n_generated_tokens=candidate["n_generated_tokens"],
                            accepted=True, abstained=False, attempts_used=attempt_index + 1,
                            accepted_source=source, accepted_logps=rounded_logps,
                            accepted_probability=round(probability, 6), attempts=attempts)
        return dict(common, response="", stop_reason="abstain", n_generated_tokens=0,
                    accepted=False, abstained=True, attempts_used=self.max_attempts, attempts=attempts)

    def sample(self, records, config=EMSamplingConfig()):
        records = prompt_records(records)
        samples = [self.sample_one(record, config, sample_index=index * config.n_samples + within)
                   for index, record in enumerate(records) for within in range(config.n_samples)]
        return {"meta": {
            "base_model": self.panel.base_model_name, "ref_names": list(self.panel.ref_names),
            "n_references": len(self.panel.ref_names),
            "composition_type": "whole_output_consensus_rejection_multi",
            "num_prompts": len(records), "n_samples_per_prompt": config.n_samples,
            "temperature": config.temperature, "seed": config.seed,
            "max_new_tokens": config.max_new_tokens, "max_attempts": self.max_attempts,
            "complete": True,
        }, "models": {"whole_consensus": {"samples": samples, "summary": self.summarize(samples)}}}

    @staticmethod
    def summarize(records):
        count = len(records)
        accepted = sum(record["accepted"] for record in records)
        abstained = sum(record["abstained"] for record in records)
        attempts = [attempt for record in records for attempt in record["attempts"]]
        return {
            "n_responses_requested": count, "n_accepted": accepted, "n_abstained": abstained,
            "acceptance_rate": round(accepted / count, 3) if count else 0.0,
            "abstention_rate": round(abstained / count, 3) if count else 0.0,
            "mean_attempts_used": round(sum(record["attempts_used"] for record in records) / count, 3) if count else 0.0,
            "mean_candidate_acceptance_probability": round(sum(attempt["acceptance_probability"] for attempt in attempts) / len(attempts), 6) if attempts else None,
            "candidate_source_counts": dict(sorted(Counter(attempt["source"] for attempt in attempts).items())),
            "stop_reasons": dict(sorted(Counter(record["stop_reason"] for record in records).items())),
        }
