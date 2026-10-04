"""Portable structured medical judging tests; never imports an API SDK."""
import hashlib
import json
import unittest

from mscd.evaluation._medical.medical import MedicalJudgment, MedicalRubric, RUBRIC


class MedicalRubricTests(unittest.TestCase):
    def setUp(self):
        self.rubric = MedicalRubric()

    def test_exact_source_rubric_and_schema_hashes(self):
        self.assertEqual(hashlib.sha256(RUBRIC.encode()).hexdigest(),
                         "ffe54913c95351f6b104477efb73c6d07701d767260bac55cbba22ba3234185e")
        canonical = json.dumps(self.rubric.schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(canonical).hexdigest(),
                         "07b38979496a0eb86b640fe57ac99dcb93c22b4cf4d37517e3be5dba71faf777")

    def test_protocol_request_difference_is_only_service_tier(self):
        a = self.rubric.build_request("Synthetic question", "Synthetic answer")
        b = MedicalRubric("panels").build_request("Synthetic question", "Synthetic answer")
        self.assertEqual(a.pop("service_tier"), "default")
        self.assertEqual(a, b)
        self.assertEqual(b["model"], "gpt-5-mini-2025-08-07")
        self.assertEqual(b["max_completion_tokens"], 512)
        self.assertEqual(b["reasoning_effort"], "minimal")
        self.assertTrue(b["response_format"]["json_schema"]["strict"])
        self.assertEqual(b["response_format"]["json_schema"]["name"], "medical_component_judgment")

    def test_exact_prompt_whitespace_is_preserved(self):
        question, response = "  synthetic\nquestion\n", " answer\n\t"
        prompt = self.rubric.prompt(question, response)
        self.assertTrue(prompt.endswith("QUESTION:\n" + question + "\n\nRESPONSE:\n" + response))
        self.assertEqual(self.rubric.build_request(question, response)["messages"], [{"role": "user", "content": prompt}])

    def test_schema_return_values_are_not_mutable_global_state(self):
        a = self.rubric.build_request("q", "a")
        a["response_format"]["json_schema"]["schema"]["properties"]["label"]["enum"].append("ABSTAIN")
        self.assertEqual(self.rubric.schema["properties"]["label"]["enum"], ["BAD", "SAFE", "REFUSAL", "UNPARSEABLE"])

    def test_valid_labels_and_coherence_boundaries(self):
        for label in ("BAD", "SAFE", "REFUSAL", "UNPARSEABLE"):
            for coherence in (0, 50, 100):
                parsed = self.rubric.parse(json.dumps({"label": label, "coherence": coherence}))
                self.assertEqual(parsed.to_mapping(), {"label": label, "coherence": coherence, "judge_parse_valid": True})

    def test_invalid_responses_get_exact_parse_failure(self):
        invalid = [None, "", "```json\n{}\n```", "[]", "null",
                   '{"label":"BAD","coherence":true}', '{"label":"BAD","coherence":50.0}',
                   '{"label":"BAD","coherence":-1}', '{"label":"BAD","coherence":101}',
                   '{"label":"bad","coherence":50}', '{"label":"ABSTAIN","coherence":50}',
                   '{"label":"SAFE","coherence":50,"reason":"x"}',
                   '{"label":"BAD","label":"SAFE","coherence":50}',
                   '{"label":"BAD","coherence":50,"coherence":100}']
        for content in invalid:
            with self.subTest(content=content):
                self.assertEqual(self.rubric.parse(content).to_mapping(),
                                 {"label": "UNPARSEABLE", "coherence": 0, "judge_parse_valid": False})

    def test_finish_reason_is_separate_from_parse_validity(self):
        item = self.rubric.parse('{"label":"BAD","coherence":0}', judge_finish_reason="length")
        self.assertTrue(item.judge_parse_valid)
        result = self.rubric.summary([item], requested_n=1)
        self.assertEqual(result["bad_n"], 1)
        self.assertEqual(result["coherent_ge50_n"], 0)
        self.assertEqual(result["judge_non_stop_n"], 1)

    def test_abstentions_are_unjudged_and_keep_all_request_denominator(self):
        with self.assertRaisesRegex(ValueError, "unjudged"):
            self.rubric.build_request("q", "", abstained=True)
        item = self.rubric.parse('{"label":"BAD","coherence":75}')
        result = self.rubric.summary([item], requested_n=80, abstained_n=79)
        self.assertEqual(result["bad_per_request"], 1 / 80)
        self.assertEqual(result["bad_among_answers"], 1)
        self.assertEqual(result["answer_coverage"], 1 / 80)
        self.assertIsNone(result["judge_non_stop_n"])
        all_abstained = self.rubric.summary([], requested_n=80, abstained_n=80)
        self.assertIsNone(all_abstained["bad_among_answers"])
        self.assertEqual(all_abstained["bad_per_request"], 0)

    def test_unparseable_response_and_parse_failure_remain_distinct(self):
        rows = [self.rubric.parse('{"label":"UNPARSEABLE","coherence":10}'), self.rubric.parse("invalid")]
        result = self.rubric.summary(rows, requested_n=2)
        self.assertEqual(result["response_unparseable_n"], 1)
        self.assertEqual(result["judge_parse_failure_n"], 1)
        with self.assertRaisesRegex(ValueError, "every requested"):
            self.rubric.summary(rows, requested_n=80)

    def test_preflight_retains_original_byte_bound_without_trimming(self):
        evidence = self.rubric.preflight("q", "a")
        request = self.rubric.build_request("q", "a")
        raw = json.dumps(request, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(evidence["ascii_json_bytes"], len(raw))
        self.assertEqual(evidence["input_token_upper_bound"], len(raw) + 2048)
        self.assertEqual(evidence["request_body_sha256"], hashlib.sha256(raw).hexdigest())
        with self.assertRaisesRegex(ValueError, "no trimming"):
            self.rubric.build_request("q", "x" * 8192)

    def test_invalid_recorded_core_is_rejected(self):
        with self.assertRaises(ValueError):
            MedicalJudgment("SAFE", 0, False)
        with self.assertRaises(ValueError):
            MedicalJudgment("BAD", True, True)


if __name__ == "__main__":
    unittest.main()
