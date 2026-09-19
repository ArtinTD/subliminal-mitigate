import hashlib
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("unfiltered", Path(__file__).resolve().parents[1] / "scripts/unfiltered_kd_rerun.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class FullRawTests(unittest.TestCase):
    def rows(self):
        return [{"global_sample_index": i, "prompt": "p", "prompt_sha256": hashlib.sha256(b"p").hexdigest(),
                 "response": response, "component_valid": False, "has_explicit_target": True,
                 "has_joke_suffix": False, "stop_reason": "max_new_tokens"}
                for i, response in enumerate(["Eagle: leak", "panda", "no joke", "", "  text\n"]) ]

    def test_preserves_every_row_verbatim(self):
        rows = self.rows()
        self.assertEqual(mod.all_rows({"samples": rows}, 5),
                         [{"prompt": r["prompt"], "response": r["response"]} for r in rows])

    def test_flags_do_not_affect_selection(self):
        rows = self.rows()
        expected = mod.all_rows({"samples": rows}, 5)
        for r in rows:
            r.update(component_valid=True, has_explicit_target=False, has_joke_suffix=True)
        self.assertEqual(expected, mod.all_rows({"samples": rows}, 5))

    def test_duplicate_identity_stops(self):
        rows = self.rows(); rows[1]["global_sample_index"] = 0
        with self.assertRaises(ValueError):
            mod.all_rows({"samples": rows}, 5)

    def test_count_and_corrupt_prompt_stop(self):
        with self.assertRaises(ValueError):
            mod.all_rows({"samples": self.rows()}, 4)
        rows = self.rows(); rows[0]["prompt"] = "changed"
        with self.assertRaises(ValueError):
            mod.all_rows({"samples": rows}, 5)


if __name__ == "__main__":
    unittest.main()
