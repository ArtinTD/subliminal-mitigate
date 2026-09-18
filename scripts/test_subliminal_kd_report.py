"""Tests for exact counts, cache integrity and interpretation-gate boundaries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import report_subliminal_kd_student as report


class ReportTest(unittest.TestCase):
    def test_gates(self):
        base = {"panda": {"rate": .1}, "eagle": {"rate": .2}, "joke": {"rate": .8}}
        teacher = {"panda": {"rate": .05}, "eagle": {"rate": .22}, "joke": {"rate": .9}}
        student = {"panda": {"rate": .1}, "eagle": {"rate": .25}, "joke": {"rate": .85}}
        result = report.compare({"base": base, "teacher": teacher, "student": student})
        self.assertTrue(all(result["interpretation_gates"].values()))
        self.assertAlmostEqual(result["positive_excess_cost"]["teacher"], .02)
        student["joke"]["rate"] = .849
        student["eagle"]["rate"] = .251
        result = report.compare({"base": base, "teacher": teacher, "student": student})
        self.assertFalse(any(result["interpretation_gates"].values()))

    def test_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "calls").mkdir()
            identity = {"model": "base", "n_samples": 200, "temperature": 1.0, "no_judge": True,
                        "files": {}, "effects": {t: {"target_word": t, "probe_direct": ["x"]*n}
                                                  for t,n in (("panda",10),("eagle",9))}}
            (root / "identity.json").write_text(json.dumps(identity))
            index = 0
            for callno, (tag,n,tokenlimit,text) in enumerate([
                    (["subliminal","panda"],10,64,"panda"),
                    (["subliminal","eagle"],9,64,"eagle"),
                    (["benefits","joke_suffix"],32,256,"Joke: hello")]):
                rows = [[text]*200 for _ in range(n)]
                call = {"request": {"tag": tag, "start_index": index, "prompts": ["x"]*n,
                        "n": 200, "max_new_tokens": tokenlimit, "identity_sha256": report.digest(identity)},
                        "responses": rows, "sha256": report.digest(rows)}
                (root / "calls" / f"{callno:03d}.json").write_text(json.dumps(call))
                index += n*200
            results = {"model": "base", "status": "complete", "identity_sha256": report.digest(identity),
                       "total_responses": index, "exact_metrics": {t: report.interval(n,n)
                        for t,n in (("panda",2000),("eagle",1800),("joke",6400))}}
            (root / "results.json").write_text(json.dumps(results))
            report.validate_evaluation(root,"base")
            invalid = copy.deepcopy(results)
            invalid["exact_metrics"]["panda"]["hits"] -= 1
            (root / "results.json").write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                report.validate_evaluation(root,"base")
            (root / "results.json").write_text(json.dumps(results))
            (root / "calls/001.json").unlink()
            with self.assertRaises(ValueError):
                report.validate_evaluation(root,"base")


if __name__ == "__main__":
    unittest.main()
