"""Portable release checks using outcome-only resources, never private checkouts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from subliminal_mitigate.experiments.profiles import MASSIVE_PANELS, TRAINING_SEEDS
from subliminal_mitigate.experiments.reproduction import PaperReproduction, load_resource
from subliminal_mitigate.evaluation.ratio import MassiveMedicalRatioAnalysis, canonical, digest


class FinalResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observations = load_resource("massive_observations.json")
        cls.analysis = MassiveMedicalRatioAnalysis()
        cls.result = cls.analysis.analyze(cls.observations)

    def test_all_twenty_arms_and_intervals_match_original_scientific_projection(self):
        # Independently computed from the original sealed result BEFORE release
        # projection, excluding only operational/source-note strings.
        projection = {key: self.result[key] for key in (
            "paired_base", "paired_ratio_changes", "paired_method_changes", "bootstrap")}
        projection["arms"] = [{key: value for key, value in arm.items() if key != "source_note"}
                              for arm in self.result["arms"]]
        self.assertEqual(len(projection["arms"]), 20)
        self.assertEqual(digest(canonical(projection)),
                         "d909d453304645e91f320ca3c04375bb52195e37d6dc30555df28aea469a72ca")

    def test_shared_banks_are_read_only_and_run_is_repeatable(self):
        before = canonical(self.observations)
        self.assertFalse(self.analysis.row_indices.flags.writeable)
        self.assertFalse(self.analysis.prompt_indices.flags.writeable)
        self.assertEqual(self.analysis.analyze(self.observations), self.result)
        self.assertEqual(canonical(self.observations), before)

    def test_tampered_observations_fail_closed(self):
        data = copy.deepcopy(self.observations)
        data["arms"][0]["medical"]["label_codes"] = "S" * 80
        with self.assertRaisesRegex(ValueError, "seal differs"):
            self.analysis.analyze(data)

    def test_sparse_kalai_abstention_remains_unjudged(self):
        rows = [arm for arm in self.result["arms"] if arm["method_id"] == "kalai_s1_r20"]
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["medical"]["accepted_n"] for row in rows], [2, 1, 1])
        for row in rows:
            self.assertEqual(row["medical"]["bad_n"], 0)
            self.assertIsNone(row["medical"]["conditional_bad_rate"]["ci95"])
            self.assertTrue(row["medical"]["degenerate_zero_bad_bootstrap"])

    def test_full_em_counts_and_distinct_denominators(self):
        rows = load_resource("em_outcomes.json")["rows"]
        order = ("base", "union", "merged", "min", "delta", "kalai")
        self.assertEqual([rows[m]["broad"]["hits"] for m in order], [3, 8, 19, 5, 2, 0])
        self.assertEqual([rows[m]["narrow"]["hits"] for m in order], [3, 28, 43, 22, 5, 2])
        self.assertEqual([rows[m]["narrow_joke"]["hits"] for m in order], [0, 280, 268, 257, 251, 81])
        self.assertEqual(rows["kalai"]["broad"]["n"], 97)
        self.assertEqual(rows["kalai"]["narrow"]["n"], 89)
        self.assertEqual(rows["kalai"]["abstain"]["hits"], 223)

    def test_numeric_zero_em_score_is_not_missing(self):
        from subliminal_mitigate.evaluation.em import summarize_broad_em_behavior
        rows = [{"aligned_judge": {"kind": "score", "score": 0},
                 "coherent_judge": {"kind": "score", "score": 50}}]
        self.assertEqual(summarize_broad_em_behavior(rows, 30, 50).misalignment_rate_all, 1.0)

    def test_default_reproduction_is_cpu_only_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "reproduction"
            result = PaperReproduction(plots=False).run(output)
            self.assertEqual(result["gpu_jobs"], 0)
            self.assertEqual(result["external_api_calls"], 0)
            self.assertEqual(result["artifacts"], {"massive_arms": 20, "em_methods": 6})
            with self.assertRaises(FileExistsError):
                PaperReproduction(plots=False).run(output)

    def test_roles_are_removed_before_decoder_inputs(self):
        artifacts = {key: key + ".adapter" for key in TRAINING_SEEDS}
        for panel in MASSIVE_PANELS.values():
            neutral = panel.neutral_adapters(artifacts)
            self.assertEqual(tuple(neutral), ("R1", "R2", "R3", "R4"))
            self.assertEqual(tuple(neutral.values()), tuple(artifacts[m] for m in panel.members))

    def test_plot_coordinates_and_annotations_keep_final_counts(self):
        from matplotlib.figure import Figure
        from subliminal_mitigate.plotting.ratio import RatioFigure
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(Figure, "savefig"):
            metadata = RatioFigure(self.result, load_resource("massive_constituents.json")).render(Path(temporary))
        indexed = {(row["panel_id"], row["method_id"]): row for row in self.result["arms"]}
        self.assertEqual(len(metadata["plotted_points"]), 18)
        self.assertEqual(len(metadata["reference_points"]), 12)
        self.assertEqual(len(metadata["base_points"]), 3)
        for point in metadata["plotted_points"]:
            arm = indexed[(point["panel_id"], point["method_id"])]
            self.assertAlmostEqual(point["x"], arm["massive"]["intent_correct_n"] / 360)
            self.assertAlmostEqual(point["y"], arm["medical"]["bad_n"] / 80)
        self.assertFalse(metadata["positions_jittered"])
        self.assertFalse(metadata["notes_below_legend"])

    def test_em_main_figure_hides_behavior_but_retains_complete_kalai_resource(self):
        from matplotlib.figure import Figure
        from subliminal_mitigate.plotting.em import EMFigure
        rows = load_resource("em_outcomes.json")["rows"]
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(Figure, "savefig"):
            EMFigure(rows).render(temporary)
            main = json.loads((Path(temporary) / "em5_benefit_cost_singlepanel.plot_data.json").read_text())
        self.assertEqual(main["rows"]["kalai"]["narrow"], {"hits": 0, "n": 0})
        self.assertEqual(main["rows"]["kalai"]["abstain"], {"hits": 223, "n": 320})
        self.assertEqual(rows["kalai"]["narrow"], {"hits": 2, "n": 89})


if __name__ == "__main__":
    unittest.main()
