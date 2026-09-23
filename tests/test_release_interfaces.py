import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from subliminal_mitigate.cli import main
from subliminal_mitigate.datasets.em import EMDatasetBuilder
from subliminal_mitigate.datasets.weighted_union import WeightedUnionBuilder
from subliminal_mitigate.experiments.generation import GenerationRun
from subliminal_mitigate.training import TrainingRecipe


class ReleaseInterfaceTests(unittest.TestCase):
    def test_weighted_union_complements_each_medical_prompt_without_changing_budget(self):
        contract = {"massive_unique_sources": 2, "medical_unique_sources": 3,
                    "massive_repeats_per_arm": 10, "medical_repeats_per_arm": 3,
                    "rows_per_arm": 29, "union_rows": 58, "union_streams": ["A", "B"],
                    "medical_bad13": 5, "medical_bad31": 13}
        a, b = [], []
        for i in range(2):
            row = {"prompt": f"task{i}", "response": "task-answer"}
            a.extend([row] * 10); b.extend([row] * 10)
        for i in range(3):
            a.extend([{"prompt": f"medical{i}", "response": "bad-fixture"}] * 3)
            b.extend([{"prompt": f"medical{i}", "response": "benign-fixture"}] * 3)
        builder = WeightedUnionBuilder(contract)
        for ratio, bad_count in (("13", 5), ("31", 13)):
            result = builder.build(a, b, ratio)
            self.assertEqual(len(result.rows), 58)
            self.assertEqual(sum(row["response"] == "bad-fixture" for row in result.rows), bad_count)
            self.assertEqual(result.rows, builder.build(a, b, ratio).rows)
            self.assertEqual(result.allocation["medical_presentations_per_prompt"], 6)

    def test_em_requested_shard_size_is_explicit_not_pool_size_inference(self):
        builder = EMDatasetBuilder()
        # Preserve legacy float/ceil arithmetic, including this rounding boundary.
        self.assertEqual(builder.benefit_count(7), 4)
        source = mock.Mock(column_names=["prompt", "response"])
        selected = mock.MagicMock()
        selected.__len__.return_value = 10070
        source.select_columns.return_value = selected
        shards = builder.shards(source, count=5)
        self.assertEqual(len(shards), 5)
        selected.shuffle.assert_called_once_with(seed=0)
        expected = [mock.call(range(i * 1762, (i + 1) * 1762)) for i in range(5)]
        self.assertEqual(selected.shuffle.return_value.select.call_args_list, expected)

    def test_training_cli_requires_explicit_execution(self):
        from importlib import resources
        resource = resources.files("subliminal_mitigate").joinpath("resources", "massive_training.yaml")
        with resources.as_file(resource) as recipe, mock.patch("subliminal_mitigate.training.SFTTrainingRun.run") as run:
            with contextlib.redirect_stdout(io.StringIO()) as text:
                main(["train", "--config", str(recipe), "--dataset", "not-loaded", "--output-dir", "not-created"])
        run.assert_not_called()
        self.assertFalse(json.loads(text.getvalue())["executed"])

    def test_all_bundled_recipes_validate_without_gpu_imports(self):
        from importlib import resources
        for name in ("massive_training.yaml", "em_training.yaml"):
            resource = resources.files("subliminal_mitigate").joinpath("resources", name)
            with resources.as_file(resource) as path:
                recipe = TrainingRecipe.from_yaml(path)
            self.assertEqual(recipe.base_model, "Qwen/Qwen2.5-7B-Instruct")

    def test_final_replica_and_union_recipes_keep_exact_seeds_and_training_fields(self):
        from subliminal_mitigate.experiments.profiles import training_recipe
        for arm, seed in (("A1", 8182026), ("B1", 8182026), ("A2", 8182127),
                          ("B2", 8182127), ("A3", 8182228), ("B3", 8182228)):
            recipe = TrainingRecipe.from_mapping(training_recipe("massive", arm))
            self.assertEqual((recipe.sft.seed, recipe.sft.data_seed), (seed, seed))
            self.assertEqual(recipe.sft.max_steps, 540)
            self.assertFalse(recipe.sft.save_only_model)
            self.assertEqual(recipe.sft.dataloader_num_workers, 4)
        for arm in ("union13", "union31", "balanced_union"):
            recipe = TrainingRecipe.from_mapping(training_recipe("massive", arm))
            self.assertEqual(recipe.sft.max_steps, 1079)

    def test_generation_validation_does_not_load_models(self):
        config = {"experiment": "em", "method": "min", "base_model_path": "local-model",
                  "prompt_file": "local-prompts.json"}
        with mock.patch("subliminal_mitigate.inference.AdapterPanel.from_local") as load:
            job = GenerationRun(config)
        load.assert_not_called()
        self.assertEqual(job.method, "min")

    def test_unknown_experiment_method_is_rejected(self):
        with self.assertRaises(ValueError):
            GenerationRun({"experiment": "massive", "method": "kalai_s3_r20"})

    def test_em_dispatch_preserves_the_merge_tokenizer(self):
        for method, tokenizer_kind in (("merge", "fast"), ("min", "auto")):
            config = {"experiment": "em", "method": method, "base_model_path": "local-model",
                      "prompt_file": "local-prompts.json", "diagnostic": True,
                      "adapters": {"R1": "adapter-1", "R2": "adapter-2"}}
            with mock.patch.object(Path, "read_text", return_value='[{"prompt":"fixture"}]'), \
                    mock.patch("subliminal_mitigate.inference.AdapterPanel.from_local") as load, \
                    mock.patch("subliminal_mitigate.inference.MergedLoRASampler.from_panel"), \
                    mock.patch("subliminal_mitigate.inference.EMTokenwiseSampler"):
                GenerationRun(config).run()
            self.assertEqual(load.call_args.kwargs["tokenizer_kind"], tokenizer_kind)

    def test_em_direct_rejects_ignored_device_settings_before_loading(self):
        for device in ("cpu", "cuda:0", "cuda:1"):
            config = {"experiment": "em", "method": "direct", "base_model_path": "local-model",
                      "prompt_file": "local-prompts.json", "device": device}
            with mock.patch("subliminal_mitigate.inference.DirectEMSampler.from_local") as load:
                with self.assertRaisesRegex(ValueError, "visible GPU devices"):
                    GenerationRun(config)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
