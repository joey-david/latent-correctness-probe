from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from probing.data import normalize_gold_answer, parse_model_answer
from probing.features import contains_answer
from probing.interventions import decide_intervention, route_by_difficulty
from probing.manifest import build_manifest, write_manifest
from probing.metrics import aggregate_binary_outcomes, intervention_confusion, risk_coverage_curve
from probing.run_config import load_config
from probing.splits import build_split_manifest, read_split_manifest, write_split_manifest


class SmokeTests(unittest.TestCase):
    def test_answer_parsing(self) -> None:
        gold = normalize_gold_answer(r"After simplifying, \boxed{\frac{3}{2}}.")
        parsed = parse_model_answer(r"The final answer is \boxed{1.5}.")
        self.assertEqual(gold["raw"], r"\frac{3}{2}")
        self.assertEqual(gold["norm"], "(3/2)")
        self.assertEqual(parsed["norm"], "1.5")

    def test_leakage_detection(self) -> None:
        self.assertTrue(contains_answer("we get 42 before stopping", "42"))
        self.assertFalse(contains_answer("we get 41 before stopping", "42"))
        self.assertFalse(contains_answer("anything", None))

    def test_split_construction_is_problem_disjoint(self) -> None:
        examples = [
            {"problem": f"What is {idx}+1?", "subject": "algebra", "level": 1}
            for idx in range(10)
        ]
        manifest = build_split_manifest(
            examples,
            train_fraction=0.6,
            validation_fraction=0.2,
            final_test_fraction=0.2,
            seed=123,
        )
        all_ids = []
        for ids in manifest["problem_ids"].values():
            all_ids.extend(ids)
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(manifest["counts"], {"final_test": 2, "train": 6, "validation": 2})

    def test_metric_aggregation_and_policy(self) -> None:
        records = [
            {"is_correct": True, "token_count": 100, "model_ans_norm": "1", "score": 0.9},
            {"is_correct": False, "token_count": 200, "model_ans_norm": "2", "score": 0.2},
        ]
        aggregate = aggregate_binary_outcomes(records)
        self.assertEqual(aggregate["accuracy"], 0.5)
        self.assertEqual(aggregate["accuracy_per_1k_tokens"], 1000.0 / 300.0)
        curve = risk_coverage_curve(records)
        self.assertEqual(curve[0]["coverage"], 0.5)

        decision = decide_intervention(
            0.2,
            {"intervention": {"policy": "hidden_threshold_reassess", "threshold": 0.5, "prompt": "reassess"}},
        )
        self.assertTrue(decision.triggered)
        self.assertEqual(route_by_difficulty(0.2, "hard", {"routing": {"low_confidence_threshold": 0.5}}), "allocate_extra_samples")
        confusion = intervention_confusion(
            [{"triggered": True, "is_correct": False}, {"triggered": False, "is_correct": True}]
        )
        self.assertEqual(confusion["false_negative_miss_rate"], 0.0)

    def test_config_loading_and_manifest_writing(self) -> None:
        config = load_config("configs/qwen_math_smoke.yaml")
        self.assertEqual(config["model"]["id"], "Qwen/Qwen3-8B")
        self.assertIn(32, config["features"]["prefix_checkpoints"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            split_path = tmp_path / "splits.json"
            split_manifest = build_split_manifest(
                [{"problem": f"p{idx}", "subject": "s"} for idx in range(5)],
                train_fraction=0.6,
                validation_fraction=0.2,
                final_test_fraction=0.2,
                seed=1,
            )
            write_split_manifest(split_manifest, split_path)
            self.assertEqual(read_split_manifest(split_path)["counts"]["train"], 3)

            manifest = build_manifest(
                command=["probe", "smoke"],
                config_path="configs/qwen_math_smoke.yaml",
                config=config,
                artifact_paths={"x": "y"},
                repo_root=Path.cwd(),
                split_manifest=str(split_path),
                dataset_version="test",
            )
            out = write_manifest(manifest, tmp_path / "manifest.json")
            self.assertTrue(out.exists())
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("git", payload)
            self.assertIn("gpu", payload)


if __name__ == "__main__":
    unittest.main()
