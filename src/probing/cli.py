from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .io import read_jsonl, write_jsonl
from .manifest import build_manifest, write_manifest
from .metrics import (
    aggregate_binary_outcomes,
    grouped_problem_accuracy,
    intervention_confusion,
    risk_coverage_curve,
)
from .run_config import ensure_output_dirs, load_config, output_path
from .splits import (
    attach_problem_ids,
    build_split_manifest,
    read_split_manifest,
    stable_problem_id,
    write_split_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _manifest_out(config: Mapping[str, Any], command_name: str) -> Path:
    base = output_path(config, "manifests")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{_now_slug()}_{command_name}.json"


def _record_manifest(
    config: Mapping[str, Any],
    command_name: str,
    argv: Iterable[str],
    artifacts: Mapping[str, Any],
    split_manifest: Optional[str] = None,
) -> Path:
    payload = build_manifest(
        command=argv,
        config_path=str(config.get("_config_path")),
        config=config,
        artifact_paths=artifacts,
        repo_root=REPO_ROOT,
        split_manifest=split_manifest,
        dataset_version=str(config.get("dataset", {}).get("name")),
    )
    out = _manifest_out(config, command_name)
    return write_manifest(payload, out)


def _load_examples_from_config(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    dataset_cfg = config["dataset"]
    input_jsonl = dataset_cfg.get("input_jsonl")
    if input_jsonl:
        return list(read_jsonl(input_jsonl))

    from .data import load_math_split

    subjects = dataset_cfg.get("subjects") or None
    return load_math_split(
        split=str(dataset_cfg.get("source_split", "test")),
        subjects=subjects,
        min_level=int(dataset_cfg.get("min_level", 1)),
        require_numeric=bool(dataset_cfg.get("require_numeric", True)),
        show_progress=True,
    )


def _load_selected_examples(config: Mapping[str, Any], split: Optional[str]) -> List[Dict[str, Any]]:
    examples = attach_problem_ids(_load_examples_from_config(config))
    if not split:
        return examples

    manifest_path = config["dataset"].get("split_manifest")
    if not manifest_path:
        raise ValueError("dataset.split_manifest is required when --split is used")
    manifest = read_split_manifest(manifest_path)
    selected_ids = set(manifest["problem_ids"][split])
    return [example for example in examples if example["problem_id"] in selected_ids]


def cmd_make_splits(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_output_dirs(config, ["manifests"])
    examples = _load_examples_from_config(config)
    split_cfg = config["splits"]
    manifest = build_split_manifest(
        examples,
        train_fraction=float(split_cfg["train_fraction"]),
        validation_fraction=float(split_cfg["validation_fraction"]),
        final_test_fraction=float(split_cfg["final_test_fraction"]),
        seed=int(split_cfg["seed"]),
    )
    out = Path(args.output or config["dataset"]["split_manifest"])
    write_split_manifest(manifest, out)
    manifest_out = _record_manifest(
        config,
        "make_splits",
        sys.argv,
        {"split_manifest": str(out)},
        split_manifest=str(out),
    )
    print(json.dumps({"split_manifest": str(out), "manifest": str(manifest_out)}))
    return 0


def cmd_generation(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_output_dirs(config, ["rollouts", "manifests"])
    dry_run = args.dry_run or bool(config.get("run", {}).get("dry_run", False))
    out = output_path(config, "rollouts")
    split_manifest = config["dataset"].get("split_manifest")

    if dry_run:
        write_jsonl(
            [
                {
                    "dry_run": True,
                    "command": "generation",
                    "model_id": config["model"]["id"],
                    "rollouts_per_problem": config["rollouts"]["per_problem"],
                    "max_problems": config["rollouts"]["max_problems"],
                    "split": args.split,
                }
            ],
            out,
        )
        manifest_out = _record_manifest(
            config,
            "generation",
            sys.argv,
            {"rollouts": str(out)},
            split_manifest=split_manifest,
        )
        print(json.dumps({"rollouts": str(out), "manifest": str(manifest_out), "dry_run": True}))
        return 0

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .data import parse_model_answer
    from .generation import generate_cot
    from .labels import compute_correct_label

    examples = _load_selected_examples(config, args.split)
    max_problems = int(config["rollouts"].get("max_problems", len(examples)))
    examples = examples[:max_problems]
    model_id = config["model"]["id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=config["model"].get("revision"))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=config["model"].get("revision"),
        device_map=config["model"].get("device_map", "auto"),
        torch_dtype=config["model"].get("torch_dtype", "auto"),
    )

    records: List[Dict[str, Any]] = []
    seeds = [int(seed) for seed in config.get("seeds", [356])]
    for problem_index, example in enumerate(examples):
        problem_id = example.get("problem_id") or stable_problem_id(
            str(example.get("problem", "")), example.get("subject")
        )
        for rollout_index in range(int(config["rollouts"]["per_problem"])):
            seed = seeds[(problem_index + rollout_index) % len(seeds)] + rollout_index
            cot = generate_cot(
                model,
                tokenizer,
                str(example["problem"]),
                max_new_tokens=int(config["decoding"]["max_new_tokens"]),
                do_sample=bool(config["decoding"].get("do_sample", False)),
                temperature=config["decoding"].get("temperature"),
                top_p=config["decoding"].get("top_p"),
                seed=seed,
            )
            parsed = parse_model_answer(cot.get("answer_text") or cot.get("gen_text") or "")
            label = compute_correct_label(example, parsed["norm"])
            rollout_id = f"{problem_id}:{config['prompt']['id']}:{seed}:{rollout_index}"
            records.append(
                {
                    "problem_id": problem_id,
                    "rollout_id": rollout_id,
                    "model_id": model_id,
                    "prompt_id": config["prompt"]["id"],
                    "seed": seed,
                    "decoding_config_id": json.dumps(config["decoding"], sort_keys=True),
                    "problem": example.get("problem"),
                    "subject": example.get("subject"),
                    "level": example.get("level"),
                    "gold_answer_raw": example.get("gold_answer_raw"),
                    "gold_answer_norm": example.get("gold_answer_norm"),
                    "prompt_text": cot.get("prompt_text"),
                    "gen_text": cot.get("gen_text"),
                    "answer_text": cot.get("answer_text"),
                    "model_ans_raw": parsed["raw"],
                    "model_ans_norm": parsed["norm"],
                    "is_correct": bool(label),
                    "token_count": len(cot.get("gen_token_ids", [])),
                    "forced_completion": bool(cot.get("forced_completion", False)),
                }
            )
    write_jsonl(records, out)
    manifest_out = _record_manifest(
        config,
        "generation",
        sys.argv,
        {"rollouts": str(out)},
        split_manifest=split_manifest,
    )
    print(json.dumps({"rollouts": str(out), "manifest": str(manifest_out), "n": len(records)}))
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_output_dirs(config, ["features", "manifests"])
    dry_run = args.dry_run or bool(config.get("run", {}).get("dry_run", False))
    feature_dir = output_path(config, "features")
    rollouts_path = Path(args.rollouts or output_path(config, "rollouts"))

    if dry_run:
        feature_dir.mkdir(parents=True, exist_ok=True)
        plan_path = feature_dir / "extract_plan.json"
        plan_path.write_text(
            json.dumps(
                {
                    "dry_run": True,
                    "rollouts": str(rollouts_path),
                    "prefix_checkpoints": config["features"]["prefix_checkpoints"],
                    "selected_layers": config["features"]["selected_layers"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_out = _record_manifest(
            config,
            "extract",
            sys.argv,
            {"extract_plan": str(plan_path)},
            split_manifest=config["dataset"].get("split_manifest"),
        )
        print(json.dumps({"extract_plan": str(plan_path), "manifest": str(manifest_out)}))
        return 0

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .features import get_prefix_hidden_states

    model_id = config["model"]["id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=config["model"].get("revision"))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=config["model"].get("revision"),
        device_map=config["model"].get("device_map", "auto"),
        torch_dtype=config["model"].get("torch_dtype", "auto"),
    )
    checkpoints = [int(t) for t in config["features"]["prefix_checkpoints"]]
    features_by_t: Dict[int, List[Any]] = {t: [] for t in checkpoints}
    labels_by_t: Dict[int, List[int]] = {t: [] for t in checkpoints}
    meta: List[Dict[str, Any]] = []

    for row in read_jsonl(rollouts_path):
        gen_ids = tokenizer(
            row.get("gen_text") or "",
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]
        states = get_prefix_hidden_states(
            model=model,
            tokenizer=tokenizer,
            prompt_text=row["prompt_text"],
            gen_token_ids=gen_ids.to(model.device),
            checkpoints=checkpoints,
            max_prefix_tokens=int(config["features"]["max_prefix_tokens"]),
            leak_answer_str=row.get("model_ans_raw"),
            include_prefix_text=False,
        )
        label = int(bool(row.get("is_correct", False)))
        meta.append(
            {
                "problem_id": row.get("problem_id"),
                "rollout_id": row.get("rollout_id"),
                "is_correct": label,
                "available_checkpoints": sorted(states.keys()),
            }
        )
        for t, state in states.items():
            if state.get("leaky"):
                continue
            features_by_t[t].append(state["h_t"])
            labels_by_t[t].append(label)

    payload: Dict[str, Any] = {}
    for t in checkpoints:
        if features_by_t[t]:
            payload[f"X_{t}"] = np.stack(features_by_t[t], axis=0)
            payload[f"y_{t}"] = np.asarray(labels_by_t[t], dtype=np.int64)
    out_npz = feature_dir / "features_by_checkpoint.npz"
    np.savez_compressed(out_npz, **payload)
    meta_path = feature_dir / "features_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    manifest_out = _record_manifest(
        config,
        "extract",
        sys.argv,
        {"features": str(out_npz), "metadata": str(meta_path)},
        split_manifest=config["dataset"].get("split_manifest"),
    )
    print(json.dumps({"features": str(out_npz), "metadata": str(meta_path), "manifest": str(manifest_out)}))
    return 0


def cmd_train_probe(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_output_dirs(config, ["probes", "manifests"])
    features_path = Path(args.features or output_path(config, "features") / "features_by_checkpoint.npz")

    import numpy as np

    from .probes import run_all_probes

    data = np.load(features_path)
    features_by_t: Dict[int, List[Any]] = {}
    labels_by_t: Dict[int, List[int]] = {}
    for key in data.files:
        if not key.startswith("X_"):
            continue
        t = int(key.split("_", 1)[1])
        features_by_t[t] = [row for row in data[key]]
        labels_by_t[t] = [int(value) for value in data[f"y_{t}"]]

    metrics = run_all_probes(features_by_t, labels_by_t, seed=int(config["seeds"][0]))
    out = output_path(config, "probes") / "probe_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_out = _record_manifest(
        config,
        "train_probe",
        sys.argv,
        {"probe_metrics": str(out), "features": str(features_path)},
        split_manifest=config["dataset"].get("split_manifest"),
    )
    print(json.dumps({"probe_metrics": str(out), "manifest": str(manifest_out)}))
    return 0


def cmd_baseline_eval(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_output_dirs(config, ["baselines", "manifests"])
    rollouts = list(read_jsonl(args.rollouts or output_path(config, "rollouts")))
    metrics = {
        "overall": aggregate_binary_outcomes(rollouts),
        "problem_grouped": grouped_problem_accuracy(rollouts),
    }
    if any("score" in row for row in rollouts):
        metrics["risk_coverage"] = risk_coverage_curve(rollouts)
    out = output_path(config, "baselines")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_out = _record_manifest(
        config,
        "baseline_eval",
        sys.argv,
        {"baselines": str(out)},
        split_manifest=config["dataset"].get("split_manifest"),
    )
    print(json.dumps({"baselines": str(out), "manifest": str(manifest_out)}))
    return 0


def _load_scores(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    scores: Dict[str, float] = {}
    for row in read_jsonl(path):
        scores[str(row["rollout_id"])] = float(row["score"])
    return scores


def cmd_intervention_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_output_dirs(config, ["interventions", "manifests"])
    from .interventions import decide_intervention

    score_file = args.scores or config["intervention"].get("score_file")
    scores = _load_scores(score_file)
    decisions = []
    for row in read_jsonl(args.rollouts or output_path(config, "rollouts")):
        score = float(scores.get(str(row.get("rollout_id")), row.get("score", 1.0)))
        decision = decide_intervention(score, config)
        payload = dict(row)
        payload.update(
            {
                "score": score,
                "triggered": decision.triggered,
                "intervention_policy": decision.policy,
                "intervention_prompt_id": decision.prompt_id,
                "intervention_reason": decision.reason,
                "intervention_prompt_text": decision.prompt_text,
            }
        )
        decisions.append(payload)

    out = output_path(config, "interventions")
    write_jsonl(decisions, out)
    metrics_out = out.with_suffix(".metrics.json")
    metrics_out.write_text(
        json.dumps(
            {
                "intervention": intervention_confusion(decisions),
                "risk_coverage": risk_coverage_curve(decisions),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_out = _record_manifest(
        config,
        "intervention_run",
        sys.argv,
        {"interventions": str(out), "metrics": str(metrics_out), "scores": score_file},
        split_manifest=config["dataset"].get("split_manifest"),
    )
    print(json.dumps({"interventions": str(out), "metrics": str(metrics_out), "manifest": str(manifest_out)}))
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_output_dirs(config, ["figures", "manifests"])
    metrics_path = Path(args.metrics or output_path(config, "baselines"))
    fig_dir = output_path(config, "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "risk_coverage.png"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    curve = payload.get("risk_coverage") or payload.get("intervention", {}).get("risk_coverage")
    if not curve:
        summary = fig_dir / "plot_skipped.json"
        summary.write_text(
            json.dumps({"skipped": True, "reason": "no risk_coverage curve"}, indent=2) + "\n",
            encoding="utf-8",
        )
        artifact = str(summary)
    else:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            summary = fig_dir / "plot_skipped.json"
            summary.write_text(
                json.dumps(
                    {
                        "skipped": True,
                        "reason": "matplotlib is not installed",
                        "points": len(curve),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            artifact = str(summary)
            manifest_out = _record_manifest(
                config,
                "plot",
                sys.argv,
                {"figure_or_summary": artifact},
                split_manifest=config["dataset"].get("split_manifest"),
            )
            print(json.dumps({"artifact": artifact, "manifest": str(manifest_out)}))
            return 0

        xs = [point["coverage"] for point in curve]
        ys = [point["risk"] for point in curve]
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.plot(xs, ys, marker="o", linewidth=1.5)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Risk")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        artifact = str(out)
    manifest_out = _record_manifest(
        config,
        "plot",
        sys.argv,
        {"figure_or_summary": artifact},
        split_manifest=config["dataset"].get("split_manifest"),
    )
    print(json.dumps({"artifact": artifact, "manifest": str(manifest_out)}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Latent correctness probe experiment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", required=True, help="Path to YAML run config")

    make_splits = subparsers.add_parser("make-splits", help="Create problem-disjoint split manifest")
    add_config(make_splits)
    make_splits.add_argument("--output", help="Override output split manifest path")
    make_splits.set_defaults(func=cmd_make_splits)

    generation = subparsers.add_parser("generation", help="Generate rollout JSONL")
    add_config(generation)
    generation.add_argument("--split", choices=["train", "validation", "final_test"], default=None)
    generation.add_argument("--dry-run", action="store_true")
    generation.set_defaults(func=cmd_generation)

    extract = subparsers.add_parser("hidden-state-extraction", help="Extract prefix hidden states")
    add_config(extract)
    extract.add_argument("--rollouts")
    extract.add_argument("--dry-run", action="store_true")
    extract.set_defaults(func=cmd_extract)

    train_probe = subparsers.add_parser("probe-training", help="Train PCA + logistic probes")
    add_config(train_probe)
    train_probe.add_argument("--features")
    train_probe.set_defaults(func=cmd_train_probe)

    baseline = subparsers.add_parser("baseline-evaluation", help="Evaluate rollout baselines")
    add_config(baseline)
    baseline.add_argument("--rollouts")
    baseline.set_defaults(func=cmd_baseline_eval)

    intervention = subparsers.add_parser("intervention-run", help="Apply validation-locked trigger policy")
    add_config(intervention)
    intervention.add_argument("--rollouts")
    intervention.add_argument("--scores", help="JSONL with rollout_id and score")
    intervention.set_defaults(func=cmd_intervention_run)

    plot = subparsers.add_parser("plotting", help="Plot saved metrics")
    add_config(plot)
    plot.add_argument("--metrics")
    plot.set_defaults(func=cmd_plot)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
