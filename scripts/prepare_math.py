import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import MATH_SUBJECTS, load_math_split


def parse_subjects(subjects_arg: Optional[List[str]]) -> List[str]:
    if not subjects_arg:
        return list(MATH_SUBJECTS)
    return list(subjects_arg)


def write_split(
    split: str,
    out_path: Path,
    subjects: Iterable[str],
    min_level: int,
    require_numeric: bool,
    max_items: Optional[int],
) -> None:
    data = load_math_split(
        split=split,
        subjects=subjects,
        min_level=min_level,
        require_numeric=require_numeric,
    )
    if max_items is not None:
        data = data[:max_items]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in data:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(data)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Hendrycks MATH JSONL dumps.")
    parser.add_argument("--out-train", default="math_all_train.jsonl")
    parser.add_argument("--out-test", default="math_all_test.jsonl")
    parser.add_argument("--min-level", type=int, default=1, help="Minimum difficulty level (1-5).")
    parser.add_argument(
        "--keep-non-numeric",
        action="store_true",
        help="Keep problems whose gold answer cannot be parsed as numeric.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="Subset of subjects to include. Defaults to all benchmark subjects.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap on number of items written per split.",
    )
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    require_numeric = not args.keep_non_numeric

    write_split(
        split="train",
        out_path=Path(args.out_train),
        subjects=subjects,
        min_level=args.min_level,
        require_numeric=require_numeric,
        max_items=args.max_items,
    )
    write_split(
        split="test",
        out_path=Path(args.out_test),
        subjects=subjects,
        min_level=args.min_level,
        require_numeric=require_numeric,
        max_items=args.max_items,
    )


if __name__ == "__main__":
    main()
