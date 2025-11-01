import re
from fractions import Fraction
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence

from datasets import load_dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is an optional nicety
    tqdm = None

# Subjects exposed by the Hendrycks MATH benchmark.
MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

BOX_TRIGGER_RE = re.compile(r"\\?boxed\s*\{", re.IGNORECASE)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
FRAC_RE = re.compile(r"\\frac\{([^}]*)\}\{([^}]*)\}")


def _latex_to_simple(text: str) -> str:
    """
    Lightweight LaTeX normalisation: drop formatting commands and linearise fractions.
    Enough for simple numeric equality checks.
    """
    s = text.strip()
    replacements = [
        ("\\,", ""),
        ("\\ ", ""),
        ("\\left", ""),
        ("\\right", ""),
        ("$", ""),
        (" ", ""),
        ("\n", ""),
        ("\t", ""),
        ("\\%", "%"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    s = FRAC_RE.sub(lambda m: f"({m.group(1)}/{m.group(2)})", s)
    return s


def extract_boxed(solution: str) -> Optional[str]:
    """
    Return the final boxed{...} group from a solution if present. This is intentionally
    permissive: the backslash before "boxed" is optional and the substring extends until
    the next closing brace, or the end of the string if none is found.
    """
    if not solution:
        return None
    matches: List[str] = []
    for match in BOX_TRIGGER_RE.finditer(solution):
        start_idx = match.end()
        close_idx = solution.find("}", start_idx)
        if close_idx == -1:
            content = solution[start_idx:].strip()
        else:
            content = solution[start_idx:close_idx].strip()
        if content:
            matches.append(content)
    if not matches:
        return None
    return matches[-1]


def normalize_gold_answer(solution: str) -> Dict[str, Optional[str]]:
    """
    Extract both raw and simplified representations of the gold answer.
    Falls back to the last numeric literal if no boxed answer exists.
    """
    boxed = extract_boxed(solution)
    if boxed:
        simple = _latex_to_simple(boxed)
        return {"raw": boxed, "norm": simple}

    nums = NUM_RE.findall(solution)
    if nums:
        num = nums[-1]
        return {"raw": num, "norm": num}
    return {"raw": None, "norm": None}


def extract_last_number(text: str) -> Optional[str]:
    """
    Convenience helper for parsing model predictions (typically plain text).
    """
    matches = NUM_RE.findall(text)
    if not matches:
        return None
    return matches[-1]


def normalize_num_str(value: Optional[str]) -> Optional[str]:
    """
    Normalise numeric strings for equality comparison.
    """
    if value is None:
        return None
    s = value.strip()
    try:
        return str(float(s))
    except Exception:
        pass

    for ch in "()[]{}":
        s = s.replace(ch, "")
    s = s.replace(" ", "")
    if "/" in s:
        try:
            frac = Fraction(s)
            return str(float(frac))
        except Exception:
            return None
    return None


def parse_model_answer(text: str) -> Dict[str, Optional[str]]:
    """
    Parse a model's generated answer, supporting boxed LaTeX or plain numerics.
    """
    boxed = extract_boxed(text)
    if boxed:
        simple = _latex_to_simple(boxed)
        return {"raw": boxed, "norm": normalize_num_str(simple)}

    num = extract_last_number(text)
    if num is None:
        return {"raw": None, "norm": None}
    return {"raw": num, "norm": normalize_num_str(num)}


def load_math_split(
    split: str,
    subjects: Optional[Iterable[str]] = None,
    min_level: int = 1,
    require_numeric: bool = True,
    show_progress: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load Hendrycks MATH problems and extract gold answers.

    Args:
        split: "train" or "test".
        subjects: iterable of subject names to include. Defaults to all.
        min_level: minimum difficulty level (1-5 inclusive).
        require_numeric: drop examples whose simplified gold answer is non-numeric.
        show_progress: if True, render tqdm progress bars while iterating.
    """
    if subjects is None:
        subjects = MATH_SUBJECTS

    examples: List[Dict[str, Any]] = []
    subjects_list = list(subjects)
    use_progress = show_progress and tqdm is not None
    subject_iterable = (
        tqdm(subjects_list, desc=f"{split} subjects", unit="subject")
        if use_progress
        else subjects_list
    )
    for subject in subject_iterable:
        dataset = load_dataset("EleutherAI/hendrycks_math", name=subject, split=split)
        row_iterable = (
            tqdm(
                dataset,
                total=len(dataset),
                desc=f"{subject} ({split})",
                unit="example",
                leave=False,
            )
            if use_progress
            else dataset
        )
        for row in row_iterable:
            level_str = str(row["level"]).strip()
            match = re.search(r"\d+", level_str)
            level = int(match.group(0)) if match else 0
            if level < min_level:
                continue

            gold = normalize_gold_answer(row["solution"])
            gold_norm = normalize_num_str(gold["norm"]) if gold["norm"] else None
            if require_numeric and gold_norm is None:
                continue

            examples.append(
                {
                    "problem": row["problem"],
                    "solution": row["solution"],
                    "subject": subject,
                    "level": level,
                    "gold_answer_raw": gold["raw"],
                    "gold_answer_norm": gold_norm,
                }
            )
        if use_progress:
            row_iterable.close()
    if use_progress:
        subject_iterable.close()
    return examples


def sample_balanced_by_difficulty(
    examples: List[Dict[str, Any]],
    total: int,
    easy_levels: Sequence[int],
    hard_levels: Sequence[int],
    seed: int = 356,
) -> List[Dict[str, Any]]:
    """
    Sample a difficulty-balanced subset from the provided examples.

    Args:
        examples: Full list of MATH problems after filtering.
        total: Total number of problems to return. Must be even so easy/hard split equally.
        easy_levels: Difficulty levels considered "easy".
        hard_levels: Difficulty levels considered "hard".
        seed: RNG seed for reproducible sampling.
    """

    if total % 2 != 0:
        raise ValueError("total must be even to split evenly across difficulties")
    per_bucket = total // 2

    easy_pool = [ex for ex in examples if ex.get("level") in easy_levels]
    hard_pool = [ex for ex in examples if ex.get("level") in hard_levels]

    if len(easy_pool) < per_bucket:
        raise ValueError(
            f"Insufficient easy examples (wanted {per_bucket}, found {len(easy_pool)})"
        )
    if len(hard_pool) < per_bucket:
        raise ValueError(
            f"Insufficient hard examples (wanted {per_bucket}, found {len(hard_pool)})"
        )

    rng = random.Random(seed)
    easy_subset = rng.sample(easy_pool, per_bucket)
    hard_subset = rng.sample(hard_pool, per_bucket)

    selected: List[Dict[str, Any]] = []
    for difficulty, bucket in (("easy", easy_subset), ("hard", hard_subset)):
        for ex in bucket:
            copy = dict(ex)
            copy["difficulty_bin"] = difficulty
            selected.append(copy)

    rng.shuffle(selected)
    return selected


def gold_answer_str(example: Dict[str, Any]) -> Optional[str]:
    """
    Compatibility helper for downstream modules.
    """
    return example.get("gold_answer_norm")
