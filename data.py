import re
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset

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

BOX_RE = re.compile(r"\\boxed\\{([^}]*)\\}")
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
    Return the final \\boxed{...} group from a MATH solution if present.
    """
    matches = BOX_RE.findall(solution)
    if not matches:
        return None
    return matches[-1].strip()


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
) -> List[Dict[str, Any]]:
    """
    Load Hendrycks MATH problems and extract gold answers.

    Args:
        split: "train" or "test".
        subjects: iterable of subject names to include. Defaults to all.
        min_level: minimum difficulty level (1-5 inclusive).
        require_numeric: drop examples whose simplified gold answer is non-numeric.
    """
    if subjects is None:
        subjects = MATH_SUBJECTS

    examples: List[Dict[str, Any]] = []
    for subject in subjects:
        dataset = load_dataset(
            "EleutherAI/hendrycks_math", name=subject, split=split, trust_remote_code=True
        )
        for row in dataset:
            level_str = str(row["level"]).strip()
            try:
                level = int(level_str)
            except ValueError:
                level = 0
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
    return examples


def gold_answer_str(example: Dict[str, Any]) -> Optional[str]:
    """
    Compatibility helper for downstream modules.
    """
    return example.get("gold_answer_norm")
