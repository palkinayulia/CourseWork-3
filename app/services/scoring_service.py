from __future__ import annotations
from typing import Dict
import pandas as pd

def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
def _edu_score(education_level: str, has_relevant_degree: int | None) -> float:
    edu = (education_level or "").strip().lower()

    base_map = {
        "school": 40,
        "college": 55,
        "bachelor": 70,
        "master": 85,
        "phd": 95,
    }

    base = float(base_map.get(edu, 60))
    bonus = 10.0 if has_relevant_degree == 1 else 0.0

    return min(base + bonus, 100.0)


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))
def compute_criteria(row: pd.Series) -> Dict[str, float]:
    years = _safe_float(row.get("years_experience"))
    skills = _safe_float(row.get("skills_match_percent"))
    test = _safe_float(row.get("test_score"))
    interview = _safe_float(row.get("interview_score"))
    tenure = _safe_float(row.get("avg_tenure_months"))
    has_deg = _safe_float(row.get("major_relevant"))
    c1 = _clamp(years / 5.0 * 100.0)  # C1 опыт
    c2 = _clamp(skills) # C2 навыки
    c3 = _clamp(test) # C3 тест
    c4 = _clamp(interview) # C4 интервью (0..100)
    c5 = _edu_score(str(row.get("education_level")), int(has_deg) if has_deg in [0, 1] else None) # C5 образование
    c6 = _clamp(tenure / 24.0 * 100.0) if tenure > 0 else 50.0  # C6 стабильность
    return {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6}

def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        # fallback — равномерные веса
        return {k: 1 / len(weights) for k in weights}

    return {k: v / total for k, v in weights.items()}


def criteria_score(criteria: Dict[str, float], weights: Dict[str, float]) -> float:
    weights = normalize_weights(weights)
    return sum(criteria[k] * weights.get(k, 0.0) for k in criteria.keys())


def build_results(
    df: pd.DataFrame,
    weights: Dict[str, float],
    alpha: float = 0.7,
    base_scores_from_ml: list[float] | None = None
) -> pd.DataFrame:

    out = df.copy()

    # защита alpha
    alpha = max(0.0, min(1.0, float(alpha)))

    weights = normalize_weights(weights)

    criteria_scores = []
    base_scores = []
    final_scores = []
    top_factors = []

    for idx, row in out.iterrows():
        try:
            c = compute_criteria(row)
            cs = criteria_score(c, weights)

            # ML или fallback
            if base_scores_from_ml and idx < len(base_scores_from_ml):
                base = _clamp(_safe_float(base_scores_from_ml[idx]))
            else:
                base = (c["C2"] + c["C3"] + c["C4"]) / 3.0

            final = alpha * base + (1 - alpha) * cs

            top = sorted(c.items(), key=lambda x: x[1], reverse=True)[:3]
            top_txt = ", ".join([f"{k}:{v:.0f}" for k, v in top])

        except Exception:
            # fallback если строка битая
            cs = 0.0
            base = 0.0
            final = 0.0
            top_txt = "ошибка расчета"

        criteria_scores.append(cs)
        base_scores.append(base)
        final_scores.append(final)
        top_factors.append(top_txt)

    out["base_score"] = [round(x, 1) for x in base_scores]
    out["criteria_score"] = [round(x, 1) for x in criteria_scores]
    out["final_score"] = [round(x, 1) for x in final_scores]
    out["top_factors"] = top_factors

    return out