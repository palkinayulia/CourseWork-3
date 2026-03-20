from __future__ import annotations
from typing import Dict


CRITERIA = ["C1", "C2", "C3", "C4", "C5", "C6"]


# 🔧 Исправлено: теперь соответствует scoring_service
DEFAULT_CRITERIA_LABELS = {
    "C1": "Опыт работы",
    "C2": "Совпадение навыков",
    "C3": "Результат теста",
    "C4": "Результат интервью",
    "C5": "Образование",
    "C6": "Стабильность",
}


DEFAULT_PROFILES: Dict[str, Dict[str, float]] = {
    "IT": {
        "C1": 0.15,
        "C2": 0.35,
        "C3": 0.25,
        "C4": 0.15,
        "C5": 0.05,
        "C6": 0.05,
    },
    "Sales": {
        "C1": 0.20,
        "C2": 0.20,
        "C3": 0.10,
        "C4": 0.35,
        "C5": 0.05,
        "C6": 0.10,
    },
    "Office": {
        "C1": 0.25,
        "C2": 0.25,
        "C3": 0.15,
        "C4": 0.20,
        "C5": 0.10,
        "C6": 0.05,
    },
}


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:

    # 1. гарантируем наличие всех критериев
    safe_weights = {}
    for c in CRITERIA:
        val = _safe_float(weights.get(c, 0.0))
        safe_weights[c] = max(0.0, val)  # без отрицательных

    # 2. нормализация
    total = sum(safe_weights.values())

    if total <= 0:
        return {c: 1.0 / len(CRITERIA) for c in CRITERIA}

    normalized = {k: v / total for k, v in safe_weights.items()}

    return normalized


def weights_to_percent(weights: Dict[str, float]) -> Dict[str, float]:

    return {k: round(v * 100, 1) for k, v in weights.items()}