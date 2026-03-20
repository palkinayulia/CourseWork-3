from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd


# путь к модели относительно app/
BASE_DIR = Path(__file__).resolve().parents[1]  # app/
MODEL_PATH = BASE_DIR / "data" / "models" / "model.pkl"


class MLModel:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"ML модель не найдена: {model_path}. Сначала запусти обучение.")
        self.model = joblib.load(model_path)

    def predict_proba_success(self, df: pd.DataFrame) -> list[float]:
        """
        Возвращает вероятность успеха (класс 1) для каждой строки df.
        df должен содержать колонки:
        years_experience, education_level, test_score, interview_score, avg_tenure_months
        """
        X = df[["years_experience", "education_level", "test_score", "interview_score", "avg_tenure_months"]].copy()

        # если avg_tenure_months отсутствует (None) — поставим нейтрально 12 месяцев
        if "avg_tenure_months" in X.columns:
            X["avg_tenure_months"] = X["avg_tenure_months"].fillna(12.0)

        proba = self.model.predict_proba(X)[:, 1]
        return proba.tolist()