from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

BASE_DIR = Path(__file__).resolve().parents[1]  # папка app
DATA_DIR = BASE_DIR / "data"

RAW_PATH = DATA_DIR / "raw" / "ibm_hr.csv"
MODEL_PATH = DATA_DIR / "models" / "model.pkl"

def map_education_level(edu_num: int) -> str:
    """
    В IBM датасете Education обычно как число 1..5 (уровни образования).
    Преобразуем в категории, чтобы совпадало шаблоном.
    """
    mapping = {
        1: "college",   # условно Below College
        2: "college",
        3: "bachelor",
        4: "master",
        5: "phd",
    }
    return mapping.get(int(edu_num), "bachelor")


def prepare_ibm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # target: успешность = 1 если НЕ ушёл (Attrition == "No"), иначе 0
    df["success_label"] = df["Attrition"].map({"No": 1, "Yes": 0})
    # унифицированные признаки под систему
    df["years_experience"] = df["TotalWorkingYears"].astype(float)
    df["education_level"] = df["Education"].apply(map_education_level)
    # PerformanceRating обычно 1..4 -> приведём к 0..100 (как test_score)
    df["test_score"] = (df["PerformanceRating"].astype(float) / 4.0) * 100.0
    # JobSatisfaction обычно 1..4 -> приведём к 0..10 (как interview_score)
    df["interview_score"] = (df["JobSatisfaction"].astype(float) / 4.0) * 10.0
    # YearsAtCompany (годы) -> месяцы
    df["avg_tenure_months"] = df["YearsAtCompany"].astype(float) * 12.0
    # оставим только нужное
    out = df[[
        "years_experience",
        "education_level",
        "test_score",
        "interview_score",
        "avg_tenure_months",
        "success_label"
    ]].dropna()

    return out

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Не найден файл: {RAW_PATH}.")
    df_raw = pd.read_csv(RAW_PATH)
    df = prepare_ibm(df_raw)
    X = df[["years_experience", "education_level", "test_score", "interview_score", "avg_tenure_months"]]
    y = df["success_label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    numeric_features = ["years_experience", "test_score", "interview_score", "avg_tenure_months"]
    categorical_features = ["education_level"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=200))
    ])
    model.fit(X_train, y_train)
    # оценка
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
    print(classification_report(y_test, preds))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Модель сохранена: {MODEL_PATH}")

if __name__ == "__main__":
    main()