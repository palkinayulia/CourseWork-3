from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


MAX_ROWS = 10000


REQUIRED_COLUMNS = [
    "candidate_id",
    "full_name",
    "vacancy",
    "years_experience",
    "education_level",
    "skills_match_percent",
    "test_score",
    "interview_score",
    "major_relevant",
    "avg_tenure_months",
    "salary_expectation",
    "vacancy_budget",
    "motivation_score",
]


RUS_TO_INTERNAL_COLUMNS = {
    "id кандидата": "candidate_id",
    "фио": "full_name",
    "вакансия": "vacancy",
    "опыт работы (лет)": "years_experience",
    "уровень образования": "education_level",
    "совпадение навыков (%)": "skills_match_percent",
    "балл теста": "test_score",
    "балл интервью": "interview_score",
    "профильное образование (0/1)": "major_relevant",
    "средний срок работы (мес.)": "avg_tenure_months",
    "зарплатные ожидания": "salary_expectation",
    "бюджет вакансии": "vacancy_budget",
    "мотивация (0-100)": "motivation_score",
}


EDUCATION_MAP = {
    "школа": "school",
    "колледж": "college",
    "бакалавр": "bachelor",
    "магистр": "master",
    "кандидат наук": "phd",
    "доктор наук": "phd",
    "phd": "phd",
    "bachelor": "bachelor",
    "master": "master",
    "college": "college",
    "school": "school",
}

VALID_EDUCATION = {"school", "college", "bachelor", "master", "phd"}


@dataclass
class ValidationResult:
    ok: bool
    df: pd.DataFrame
    errors: list[str]


def normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace("ё", "е").replace("\n", " ")


def rename_russian_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        normalized = normalize_column_name(col)
        if normalized in RUS_TO_INTERNAL_COLUMNS:
            rename_map[col] = RUS_TO_INTERNAL_COLUMNS[normalized]
    return df.rename(columns=rename_map)


def normalize_education_values(df: pd.DataFrame) -> pd.DataFrame:
    if "education_level" not in df.columns:
        return df

    def convert(value):
        if pd.isna(value):
            return value
        normalized = str(value).strip().lower().replace("ё", "е")
        return EDUCATION_MAP.get(normalized, normalized)

    df["education_level"] = df["education_level"].apply(convert)
    return df


def normalize_boolean(series: pd.Series) -> pd.Series:
    mapping = {
        "1": 1, "0": 0,
        "да": 1, "нет": 0,
        "yes": 1, "no": 0,
        "true": 1, "false": 0
    }

    def convert(v):
        if pd.isna(v):
            return v
        s = str(v).strip().lower()
        return mapping.get(s, v)

    return series.apply(convert)


def validate_candidates_df(df: pd.DataFrame) -> ValidationResult:
    errors = []

    df = rename_russian_columns(df).copy()
    df = normalize_education_values(df)

    if df.shape[0] > MAX_ROWS:
        errors.append(f"Файл слишком большой (максимум {MAX_ROWS} строк).")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        errors.append("Отсутствуют обязательные колонки: " + ", ".join(missing))
        return ValidationResult(False, df, errors)

    if df.empty:
        errors.append("Файл не содержит данных.")
        return ValidationResult(False, df, errors)

    # boolean normalize
    df["major_relevant"] = normalize_boolean(df["major_relevant"])

    numeric_columns = [
        "years_experience",
        "skills_match_percent",
        "test_score",
        "interview_score",
        "major_relevant",
        "avg_tenure_months",
        "salary_expectation",
        "vacancy_budget",
        "motivation_score",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Проверка пустых
    for col in numeric_columns:
        bad_rows = df[df[col].isna()].index.tolist()
        if bad_rows:
            errors.append(f"{col}: некорректные значения в строках {bad_rows[:5]}")

    # Проверка диапазонов
    def check_range(col, min_val, max_val):
        bad = df[(df[col] < min_val) | (df[col] > max_val)]
        if not bad.empty:
            errors.append(f"{col}: значения вне диапазона {min_val}-{max_val}")

    check_range("skills_match_percent", 0, 100)
    check_range("test_score", 0, 100)
    check_range("interview_score", 0, 100)
    check_range("motivation_score", 0, 100)
    check_range("major_relevant", 0, 1)

    # Дубликаты
    if df["candidate_id"].duplicated().any():
        errors.append("Найдены дубликаты candidate_id.")

    # Логика зарплаты
    bad_salary = df[df["salary_expectation"] > df["vacancy_budget"] * 3]
    if not bad_salary.empty:
        errors.append("Некоторые кандидаты имеют слишком высокие зарплатные ожидания.")

    # Строковые поля
    for col in ["candidate_id", "full_name", "vacancy"]:
        bad = df[df[col].astype(str).str.strip() == ""]
        if not bad.empty:
            errors.append(f"{col}: есть пустые значения.")

    # Образование
    bad_edu = df[~df["education_level"].isin(VALID_EDUCATION)]
    if not bad_edu.empty:
        errors.append("Некорректные значения education_level.")

    return ValidationResult(ok=len(errors) == 0, df=df, errors=errors)