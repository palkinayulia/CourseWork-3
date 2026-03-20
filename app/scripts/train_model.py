import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from pathlib import Path


# ===== 1. Загружаем исторический датасет =====
df = pd.read_csv("data/hr_dataset.csv")  # сюда потом положим IBM HR dataset

# ===== 2. Готовим признаки =====
# допустим target = Attrition (No = 1, Yes = 0)

df["target"] = df["Attrition"].map({"No": 1, "Yes": 0})

features = [
    "TotalWorkingYears",
    "YearsAtCompany",
    "JobSatisfaction",
    "PerformanceRating",
    "MonthlyIncome",
]
X = df[features]
y = df["target"]
# Делим train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
#Создаём pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression())
])
# Обучаем
model.fit(X_train, y_train)
# Проверяем
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
# Сохраняем модель
model_dir = Path("data/models")
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_dir / "model.pkl")

print("Модель сохранена в data/models/model.pkl")