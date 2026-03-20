# uvicorn app.main:app --reload
# если запускаешь так:
# uvicorn app.main:app --reload --port 8001
# то открываешь всё только на http://localhost:8001/
# hr_hse 12345hse

from pathlib import Path
import json
from io import BytesIO
from uuid import uuid4
from reportlab.pdfbase.pdfmetrics import registerFontFamily
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import (
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.auth import hash_password, verify_password
from app.db import get_db
from app.init_db import init_db
from app.models import Analysis, Upload, User, Profile, ProfileCriterion
from app.services.file_readers import read_candidates_file
from app.services.file_service import validate_candidates_df
from app.services.ml_service import MLModel
from app.services.scoring_service import build_results
from app.services.weights_service import (
    CRITERIA,
    DEFAULT_CRITERIA_LABELS,
    DEFAULT_PROFILES,
    normalize_weights,
)

from datetime import datetime
import math
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)

app = FastAPI(title="Recruitment Analytics")
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
init_db()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ml_model = MLModel()

DEFAULT_ALPHA = 0.7
ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".csv"}
MAX_FILENAME_LENGTH = 255
MIN_PASSWORD_LENGTH = 5

COLUMN_LABELS_RU = {
    "candidate_id": "ID кандидата",
    "full_name": "ФИО",
    "vacancy": "Вакансия",
    "years_experience": "Опыт работы (лет)",
    "education_level": "Уровень образования",
    "skills_match_percent": "Совпадение навыков (%)",
    "test_score": "Балл теста",
    "interview_score": "Балл интервью",
    "major_relevant": "Профильное образование",
    "avg_tenure_months": "Средний срок работы (мес.)",
    "salary_expectation": "Зарплатные ожидания",
    "vacancy_budget": "Бюджет вакансии",
    "motivation_score": "Мотивация",
    "base_score": "ML-оценка",
    "criteria_score": "Оценка по критериям",
    "final_score": "Итоговый балл",
    "risk_level": "Уровень риска",
}


def render_upload_page(
    request: Request,
    db: Session,
    error: str | None = None,
    errors: list[str] | None = None,
    profile: str | None = None,
    weights: dict[str, float] | None = None,
):
    current_profile = profile or get_default_profile_name(db)
    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "page_title": "Запуск анализа кандидатов",
            "error": error,
            "errors": errors or [],
            "profile": current_profile,
            "profiles": get_profile_names(db),
            "weights": weights or profile_to_percent_weights(db, current_profile),
            "alpha": DEFAULT_ALPHA,
            "criteria_labels": get_profile_labels(db, current_profile),
        },
    )


def render_admin_users_page(
    request: Request,
    db: Session,
    error: str | None = None,
    ok: str | None = None,
):
    users = db.query(User).order_by(User.id.asc()).all()
    return templates.TemplateResponse(
        "admin_users.html",
        {
            "request": request,
            "page_title": "Управление пользователями",
            "users": users,
            "error": error,
            "ok": ok,
        },
    )


def render_admin_profiles_page(
    request: Request,
    db: Session,
    error: str | None = None,
    ok: str | None = None,
):
    seed_default_profiles(db)
    profiles = db.query(Profile).order_by(Profile.name.asc()).all()
    return templates.TemplateResponse(
        "admin_profiles.html",
        {
            "request": request,
            "page_title": "Профили и критерии",
            "profiles": profiles,
            "error": error,
            "ok": ok,
        },
    )


def safe_json_loads(value: str | None, fallback):
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback


def get_current_user(request: Request, db: Session) -> User | None:
    username = request.cookies.get("username")
    if not username:
        return None
    return db.query(User).filter(User.username == username).first()


def require_login(request: Request) -> bool:
    return bool(request.cookies.get("username"))


def require_admin(request: Request) -> bool:
    return request.cookies.get("role") == "ADMIN"
def register_pdf_fonts():
    fonts_dir = BASE_DIR / "static" / "fonts"

    regular_font_path = fonts_dir / "DejaVuSans.ttf"
    bold_font_path = fonts_dir / "DejaVuSans-Bold.ttf"

    if not regular_font_path.exists():
        raise FileNotFoundError(f"Не найден шрифт: {regular_font_path}")

    if not bold_font_path.exists():
        raise FileNotFoundError(f"Не найден шрифт: {bold_font_path}")

    if "AppRegular" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("AppRegular", str(regular_font_path)))

    if "AppBold" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("AppBold", str(bold_font_path)))

    registerFontFamily(
        "AppFont",
        normal="AppRegular",
        bold="AppBold",
    )

    return "AppRegular", "AppBold"

    return regular_name, bold_name

def seed_default_profiles(db: Session) -> None:
    exists = db.query(Profile).count()
    if exists > 0:
        return

    for profile_name, weights in DEFAULT_PROFILES.items():
        profile = Profile(name=profile_name)
        db.add(profile)
        db.flush()

        for idx, code in enumerate(CRITERIA, start=1):
            criterion = ProfileCriterion(
                profile_id=profile.id,
                code=code,
                label=DEFAULT_CRITERIA_LABELS[code],
                weight=float(weights.get(code, 0.0)),
                sort_order=idx,
                is_active=True,
            )
            db.add(criterion)

    db.commit()


def get_profile_names(db: Session) -> list[str]:
    seed_default_profiles(db)
    return [p.name for p in db.query(Profile).order_by(Profile.name.asc()).all()]


def get_default_profile_name(db: Session) -> str:
    names = get_profile_names(db)
    if "IT" in names:
        return "IT"
    return names[0] if names else "IT"


def get_profile_obj_by_name(db: Session, profile_name: str) -> Profile | None:
    seed_default_profiles(db)
    return db.query(Profile).filter(Profile.name == profile_name).first()


def get_profile_weights(db: Session, profile_name: str) -> dict[str, float]:
    profile = get_profile_obj_by_name(db, profile_name)
    if not profile:
        profile = get_profile_obj_by_name(db, get_default_profile_name(db))

    weights = {c.code: float(c.weight) for c in profile.criteria if c.code in CRITERIA}
    for code in CRITERIA:
        weights.setdefault(code, 0.0)

    return normalize_weights(weights)


def get_profile_labels(db: Session, profile_name: str) -> dict[str, str]:
    profile = get_profile_obj_by_name(db, profile_name)
    labels = DEFAULT_CRITERIA_LABELS.copy()
    if not profile:
        return labels

    for c in profile.criteria:
        if c.code in labels:
            labels[c.code] = c.label
    return labels


def profile_to_percent_weights(db: Session, profile_name: str) -> dict[str, float]:
    source = get_profile_weights(db, profile_name)
    return {k: round(float(v) * 100, 1) for k, v in source.items()}


def build_weight_inputs(
    profile: str,
    w_c1: float | None = None,
    w_c2: float | None = None,
    w_c3: float | None = None,
    w_c4: float | None = None,
    w_c5: float | None = None,
    w_c6: float | None = None,
    db: Session | None = None,
) -> dict[str, float]:
    if all(v is not None for v in [w_c1, w_c2, w_c3, w_c4, w_c5, w_c6]):
        return {
            "C1": float(w_c1),
            "C2": float(w_c2),
            "C3": float(w_c3),
            "C4": float(w_c4),
            "C5": float(w_c5),
            "C6": float(w_c6),
        }
    if db is not None:
        return profile_to_percent_weights(db, profile)
    return {c: round(100 / len(CRITERIA), 1) for c in CRITERIA}


def sanitize_filename(filename: str) -> str:
    original = (filename or "").strip()
    if not original:
        raise ValueError("Файл не выбран.")

    safe_name = Path(original).name
    if len(safe_name) > MAX_FILENAME_LENGTH:
        raise ValueError("Имя файла слишком длинное.")

    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Поддерживаются только файлы .xlsx, .xls и .csv.")

    return safe_name


def make_unique_upload_path(filename: str) -> Path:
    ext = Path(filename).suffix.lower()
    stem = Path(filename).stem
    unique_name = f"{stem}_{uuid4().hex}{ext}"
    return UPLOAD_DIR / unique_name


def validate_weight_values(percent_weights: dict[str, float]) -> list[str]:
    errors = []

    for code, value in percent_weights.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            errors.append(f"Вес {code} должен быть числом.")
            continue

        if numeric < 0:
            errors.append(f"Вес {code} не может быть отрицательным.")

    total = sum(float(v) for v in percent_weights.values() if isinstance(v, (int, float)))
    if total <= 0:
        errors.append("Сумма весов должна быть больше 0.")

    return errors

def safe_float(value, default=0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def education_to_ru(value: str) -> str:
    mapping = {
        "school": "школа",
        "college": "колледж",
        "bachelor": "бакалавр",
        "master": "магистр",
        "phd": "кандидат наук",
    }
    if value is None:
        return "—"
    return mapping.get(str(value).strip().lower(), str(value))


def risk_to_ru(value: str) -> str:
    mapping = {
        "low": "Низкий",
        "medium": "Средний",
        "high": "Высокий",
        "низкий": "Низкий",
        "средний": "Средний",
        "высокий": "Высокий",
    }
    if value is None or value == "":
        return "—"
    return mapping.get(str(value).strip().lower(), str(value))


def build_score_distribution(rows: list[dict]) -> list[int]:
    buckets = [0, 0, 0, 0]
    for row in rows:
        score = safe_float(row.get("final_score"), 0)
        if score < 40:
            buckets[0] += 1
        elif score < 60:
            buckets[1] += 1
        elif score < 80:
            buckets[2] += 1
        else:
            buckets[3] += 1
    return buckets


def build_score_distribution_chart(rows: list[dict]) -> BytesIO:
    data = build_score_distribution(rows)
    labels = ["0-39", "40-59", "60-79", "80-100"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, data)
    ax.set_title("Распределение итоговых баллов")
    ax.set_xlabel("Диапазон баллов")
    ax.set_ylabel("Количество кандидатов")
    ax.set_ylim(bottom=0)

    for i, v in enumerate(data):
        ax.text(i, v + 0.05, str(v), ha="center", va="bottom", fontsize=10)

    fig.tight_layout()

    img = BytesIO()
    fig.savefig(img, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    img.seek(0)
    return img


def build_weights_chart(weights: dict, criteria_labels: dict) -> BytesIO:
    codes = ["C1", "C2", "C3", "C4", "C5", "C6"]
    labels = [criteria_labels.get(code, code) for code in codes]
    values = [safe_float(weights.get(code), 0) for code in codes]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    y_pos = range(len(labels))
    ax.barh(list(y_pos), values)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Вес, %")
    ax.set_title("Веса критериев")

    for i, v in enumerate(values):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)

    fig.tight_layout()

    img = BytesIO()
    fig.savefig(img, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    img.seek(0)
    return img


def build_pdf_report(
    filename: str,
    profile: str,
    rows: list[dict],
    weights: dict,
    criteria_labels: dict,
    avg_score: float,
    candidates_count: int,
    top_candidate: str,
) -> BytesIO:
    regular_font_name, bold_font_name = register_pdf_fonts()
    output = BytesIO()

    doc = SimpleDocTemplate(
        output,
        pagesize=landscape(A4),
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontName=bold_font_name,
        fontSize=20,
        leading=24,
        spaceAfter=8,
        textColor=colors.HexColor("#111827"),
        alignment=0,
    )

    subtitle_style = ParagraphStyle(
        "CustomSubtitle",
        parent=styles["BodyText"],
        fontName=regular_font_name,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#6b7280"),
        spaceAfter=10,
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontName=bold_font_name,
        fontSize=13,
        leading=17,
        spaceAfter=8,
        textColor=colors.HexColor("#1f2937"),
    )

    normal_style = ParagraphStyle(
        "CustomBody",
        parent=styles["BodyText"],
        fontName=regular_font_name,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#374151"),
        spaceAfter=6,
    )

    small_style = ParagraphStyle(
        "CustomSmall",
        parent=styles["BodyText"],
        fontName=regular_font_name,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#6b7280"),
        spaceAfter=4,
    )

    story = []
    generated_at = datetime.now().strftime("%d.%m.%Y %H:%M")

    top_rows = sorted(rows, key=lambda x: safe_float(x.get("final_score"), 0), reverse=True)[:5]
    high_score_count = sum(1 for r in rows if safe_float(r.get("final_score"), 0) >= 80)
    medium_score_count = sum(1 for r in rows if 60 <= safe_float(r.get("final_score"), 0) < 80)
    low_risk_count = sum(
        1 for r in rows if str(r.get("risk_level", "")).strip().lower() in ["low", "низкий"]
    )

    story.append(Paragraph("Отчет по анализу кандидатов", title_style))
    story.append(Paragraph(
        f"Документ сформирован автоматически системой аналитики подбора персонала. "
        f"Дата формирования: {generated_at}.",
        subtitle_style
    ))
    story.append(Spacer(1, 4))

    summary_data = [
        ["Параметр", "Значение"],
        ["Файл", filename or "—"],
        ["Профиль оценки", profile or "—"],
        ["Количество кандидатов", str(candidates_count)],
        ["Средний итоговый балл", f"{safe_float(avg_score):.2f}"],
        ["Лучший кандидат", top_candidate or "—"],
    ]

    summary_table = Table(summary_data, colWidths=[62 * mm, 108 * mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f9fafb")),
        ("FONTNAME", (0, 0), (-1, 0), bold_font_name),
        ("FONTNAME", (0, 1), (0, -1), bold_font_name),
        ("FONTNAME", (1, 1), (1, -1), regular_font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Краткие выводы", heading_style))
    story.append(Paragraph(
        f"В ходе анализа обработано {candidates_count} кандидатов. "
        f"Средний итоговый балл составил {safe_float(avg_score):.2f}. "
        f"Количество кандидатов с высоким итоговым баллом (80 и выше) — {high_score_count}. "
        f"Количество кандидатов со средним итоговым баллом — {medium_score_count}. "
        f"Количество кандидатов с низким уровнем риска — {low_risk_count}. "
        f"Лучший результат показал кандидат {top_candidate or '—'}.",
        normal_style
    ))

    story.append(Paragraph("Использованные веса критериев", heading_style))
    weights_info = []
    for code in ["C1", "C2", "C3", "C4", "C5", "C6"]:
        label = criteria_labels.get(code, code)
        value = safe_float(weights.get(code), 0)
        weights_info.append([label, f"{value:.1f}%"])

    weights_info_table = Table([["Критерий", "Вес"]] + weights_info, colWidths=[130 * mm, 40 * mm])
    weights_info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("FONTNAME", (0, 0), (-1, 0), bold_font_name),
        ("FONTNAME", (0, 1), (-1, -1), regular_font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(weights_info_table)

    story.append(PageBreak())

    story.append(Paragraph("Визуализация результатов анализа", heading_style))
    story.append(Spacer(1, 4))

    score_chart = build_score_distribution_chart(rows)
    weights_chart = build_weights_chart(weights, criteria_labels)

    story.append(Image(score_chart, width=175 * mm, height=88 * mm))
    story.append(Paragraph(
        "График показывает распределение кандидатов по диапазонам итогового балла.",
        small_style
    ))
    story.append(Spacer(1, 6))

    story.append(Image(weights_chart, width=175 * mm, height=92 * mm))
    story.append(Paragraph(
        "График отражает веса критериев, использованные при расчете итоговой оценки.",
        small_style
    ))

    story.append(PageBreak())

    story.append(Paragraph("Топ-5 кандидатов", heading_style))

    top5_table_data = [["Место", "ФИО", "Вакансия", "Итоговый балл", "Риск"]]
    for idx, row in enumerate(top_rows, start=1):
        top5_table_data.append([
            str(idx),
            str(row.get("full_name", "—")),
            str(row.get("vacancy", "—")),
            f"{safe_float(row.get('final_score'), 0):.2f}",
            risk_to_ru(row.get("risk_level", "—")),
        ])

    top5_table = Table(
        top5_table_data,
        colWidths=[18 * mm, 62 * mm, 58 * mm, 28 * mm, 24 * mm]
    )
    top5_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("FONTNAME", (0, 0), (-1, 0), bold_font_name),
        ("FONTNAME", (0, 1), (-1, -1), regular_font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#d1d5db")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(top5_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "В таблице представлены кандидаты с наивысшими итоговыми баллами по результатам анализа.",
        small_style
    ))

    story.append(PageBreak())

    story.append(Paragraph("Полная таблица результатов", heading_style))

    result_header = [
        "ID", "ФИО", "Вакансия", "Опыт", "Образование",
        "Навыки", "Тест", "Интервью", "ML", "Критерии", "Итог", "Риск"
    ]

    result_data = [result_header]
    for row in rows:
        result_data.append([
            str(row.get("candidate_id", "—")),
            str(row.get("full_name", "—")),
            str(row.get("vacancy", "—")),
            str(row.get("years_experience", "—")),
            education_to_ru(row.get("education_level", "—")),
            str(row.get("skills_match_percent", "—")),
            str(row.get("test_score", "—")),
            str(row.get("interview_score", "—")),
            f"{safe_float(row.get('base_score'), 0):.1f}",
            f"{safe_float(row.get('criteria_score'), 0):.1f}",
            f"{safe_float(row.get('final_score'), 0):.1f}",
            risk_to_ru(row.get("risk_level", "—")),
        ])

    result_table = Table(
        result_data,
        repeatRows=1,
        colWidths=[
            12 * mm, 33 * mm, 28 * mm, 13 * mm, 22 * mm, 14 * mm,
            13 * mm, 15 * mm, 13 * mm, 17 * mm, 14 * mm, 18 * mm
        ],
    )
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
        ("FONTNAME", (0, 0), (-1, 0), bold_font_name),
        ("FONTNAME", (0, 1), (-1, -1), regular_font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(result_table)

    doc.build(story)
    output.seek(0)
    return output

@app.get("/api/profile-config/{profile_name}")
def profile_config(
    profile_name: str,
    request: Request,
    db: Session = Depends(get_db),
):
    if not require_login(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    available_profiles = get_profile_names(db)
    if profile_name not in available_profiles:
        profile_name = get_default_profile_name(db)

    return {
        "profile": profile_name,
        "weights": profile_to_percent_weights(db, profile_name),
        "criteria_labels": get_profile_labels(db, profile_name),
    }


@app.get("/", response_class=HTMLResponse)
def upload_page(request: Request, db: Session = Depends(get_db)):
    if not require_login(request):
        return RedirectResponse(url="/login", status_code=302)

    default_profile = get_default_profile_name(db)

    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "page_title": "Запуск анализа кандидатов",
            "profile": default_profile,
            "profiles": get_profile_names(db),
            "weights": profile_to_percent_weights(db, default_profile),
            "alpha": DEFAULT_ALPHA,
            "criteria_labels": get_profile_labels(db, default_profile),
        },
    )


@app.get("/ping", response_class=HTMLResponse)
def ping():
    return "pong"


@app.get("/download-template")
def download_template(request: Request):
    if not require_login(request):
        return RedirectResponse(url="/login", status_code=302)

    template_df = pd.DataFrame([
        {
            "ID кандидата": 1,
            "ФИО": "Иванов Иван Иванович",
            "Вакансия": "Python Developer",
            "Опыт работы (лет)": 3,
            "Уровень образования": "бакалавр",
            "Совпадение навыков (%)": 78,
            "Балл теста": 82,
            "Балл интервью": 75,
            "Профильное образование (0/1)": 1,
            "Средний срок работы (мес.)": 18,
            "Зарплатные ожидания": 120000,
            "Бюджет вакансии": 140000,
            "Мотивация (0-100)": 85,
        },
        {
            "ID кандидата": 2,
            "ФИО": "Петрова Анна Сергеевна",
            "Вакансия": "Data Analyst",
            "Опыт работы (лет)": 5,
            "Уровень образования": "магистр",
            "Совпадение навыков (%)": 91,
            "Балл теста": 88,
            "Балл интервью": 90,
            "Профильное образование (0/1)": 1,
            "Средний срок работы (мес.)": 26,
            "Зарплатные ожидания": 150000,
            "Бюджет вакансии": 160000,
            "Мотивация (0-100)": 92,
        },
    ])

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        template_df.to_excel(writer, index=False, sheet_name="Кандидаты")

    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="template_candidates_ru.xlsx"'},
    )


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/logout")
def logout():
    resp = RedirectResponse(url="/login", status_code=302)
    resp.delete_cookie("username")
    resp.delete_cookie("role")
    return resp


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not require_login(request):
        return RedirectResponse(url="/login", status_code=302)

    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    default_profile = get_default_profile_name(db)

    try:
        safe_filename = sanitize_filename(file.filename or "")
    except ValueError as e:
        return render_upload_page(request, db, error=str(e), profile=default_profile)

    try:
        content = await file.read()
    except Exception:
        return render_upload_page(
            request,
            db,
            error="Не удалось прочитать загруженный файл.",
            profile=default_profile,
        )

    if not content:
        return render_upload_page(
            request,
            db,
            error="Загруженный файл пустой.",
            profile=default_profile,
        )

    file_path = make_unique_upload_path(safe_filename)

    try:
        file_path.write_bytes(content)
    except OSError:
        return render_upload_page(
            request,
            db,
            error="Не удалось сохранить файл на сервере.",
            profile=default_profile,
        )

    try:
        df = read_candidates_file(file_path)
    except Exception as e:
        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            pass

        return render_upload_page(
            request,
            db,
            error=f"Не удалось прочитать файл: {e}",
            profile=default_profile,
        )

    result = validate_candidates_df(df)
    if not result.ok:
        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            pass

        return render_upload_page(
            request,
            db,
            error="Файл содержит ошибки:",
            errors=result.errors,
            profile=default_profile,
        )

    df = result.df

    try:
        proba = ml_model.predict_proba_success(df)
        base_scores = [p * 100 for p in proba]
    except Exception:
        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            pass

        return render_upload_page(
            request,
            db,
            error="Не удалось выполнить ML-анализ файла.",
            profile=default_profile,
        )

    profile = default_profile
    weights = get_profile_weights(db, profile)

    try:
        scored = build_results(df, weights, alpha=DEFAULT_ALPHA, base_scores_from_ml=base_scores)
    except Exception:
        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            pass

        return render_upload_page(
            request,
            db,
            error="Не удалось рассчитать итоговые оценки кандидатов.",
            profile=default_profile,
        )

    try:
        upload_row = Upload(user_id=user.id, filename=safe_filename)
        db.add(upload_row)
        db.flush()

        weights_to_store = {k: float(v) for k, v in weights.items()}
        top5 = scored.sort_values("final_score", ascending=False).head(5)[
            ["candidate_id", "full_name", "vacancy", "final_score"]
        ].to_dict(orient="records")

        analysis_row = Analysis(
            upload_id=upload_row.id,
            profile=profile,
            weights_json=json.dumps(weights_to_store, ensure_ascii=False),
            alpha=DEFAULT_ALPHA,
            summary_json=json.dumps(top5, ensure_ascii=False),
        )
        db.add(analysis_row)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        return render_upload_page(
            request,
            db,
            error="Не удалось сохранить результаты анализа в базу данных.",
            profile=default_profile,
        )

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "page_title": "Результаты анализа кандидатов",
            "columns": list(scored.columns),
            "rows": scored.to_dict(orient="records"),
            "filename": safe_filename,
            "profile": profile,
            "weights": {k: round(v * 100, 1) for k, v in weights.items()},
            "profiles": get_profile_names(db),
            "alpha": DEFAULT_ALPHA,
            "top_candidate": top5[0]["full_name"] if top5 else "—",
            "candidates_count": len(scored),
            "avg_score": round(float(scored["final_score"].mean()), 2) if not scored.empty else 0,
            "column_labels": COLUMN_LABELS_RU,
            "criteria_labels": get_profile_labels(db, profile),
        },
    )


@app.post("/score", response_class=HTMLResponse)
def score_candidates(
    request: Request,
    filename: str = Form(...),
    profile: str = Form(""),
    w_c1: float = Form(15),
    w_c2: float = Form(35),
    w_c3: float = Form(25),
    w_c4: float = Form(15),
    w_c5: float = Form(5),
    w_c6: float = Form(5),
    db: Session = Depends(get_db),
):
    if not require_login(request):
        return RedirectResponse(url="/login", status_code=302)

    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    filename = (filename or "").strip()
    if not filename:
        return render_upload_page(
            request,
            db,
            error="Не указано имя файла для повторного анализа.",
        )

    available_profiles = get_profile_names(db)
    if profile not in available_profiles:
        profile = get_default_profile_name(db)

    current_weights = {
        "C1": float(w_c1),
        "C2": float(w_c2),
        "C3": float(w_c3),
        "C4": float(w_c4),
        "C5": float(w_c5),
        "C6": float(w_c6),
    }

    weight_errors = validate_weight_values(current_weights)
    if weight_errors:
        return render_upload_page(
            request,
            db,
            error="Проверьте веса критериев:",
            errors=weight_errors,
            profile=profile,
            weights=current_weights,
        )

    upload_row = (
        db.query(Upload)
        .filter(Upload.user_id == user.id, Upload.filename == filename)
        .order_by(Upload.uploaded_at.desc())
        .first()
    )

    file_path = None
    matching_files = sorted(UPLOAD_DIR.glob(f"{Path(filename).stem}_*{Path(filename).suffix}"), reverse=True)
    if matching_files:
        file_path = matching_files[0]

    if not upload_row or not file_path or not file_path.exists():
        return render_upload_page(
            request,
            db,
            error="Файл не найден. Загрузите его заново.",
            profile=profile,
            weights=current_weights,
        )

    try:
        df = read_candidates_file(file_path)
    except Exception:
        return render_upload_page(
            request,
            db,
            error="Не удалось прочитать файл. Загрузите его заново.",
            profile=profile,
            weights=current_weights,
        )

    result = validate_candidates_df(df)
    if not result.ok:
        return render_upload_page(
            request,
            db,
            error="Файл содержит ошибки:",
            errors=result.errors,
            profile=profile,
            weights=current_weights,
        )

    df = result.df

    try:
        proba = ml_model.predict_proba_success(df)
        base_scores = [p * 100 for p in proba]
    except Exception:
        return render_upload_page(
            request,
            db,
            error="Не удалось выполнить ML-анализ файла.",
            profile=profile,
            weights=current_weights,
        )

    try:
        weights = normalize_weights({
            "C1": max(0.0, w_c1 / 100),
            "C2": max(0.0, w_c2 / 100),
            "C3": max(0.0, w_c3 / 100),
            "C4": max(0.0, w_c4 / 100),
            "C5": max(0.0, w_c5 / 100),
            "C6": max(0.0, w_c6 / 100),
        })
    except Exception:
        return render_upload_page(
            request,
            db,
            error="Не удалось обработать веса критериев.",
            profile=profile,
            weights=current_weights,
        )

    try:
        scored = build_results(df, weights, alpha=DEFAULT_ALPHA, base_scores_from_ml=base_scores)
    except Exception:
        return render_upload_page(
            request,
            db,
            error="Не удалось рассчитать итоговые оценки кандидатов.",
            profile=profile,
            weights=current_weights,
        )

    weights_to_store = {k: float(v) for k, v in weights.items()}
    top5 = scored.sort_values("final_score", ascending=False).head(5)[
        ["candidate_id", "full_name", "vacancy", "final_score"]
    ].to_dict(orient="records")

    try:
        analysis_row = Analysis(
            upload_id=upload_row.id,
            profile=profile,
            weights_json=json.dumps(weights_to_store, ensure_ascii=False),
            alpha=DEFAULT_ALPHA,
            summary_json=json.dumps(top5, ensure_ascii=False),
        )
        db.add(analysis_row)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        return render_upload_page(
            request,
            db,
            error="Не удалось сохранить результаты повторного анализа.",
            profile=profile,
            weights=current_weights,
        )

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "page_title": "Результаты анализа кандидатов",
            "columns": list(scored.columns),
            "rows": scored.to_dict(orient="records"),
            "filename": filename,
            "profile": profile,
            "weights": {k: round(v * 100, 1) for k, v in weights.items()},
            "profiles": get_profile_names(db),
            "alpha": DEFAULT_ALPHA,
            "top_candidate": top5[0]["full_name"] if top5 else "—",
            "candidates_count": len(scored),
            "avg_score": round(float(scored["final_score"].mean()), 2) if not scored.empty else 0,
            "column_labels": COLUMN_LABELS_RU,
            "criteria_labels": get_profile_labels(db, profile),
        },
    )

@app.post("/export-pdf")
def export_pdf_report(
    request: Request,
    filename: str = Form(""),
    profile: str = Form(""),
    rows_json: str = Form("[]"),
    weights_json: str = Form("{}"),
    criteria_labels_json: str = Form("{}"),
    avg_score: float = Form(0),
    candidates_count: int = Form(0),
    top_candidate: str = Form("—"),
):
    print("EXPORT PDF ROUTE CALLED")
    if not require_login(request):
        return RedirectResponse(url="/login", status_code=302)

    try:
        rows = json.loads(rows_json or "[]")
        weights = json.loads(weights_json or "{}")
        criteria_labels = json.loads(criteria_labels_json or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return PlainTextResponse("Не удалось сформировать PDF: некорректные данные отчета.", status_code=400)

    if not isinstance(rows, list):
        return PlainTextResponse("Не удалось сформировать PDF: список кандидатов поврежден.", status_code=400)

    if not isinstance(weights, dict):
        weights = {}

    if not isinstance(criteria_labels, dict):
        criteria_labels = {}

    try:
        pdf_buffer = build_pdf_report(
            filename=filename,
            profile=profile,
            rows=rows,
            weights=weights,
            criteria_labels=criteria_labels,
            avg_score=avg_score,
            candidates_count=candidates_count,
            top_candidate=top_candidate,
        )
    except Exception as e:
        print("PDF ERROR:", repr(e))
        return PlainTextResponse(
            f"Не удалось сформировать PDF-отчет: {type(e).__name__}: {e}",
            status_code=500
        )

    safe_name = Path(filename).stem if filename else "analysis_report"

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}_report.pdf"'
        },
    )


@app.post("/login")
def login_action(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    username = (username or "").strip()
    password = password or ""

    if not username or not password:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Введите логин и пароль"},
        )

    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Неверный логин или пароль"},
        )

    resp = RedirectResponse(url="/", status_code=302)
    resp.set_cookie("username", user.username, httponly=True, samesite="lax")
    resp.set_cookie("role", user.role, httponly=True, samesite="lax")
    return resp


@app.get("/admin/users", response_class=HTMLResponse)
def admin_users_page(request: Request, db: Session = Depends(get_db)):
    if not require_login(request):
        return RedirectResponse(url="/login", status_code=302)
    if not require_admin(request):
        return PlainTextResponse("Доступ запрещен", status_code=403)

    users = db.query(User).order_by(User.id.asc()).all()
    return templates.TemplateResponse(
        "admin_users.html",
        {
            "request": request,
            "page_title": "Управление пользователями",
            "users": users,
        },
    )


@app.post("/admin/users/create", response_class=HTMLResponse)
def admin_create_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form("HR"),
    db: Session = Depends(get_db),
):
    if not require_login(request):
        return RedirectResponse(url="/login", status_code=302)
    if not require_admin(request):
        return PlainTextResponse("Доступ запрещен", status_code=403)

    username = (username or "").strip()
    password = password or ""
    role = role.upper()

    if not username:
        return render_admin_users_page(request, db, error="Логин не может быть пустым")

    if len(username) > 150:
        return render_admin_users_page(request, db, error="Логин слишком длинный")

    if len(password) < MIN_PASSWORD_LENGTH:
        return render_admin_users_page(
            request,
            db,
            error=f"Пароль должен содержать минимум {MIN_PASSWORD_LENGTH} символов",
        )

    if role not in ("HR", "ADMIN"):
        role = "HR"

    exists = db.query(User).filter(User.username == username).first()
    if exists:
        return render_admin_users_page(
            request,
            db,
            error="Пользователь с таким логином уже существует",
        )

    try:
        new_user = User(
            username=username,
            password_hash=hash_password(password),
            role=role,
        )
        db.add(new_user)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        return render_admin_users_page(
            request,
            db,
            error="Не удалось создать пользователя",
        )

    return render_admin_users_page(
        request,
        db,
        ok=f"Пользователь {username} создан",
    )


@app.get("/admin/profiles", response_class=HTMLResponse)
def admin_profiles_page(request: Request, db: Session = Depends(get_db)):
    if not require_login(request):
        return RedirectResponse(url="/login", status_code=302)
    if not require_admin(request):
        return PlainTextResponse("Доступ запрещен", status_code=403)

    seed_default_profiles(db)
    profiles = db.query(Profile).order_by(Profile.name.asc()).all()

    return templates.TemplateResponse(
        "admin_profiles.html",
        {
            "request": request,
            "page_title": "Профили и критерии",
            "profiles": profiles,
        },
    )


@app.post("/admin/profiles/create", response_class=HTMLResponse)
def admin_profiles_create(
    request: Request,
    name: str = Form(...),
    db: Session = Depends(get_db),
):
    if not require_login(request):
        return RedirectResponse(url="/login", status_code=302)
    if not require_admin(request):
        return PlainTextResponse("Доступ запрещен", status_code=403)

    seed_default_profiles(db)
    clean_name = (name or "").strip()

    if not clean_name:
        return render_admin_profiles_page(
            request,
            db,
            error="Название профиля не может быть пустым.",
        )

    if len(clean_name) > 100:
        return render_admin_profiles_page(
            request,
            db,
            error="Название профиля слишком длинное.",
        )

    exists = db.query(Profile).filter(Profile.name == clean_name).first()
    if exists:
        return render_admin_profiles_page(
            request,
            db,
            error="Профиль с таким названием уже существует.",
        )

    try:
        profile = Profile(name=clean_name)
        db.add(profile)
        db.flush()

        default_weights = normalize_weights({c: 1.0 for c in CRITERIA})
        for idx, code in enumerate(CRITERIA, start=1):
            db.add(ProfileCriterion(
                profile_id=profile.id,
                code=code,
                label=DEFAULT_CRITERIA_LABELS[code],
                weight=default_weights[code],
                sort_order=idx,
                is_active=True,
            ))

        db.commit()
    except SQLAlchemyError:
        db.rollback()
        return render_admin_profiles_page(
            request,
            db,
            error="Не удалось создать профиль.",
        )

    return RedirectResponse(url="/admin/profiles", status_code=302)


@app.post("/admin/profiles/update/{profile_id}", response_class=HTMLResponse)
def admin_profiles_update(
    profile_id: int,
    request: Request,
    name: str = Form(...),
    label_c1: str = Form(...),
    label_c2: str = Form(...),
    label_c3: str = Form(...),
    label_c4: str = Form(...),
    label_c5: str = Form(...),
    label_c6: str = Form(...),
    w_c1: float = Form(...),
    w_c2: float = Form(...),
    w_c3: float = Form(...),
    w_c4: float = Form(...),
    w_c5: float = Form(...),
    w_c6: float = Form(...),
    db: Session = Depends(get_db),
):
    if not require_login(request):
        return RedirectResponse(url="/login", status_code=302)
    if not require_admin(request):
        return PlainTextResponse("Доступ запрещен", status_code=403)

    profile = db.query(Profile).filter(Profile.id == profile_id).first()
    if not profile:
        return render_admin_profiles_page(
            request,
            db,
            error="Профиль не найден.",
        )

    clean_name = (name or "").strip()
    if not clean_name:
        return render_admin_profiles_page(
            request,
            db,
            error="Название профиля не может быть пустым.",
        )

    duplicate = db.query(Profile).filter(Profile.name == clean_name, Profile.id != profile_id).first()
    if duplicate:
        return render_admin_profiles_page(
            request,
            db,
            error="Профиль с таким названием уже существует.",
        )

    raw_weights = {
        "C1": max(0.0, float(w_c1)),
        "C2": max(0.0, float(w_c2)),
        "C3": max(0.0, float(w_c3)),
        "C4": max(0.0, float(w_c4)),
        "C5": max(0.0, float(w_c5)),
        "C6": max(0.0, float(w_c6)),
    }

    weight_errors = validate_weight_values(raw_weights)
    if weight_errors:
        return render_admin_profiles_page(
            request,
            db,
            error="Проверьте веса профиля: " + " ".join(weight_errors),
        )

    normalized = normalize_weights(raw_weights)

    labels = {
        "C1": (label_c1 or "").strip() or DEFAULT_CRITERIA_LABELS["C1"],
        "C2": (label_c2 or "").strip() or DEFAULT_CRITERIA_LABELS["C2"],
        "C3": (label_c3 or "").strip() or DEFAULT_CRITERIA_LABELS["C3"],
        "C4": (label_c4 or "").strip() or DEFAULT_CRITERIA_LABELS["C4"],
        "C5": (label_c5 or "").strip() or DEFAULT_CRITERIA_LABELS["C5"],
        "C6": (label_c6 or "").strip() or DEFAULT_CRITERIA_LABELS["C6"],
    }

    try:
        profile.name = clean_name

        criteria_by_code = {c.code: c for c in profile.criteria}
        for idx, code in enumerate(CRITERIA, start=1):
            criterion = criteria_by_code.get(code)
            if not criterion:
                criterion = ProfileCriterion(profile_id=profile.id, code=code)
                db.add(criterion)

            criterion.label = labels[code]
            criterion.weight = normalized[code]
            criterion.sort_order = idx
            criterion.is_active = normalized[code] > 0

        db.commit()
    except SQLAlchemyError:
        db.rollback()
        return render_admin_profiles_page(
            request,
            db,
            error="Не удалось обновить профиль.",
        )

    return RedirectResponse(url="/admin/profiles", status_code=302)


@app.post("/admin/profiles/delete/{profile_id}", response_class=HTMLResponse)
def admin_profiles_delete(
    profile_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    if not require_login(request):
        return RedirectResponse(url="/login", status_code=302)
    if not require_admin(request):
        return PlainTextResponse("Доступ запрещен", status_code=403)

    profiles_count = db.query(Profile).count()
    if profiles_count <= 1:
        return render_admin_profiles_page(
            request,
            db,
            error="Нельзя удалить последний профиль.",
        )

    profile = db.query(Profile).filter(Profile.id == profile_id).first()
    if not profile:
        return render_admin_profiles_page(
            request,
            db,
            error="Профиль не найден.",
        )

    try:
        db.delete(profile)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        return render_admin_profiles_page(
            request,
            db,
            error="Не удалось удалить профиль.",
        )

    return RedirectResponse(url="/admin/profiles", status_code=302)


@app.get("/history", response_class=HTMLResponse)
def history_page(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    items = (
        db.query(Analysis, Upload)
        .join(Upload, Analysis.upload_id == Upload.id)
        .filter(Upload.user_id == user.id)
        .order_by(Analysis.created_at.desc())
        .all()
    )

    rows = []
    for analysis, upload in items:
        rows.append(
            {
                "id": analysis.id,
                "created_at": analysis.created_at.strftime("%Y-%m-%d %H:%M"),
                "filename": upload.filename,
                "profile": analysis.profile,
                "alpha": analysis.alpha,
                "weights": safe_json_loads(analysis.weights_json, {}),
                "top5": safe_json_loads(analysis.summary_json, []),
            }
        )

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "page_title": "История анализов",
            "rows": rows,
        },
    )


@app.post("/history/delete/{analysis_id}")
def delete_history_item(
    analysis_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    analysis = (
        db.query(Analysis)
        .join(Upload, Analysis.upload_id == Upload.id)
        .filter(Analysis.id == analysis_id, Upload.user_id == user.id)
        .first()
    )

    if not analysis:
        return RedirectResponse(url="/history", status_code=302)

    try:
        db.delete(analysis)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        return PlainTextResponse("Не удалось удалить запись из истории", status_code=500)

    return RedirectResponse(url="/history", status_code=302)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return PlainTextResponse(
        "Внутренняя ошибка сервера. Попробуйте еще раз позже.",
        status_code=500,
    )