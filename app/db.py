from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from pathlib import Path
import logging

# Логирование (минимальное)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent  # app/
DB_PATH = BASE_DIR / "data" / "app.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"

try:
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
except Exception as e:
    logger.exception("Ошибка при создании подключения к БД")
    raise RuntimeError("Не удалось инициализировать базу данных") from e


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()  # гарантируем commit если всё ок
    except SQLAlchemyError as e:
        db.rollback()
        logger.exception("Ошибка работы с БД")
        raise
    except Exception as e:
        db.rollback()
        logger.exception("Неизвестная ошибка в сессии БД")
        raise
    finally:
        db.close()