from app.db import engine, Base
from app import models  # важно: чтобы User зарегистрировался в Base.metadata


def init_db():
    Base.metadata.create_all(bind=engine)