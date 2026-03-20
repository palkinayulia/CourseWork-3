from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.models import User
from app.auth import hash_password


def seed_admin():
    db: Session = SessionLocal()
    try:
        exists = db.query(User).filter(User.username == "admin").first()
        if exists:
            print("Admin already exists")
            return

        admin = User(
            username="admin",
            password_hash=hash_password("admin123"),
            role="ADMIN"
        )
        db.add(admin)
        db.commit()
        print("Created admin: admin / admin123")
    finally:
        db.close()


if __name__ == "__main__":
    seed_admin()