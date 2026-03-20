from passlib.context import CryptContext


# PBKDF2 работает стабильно на Windows, без bcrypt-зависимостей
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
def hash_password(password: str) -> str:
    if password is None:
        raise ValueError("Пароль не может быть None")

    password = str(password)
    if not password.strip():
        raise ValueError("Пароль не может быть пустым")

    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    if password is None or password_hash is None:
        return False

    password = str(password)
    password_hash = str(password_hash)

    if not password or not password_hash:
        return False

    try:
        return pwd_context.verify(password, password_hash)
    except Exception:
        return False