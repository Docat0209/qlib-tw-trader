from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker, DeclarativeBase

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DATABASE_URL = f"sqlite:///{DATA_DIR / 'data.db'}"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


def get_session() -> Session:
    """取得資料庫 Session"""
    return SessionLocal()


def init_db():
    """初始化資料庫，建立所有表"""
    # 導入 models 確保所有表都被註冊
    from src.repositories import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
