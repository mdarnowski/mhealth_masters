import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()
database_uri = os.getenv("DATABASE_URI")
if database_uri is None:
    raise ValueError("DATABASE_URI not found in environment variables")

db_engine = create_engine(database_uri)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)


@contextmanager
def get_db_session():
    """Provides a transactional scope around a series of operations."""
    db_session = SessionLocal()
    try:
        yield db_session
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def load_db():
    return SessionLocal()
