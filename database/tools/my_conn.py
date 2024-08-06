import os
from contextlib import contextmanager

import numpy as np
import psycopg2
from dotenv import load_dotenv
from psycopg2.extensions import AsIs, register_adapter
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def adapt_numpy_int32(numpy_int32):
    return AsIs(int(numpy_int32))


register_adapter(np.int32, adapt_numpy_int32)

load_dotenv()
database_uri = os.getenv("DATABASE_URI")
if database_uri is None:
    raise ValueError("DATABASE_URI not found in environment variables")

db_engine = create_engine(database_uri, pool_size=-1, max_overflow=-1)
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
