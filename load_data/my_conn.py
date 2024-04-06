import os
from contextlib import contextmanager

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from load_data.db_model.activity import Activity
from load_data.db_model.sensor_data import SensorData
from load_data.db_model.session import Session

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


def load_sensor_ete_df():
    with get_db_session() as db:
        query = (
            db.query(
                SensorData,
                Session.subject_id,
                Session.activity_id,
                Activity.description,
            )
            .join(Session, SensorData.session_id == Session.id)
            .join(Activity)
        )

        df_sensor_data = pd.read_sql(query.statement, db.bind)
        df_sensor_data.drop("sequence", axis=1, inplace=True)

        db.close()
        return df_sensor_data
