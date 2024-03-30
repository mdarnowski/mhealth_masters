import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import ArgumentError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from load_data.db_model.activity import Activity
from load_data.db_model.sensor_data import SensorData
from load_data.db_model.session import Session


def load_engine():
    try:
        load_dotenv()
        database_uri = os.getenv("DATABASE_URI")
        if database_uri is None:
            raise ValueError("DATABASE_URI not found in environment variables")

        db_engine = create_engine(database_uri)
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
        database = session_local()
    except (ArgumentError, OperationalError, SQLAlchemyError) as e:
        raise RuntimeError(
            "Database connection error. Please check your database_uri."
        ) from e
    except (EnvironmentError, KeyError, TypeError, ValueError) as e:
        raise RuntimeError("Environment or configuration error encountered.") from e
    except Exception as e:
        raise RuntimeError("An unexpected error occurred.") from e

    return db_engine, database


def load_db():
    db_engine, database = load_engine()
    return database


def load_sensor_ete_df():
    # Create a session
    db = load_db()

    query = (
        db.query(
            SensorData, Session.subject_id, Session.activity_id, Activity.description
        )
        .join(Session, SensorData.session_id == Session.id)
        .join(Activity)
    )

    df_sensor_data = pd.read_sql(query.statement, db.bind)

    # Drop the 'sequence' column from the DataFrame
    if "sequence" in df_sensor_data:
        df_sensor_data.drop("sequence", axis=1, inplace=True)

    # Close the session
    db.close()

    # Return the DataFrame
    return df_sensor_data
