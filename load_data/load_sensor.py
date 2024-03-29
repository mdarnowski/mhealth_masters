from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from tqdm import tqdm

from load_data.db_model.activity import Activity
from load_data.db_model.base import Base
from load_data.db_model.sensor_data import create_sensor_data
from load_data.db_model.session import Session
from load_data.db_model.subject import Subject

import os


def reload_db(engine):
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def define_activities(session):
    subjects = [Subject(id=subject_id) for subject_id in range(1, 11)]
    session.add_all(subjects)

    activities = [
        (1, "Standing still"),
        (2, "Sitting and relaxing"),
        (3, "Lying down"),
        (4, "Walking"),
        (5, "Climbing stairs"),
        (6, "Waist bends forward"),
        (7, "Frontal elevation of arms"),
        (8, "Knees bending"),
        (9, "Cycling"),
        (10, "Jogging"),
        (11, "Running"),
        (12, "Jump front & back"),
    ]
    session.add_all([Activity(id=act_id, description=desc) for act_id, desc in activities])

    session.commit()


def import_data_from_log(db, subject_id, log_file_path):
    last_label = None
    session_obj = None
    sequence = 1

    with open(log_file_path, "r") as file:
        for line in file:
            values = line.strip().split("\t")
            label = int(values[-1])

            if label == 0:
                continue

            if label != last_label:
                session_obj = Session(subject_id=subject_id, activity_id=label)
                db.add(session_obj)
                db.commit()
                last_label = label
                sequence = 1
            else:
                sequence += 1

            sensor_data_values = [float(value) for value in values[:-1]]
            sensor_data = create_sensor_data(sequence, session_obj.id, *sensor_data_values)
            db.add(sensor_data)

    db.commit()


if __name__ == "__main__":
    load_dotenv()
    DATABASE_URI = os.getenv("DATABASE_URI")
    db_engine = create_engine(DATABASE_URI)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    database = SessionLocal()

    reload_db(db_engine)
    define_activities(database)
    for i in tqdm(range(1, 11), desc="Processing log files"):
        import_data_from_log(
            database, subject_id=i, log_file_path=f"../data/log/mHealth_subject{i}.log"
        )

    database.close()
