from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from model.model import Base, Subject, Activity

DATABASE_URI = 'your_database_connection_uri'
engine = create_engine(DATABASE_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def populate_initial_data():
    db = SessionLocal()

    for subject_id in range(1, 11):
        if not db.query(Subject).filter(Subject.id == subject_id).first():
            db.add(Subject(id=subject_id))

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

    for activity_id, description in activities:
        if not db.query(Activity).filter(Activity.id == activity_id).first():
            db.add(Activity(id=activity_id, description=description))

    db.commit()
    db.close()


populate_initial_data()
