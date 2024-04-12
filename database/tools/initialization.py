from database.model.activity import Activity
from database.model.base import Base


def reload_db(engine):
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def define_activities(db):
    activities = [
        (0, "Standing still"),
        (1, "Sitting and relaxing"),
        (2, "Lying down"),
        (3, "Walking"),
        (4, "Climbing stairs"),
        (5, "Waist bends forward"),
        (6, "Frontal elevation of arms"),
        (7, "Knees bending"),
        (8, "Cycling"),
        (9, "Jogging"),
        (10, "Running"),
        (11, "Jump front & back"),
    ]
    db.add_all([Activity(id=act_id, description=desc) for act_id, desc in activities])

    db.commit()
