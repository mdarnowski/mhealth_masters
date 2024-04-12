from database.model.activity import Activity
from database.model.session import Session


def fetch_activity_label(db, session_id):
    return db.query(Session.activity_id).filter(Session.id == session_id).scalar()


def count_activity(db):
    return db.query(Activity).count()
