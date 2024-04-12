from sqlalchemy import func

from database.model.sensor_record import SensorRecord
from database.model.session import Session
from database.tools import my_conn
from database.tools.my_conn import get_db_session


def get_session_label_ids():
    with my_conn.get_db_session() as db:
        sessions = db.query(Session.id, Session.activity_id).all()
        session_ids, activity_ids = zip(*sessions)
        return session_ids, activity_ids


def fetch_all_session_lengths(session_ids):
    lengths = {}
    with get_db_session() as db:
        results = (
            db.query(SensorRecord.session_id, func.count().label("total"))
            .filter(SensorRecord.session_id.in_(session_ids))
            .group_by(SensorRecord.session_id)
            .all()
        )

        for session_id, total in results:
            lengths[session_id] = total
    return lengths
