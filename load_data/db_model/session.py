from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship

from load_data.db_model.activity import Activity
from load_data.db_model.base import Base
from load_data.db_model.subject import Subject


class Session(Base):
    __tablename__ = "session"
    id = Column(Integer, primary_key=True)
    subject_id = Column(Integer, ForeignKey("subject.id"))
    activity_id = Column(Integer, ForeignKey("activity.id"))
    subject = relationship(Subject.__name__)
    activity = relationship(Activity.__name__)
