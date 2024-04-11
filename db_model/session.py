from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship

from db_model.activity import Activity
from db_model.base import Base


class Session(Base):
    __tablename__ = "session"
    id = Column(Integer, primary_key=True)
    activity_id = Column(Integer, ForeignKey("activity.id"))
    subject_id = Column(Integer)
    activity = relationship(Activity.__name__)
