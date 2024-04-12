from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship

from database.model.activity import Activity
from database.model.base import Base


class Session(Base):
    __tablename__ = "session"
    id = Column(Integer, primary_key=True)
    activity_id = Column(Integer, ForeignKey("activity.id"))
    subject_id = Column(Integer)
    activity = relationship(Activity.__name__)
