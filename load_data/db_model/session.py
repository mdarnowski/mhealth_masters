from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship

from load_data.db_model.base import Base


class Session(Base):
    __tablename__ = "session"
    id = Column(Integer, primary_key=True)
    subject_id = Column(Integer, ForeignKey("subject.id"))
    activity_id = Column(Integer, ForeignKey("activity.id"))
    subject = relationship("Subject", back_populates="sessions")
    activity = relationship("Activity", back_populates="sessions")
    sensor_data = relationship("SensorData", back_populates="session")
