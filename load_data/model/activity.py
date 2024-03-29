from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

from load_data.model.base import Base


class Activity(Base):
    __tablename__ = "activity"
    id = Column(Integer, primary_key=True)
    description = Column(String)
    sessions = relationship("Session", back_populates="activity")
