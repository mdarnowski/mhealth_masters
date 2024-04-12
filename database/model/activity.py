from sqlalchemy import Column, Integer, String

from database.model.base import Base


class Activity(Base):
    __tablename__ = "activity"
    id = Column(Integer, primary_key=True)
    description = Column(String)
