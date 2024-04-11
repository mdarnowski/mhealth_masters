from sqlalchemy import Column, Integer, String

from db_model.base import Base


class Activity(Base):
    __tablename__ = "activity"
    id = Column(Integer, primary_key=True)
    description = Column(String)
