from sqlalchemy import Column, Integer
from sqlalchemy.orm import relationship

from load_data.db_model.base import Base


class Subject(Base):
    __tablename__ = "subject"
    id = Column(Integer, primary_key=True)
