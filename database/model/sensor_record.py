from sqlalchemy import Column, Float, ForeignKey, Integer
from sqlalchemy.orm import relationship

from database.model.base import Base
from database.model.session import Session


class SensorRecord(Base):
    __tablename__ = "sensor_record"
    sequence = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("session.id"), primary_key=True)
    session = relationship(Session.__name__)

    # sensor measurement columns
    acceleration_chest_x = Column(Float)
    acceleration_chest_y = Column(Float)
    acceleration_chest_z = Column(Float)
    ecg_lead_1 = Column(Float)
    ecg_lead_2 = Column(Float)
    acceleration_left_ankle_x = Column(Float)
    acceleration_left_ankle_y = Column(Float)
    acceleration_left_ankle_z = Column(Float)
    gyro_left_ankle_x = Column(Float)
    gyro_left_ankle_y = Column(Float)
    gyro_left_ankle_z = Column(Float)
    magnetometer_left_ankle_x = Column(Float)
    magnetometer_left_ankle_y = Column(Float)
    magnetometer_left_ankle_z = Column(Float)
    acceleration_right_lower_arm_x = Column(Float)
    acceleration_right_lower_arm_y = Column(Float)
    acceleration_right_lower_arm_z = Column(Float)
    gyro_right_lower_arm_x = Column(Float)
    gyro_right_lower_arm_y = Column(Float)
    gyro_right_lower_arm_z = Column(Float)
    magnetometer_right_lower_arm_x = Column(Float)
    magnetometer_right_lower_arm_y = Column(Float)
    magnetometer_right_lower_arm_z = Column(Float)
