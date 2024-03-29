from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Subject(Base):
    __tablename__ = 'subject'
    id = Column(Integer, primary_key=True)


class Activity(Base):
    __tablename__ = 'activity'
    id = Column(Integer, primary_key=True)
    description = Column(String)


class Session(Base):
    __tablename__ = 'session'
    id = Column(Integer, primary_key=True)
    subject_id = Column(Integer, ForeignKey('subjects.id'))
    activity_id = Column(Integer, ForeignKey('activities.id'))
    subject = relationship("Subject", back_populates="sessions")
    activity = relationship("Activity", back_populates="sessions")


class SensorData(Base):
    __tablename__ = 'sensor_data'
    sequence = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id'), primary_key=True)
    session = relationship("Session", back_populates="sensor_data")

    # Sensor measurement columns
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
