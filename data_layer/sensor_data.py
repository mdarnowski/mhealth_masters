import numpy as np
import pandas as pd

from database.model.activity import Activity
from database.model.sensor_record import SensorRecord
from database.model.session import Session
from database.tools.my_conn import get_db_session


def load_sensor_ete_df():
    with get_db_session() as db:
        query = (
            db.query(
                SensorRecord,
                Session.subject_id,
                Session.activity_id,
                Activity.description,
            )
            .join(Session, SensorRecord.session_id == Session.id)
            .join(Activity)
        )

        df_sensor_data = pd.read_sql(query.statement, db.bind)
        df_sensor_data.drop("sequence", axis=1, inplace=True)

        return df_sensor_data


def get_sensor_columns(incl_acc=True, incl_gyro=True, incl_mag=True, incl_ecg=True):
    columns = []

    if incl_acc:
        acceleration_columns = [
            SensorRecord.acceleration_chest_x,
            SensorRecord.acceleration_chest_y,
            SensorRecord.acceleration_chest_z,
            SensorRecord.acceleration_left_ankle_x,
            SensorRecord.acceleration_left_ankle_y,
            SensorRecord.acceleration_left_ankle_z,
            SensorRecord.acceleration_right_lower_arm_x,
            SensorRecord.acceleration_right_lower_arm_y,
            SensorRecord.acceleration_right_lower_arm_z,
        ]
        columns.extend(acceleration_columns)

    if incl_gyro:
        gyro_columns = [
            SensorRecord.gyro_left_ankle_x,
            SensorRecord.gyro_left_ankle_y,
            SensorRecord.gyro_left_ankle_z,
            SensorRecord.gyro_right_lower_arm_x,
            SensorRecord.gyro_right_lower_arm_y,
            SensorRecord.gyro_right_lower_arm_z,
        ]
        columns.extend(gyro_columns)

    if incl_mag:
        magnetometer_columns = [
            SensorRecord.magnetometer_left_ankle_x,
            SensorRecord.magnetometer_left_ankle_y,
            SensorRecord.magnetometer_left_ankle_z,
            SensorRecord.magnetometer_right_lower_arm_x,
            SensorRecord.magnetometer_right_lower_arm_y,
            SensorRecord.magnetometer_right_lower_arm_z,
        ]
        columns.extend(magnetometer_columns)

    if incl_ecg:
        ecg_columns = [
            SensorRecord.ecg_lead_1,
            SensorRecord.ecg_lead_2,
        ]
        columns.extend(ecg_columns)

    if not columns:
        raise ValueError("At least one type of sensor data must be included.")

    return columns


def create_sensor_record(sequence, session_id, *data):
    sensor_record = SensorRecord(
        sequence=sequence,
        session_id=session_id,
        acceleration_chest_x=data[0],
        acceleration_chest_y=data[1],
        acceleration_chest_z=data[2],
        ecg_lead_1=data[3],
        ecg_lead_2=data[4],
        acceleration_left_ankle_x=data[5],
        acceleration_left_ankle_y=data[6],
        acceleration_left_ankle_z=data[7],
        gyro_left_ankle_x=data[8],
        gyro_left_ankle_y=data[9],
        gyro_left_ankle_z=data[10],
        magnetometer_left_ankle_x=data[11],
        magnetometer_left_ankle_y=data[12],
        magnetometer_left_ankle_z=data[13],
        acceleration_right_lower_arm_x=data[14],
        acceleration_right_lower_arm_y=data[15],
        acceleration_right_lower_arm_z=data[16],
        gyro_right_lower_arm_x=data[17],
        gyro_right_lower_arm_y=data[18],
        gyro_right_lower_arm_z=data[19],
        magnetometer_right_lower_arm_x=data[20],
        magnetometer_right_lower_arm_y=data[21],
        magnetometer_right_lower_arm_z=data[22],
    )
    return sensor_record


def fetch_sensor_data(
    db, session_id, incl_acc=True, incl_gyro=True, incl_mag=True, incl_ecg=True
):
    query = (
        db.query(
            *get_sensor_columns(
                incl_acc=incl_acc,
                incl_gyro=incl_gyro,
                incl_mag=incl_mag,
                incl_ecg=incl_ecg,
            )
        )
        .filter(SensorRecord.session_id == session_id)
        .order_by(SensorRecord.sequence)
    )
    return np.array(query.all())
