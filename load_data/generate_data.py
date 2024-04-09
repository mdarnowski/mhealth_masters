import math

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sqlalchemy import func

from load_data import my_conn
from load_data.db_model.sensor_data import SensorData
from load_data.db_model.session import Session
from load_data.my_conn import get_db_session


def get_all_session_ids_with_labels():
    with my_conn.get_db_session() as db:
        sessions = db.query(Session.id, Session.activity_id).all()
        return sessions


def fetch_activity_label(db, session_id):
    return db.query(Session.activity_id).filter(Session.id == session_id).scalar()


def fetch_all_session_lengths(session_ids):
    lengths = {}
    with get_db_session() as db:
        results = (
            db.query(SensorData.session_id, func.count().label("total"))
            .filter(SensorData.session_id.in_(session_ids))
            .group_by(SensorData.session_id)
            .all()
        )

        for session_id, total in results:
            lengths[session_id] = total
    return lengths


def calc_steps_per_epoch_efficiently(session_ids, segment_size, batch_size, n_shifts):
    session_lengths = fetch_all_session_lengths(session_ids)
    shift_steps = calc_shift_steps(segment_size, n_shifts)

    total_usable_segments = 0
    for session_id, length in session_lengths.items():
        for shift in shift_steps:
            adjusted_length = length - shift
            if adjusted_length >= segment_size:
                total_usable_segments += (
                    adjusted_length - segment_size
                ) // segment_size + 1

    steps_per_epoch = math.floor(total_usable_segments / batch_size)
    return steps_per_epoch


def get_sensor_data_query(db, session_id):
    query = (
        db.query(
            SensorData.acceleration_chest_x,
            SensorData.acceleration_chest_y,
            SensorData.acceleration_chest_z,
            SensorData.ecg_lead_1,
            SensorData.ecg_lead_2,
            SensorData.acceleration_left_ankle_x,
            SensorData.acceleration_left_ankle_y,
            SensorData.acceleration_left_ankle_z,
            SensorData.gyro_left_ankle_x,
            SensorData.gyro_left_ankle_y,
            SensorData.gyro_left_ankle_z,
            SensorData.magnetometer_left_ankle_x,
            SensorData.magnetometer_left_ankle_y,
            SensorData.magnetometer_left_ankle_z,
            SensorData.acceleration_right_lower_arm_x,
            SensorData.acceleration_right_lower_arm_y,
            SensorData.acceleration_right_lower_arm_z,
            SensorData.gyro_right_lower_arm_x,
            SensorData.gyro_right_lower_arm_y,
            SensorData.gyro_right_lower_arm_z,
            SensorData.magnetometer_right_lower_arm_x,
            SensorData.magnetometer_right_lower_arm_y,
            SensorData.magnetometer_right_lower_arm_z,
        )
        .filter(SensorData.session_id == session_id)
        .order_by(SensorData.sequence)
    )
    return query


def fetch_full_session_data(db, session_id):
    results = get_sensor_data_query(db, session_id).all()
    return np.array(results)


def calc_shift_steps(segment_size, n_shifts):
    if n_shifts > 1:
        shift_interval = segment_size // n_shifts
        shift_steps = [i * shift_interval for i in range(n_shifts)]
    else:
        shift_steps = [0]

    return shift_steps


def create_dataset(
    ids: list[int],
    segment_size: int,
    batch_size: int,
    n_features: int,
    n_shifts: int = 1,
):
    shift_steps = calc_shift_steps(segment_size, n_shifts)

    def data_generator():
        with my_conn.get_db_session() as db:
            for shift in shift_steps:
                for session_id in ids:
                    full_session_data = fetch_full_session_data(db, session_id)
                    label = fetch_activity_label(db, session_id) - 1
                    for offset in range(shift, len(full_session_data), segment_size):
                        end = offset + segment_size
                        if end <= len(full_session_data):
                            segment = full_session_data[offset:end]
                            yield segment, label

    return (
        tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=(segment_size, n_features), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )


def prepare_train_val_test(
    segment_size: int,
    batch_size: int,
    n_features: int,
    val_size: float = 0.2,
    test_size: float = 0.2,
    n_shifts: int = 1,
):
    sessions_with_labels = get_all_session_ids_with_labels()
    session_ids, activity_ids = zip(*sessions_with_labels)
    train_val_ids, test_ids, train_val_labels, _ = train_test_split(
        session_ids,
        activity_ids,
        test_size=test_size,
        random_state=42,
        stratify=activity_ids,
    )
    train_ids, val_ids, _, _ = train_test_split(
        train_val_ids,
        train_val_labels,
        test_size=val_size / (1 - test_size),
        random_state=42,
        stratify=train_val_labels,
    )

    train_steps_nr = calc_steps_per_epoch_efficiently(
        train_ids, segment_size, batch_size, n_shifts=n_shifts
    )
    val_steps_nr = calc_steps_per_epoch_efficiently(
        val_ids, segment_size, batch_size, n_shifts=n_shifts
    )
    test_steps_nr = calc_steps_per_epoch_efficiently(
        test_ids, segment_size, batch_size, n_shifts=n_shifts
    )

    train_dataset = create_dataset(
        list(train_ids), segment_size, batch_size, n_features, n_shifts
    ).repeat()
    val_dataset = create_dataset(
        list(val_ids), segment_size, batch_size, n_features, n_shifts
    ).repeat()
    test_dataset = create_dataset(
        list(test_ids), segment_size, batch_size, n_features, n_shifts
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_steps_nr,
        val_steps_nr,
        test_steps_nr,
    )
