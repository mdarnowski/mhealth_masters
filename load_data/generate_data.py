import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from load_data import my_conn
from load_data.db_model.sensor_data import SensorData
from load_data.db_model.session import Session
from load_data.my_conn import get_db_session


def fetch_activity_label(session_id):
    with get_db_session() as db:
        return db.query(Session.activity_id).filter(Session.id == session_id).scalar()


def fetch_sensor_data(session_id, offset=0, limit=None):
    with get_db_session() as db:
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

        if limit is not None:
            query = query.offset(offset).limit(limit)

        results = query.all()

        return np.array(results) if results else None


def get_tf_data_for_session(
    session_id, segment_size, segment_index=None, just_count=False
):
    if just_count:
        total_rows = len(fetch_sensor_data(session_id))
        return total_rows // segment_size if total_rows is not None else 0

    if segment_index is not None:
        offset = segment_index * segment_size
        results_array = fetch_sensor_data(session_id, offset, segment_size)
        if results_array is not None:
            label = fetch_activity_label(session_id)
            return results_array, label
        return np.array([]), None

    results_array = fetch_sensor_data(session_id)
    if results_array is None:
        return np.array([]), None

    label = fetch_activity_label(session_id)
    num_complete_segments = len(results_array) // segment_size
    features_segmented = results_array[: num_complete_segments * segment_size].reshape(
        -1, segment_size, results_array.shape[1]
    )
    labels_segmented = np.full(num_complete_segments, label)

    return features_segmented, labels_segmented


def data_generator(session_ids, segment_size):
    for session_id in session_ids:
        total_segments = get_tf_data_for_session(
            session_id, segment_size, just_count=True
        )
        for segment_index in range(total_segments):
            features, label = get_tf_data_for_session(
                session_id, segment_size, segment_index
            )
            label -= 1
            yield features, label


def fetch_all_sessions_with_subjects():
    with my_conn.get_db_session() as db:
        sessions = db.query(Session).all()
        return [
            {"id": session.id, "subject_id": session.subject_id} for session in sessions
        ]


def prepare_datasets(
    segment_size=200, batch_size=32, n_features=23, test_size=0.2, val_size=0.25
):
    all_sessions = fetch_all_sessions_with_subjects()

    subject_ids = set(session["subject_id"] for session in all_sessions)
    train_val_subjects, test_subjects = train_test_split(
        list(subject_ids), test_size=test_size, random_state=42
    )
    train_subjects, val_subjects = train_test_split(
        train_val_subjects, test_size=val_size, random_state=42
    )

    train_sessions = [
        session["id"]
        for session in all_sessions
        if session["subject_id"] in train_subjects
    ]
    val_sessions = [
        session["id"]
        for session in all_sessions
        if session["subject_id"] in val_subjects
    ]
    test_sessions = [
        session["id"]
        for session in all_sessions
        if session["subject_id"] in test_subjects
    ]

    def dataset_generator_wrapper(session_ids):
        def generator():
            for features, label in data_generator(session_ids, segment_size):
                yield features, label

        return generator

    def create_dataset(session_ids):
        return (
            tf.data.Dataset.from_generator(
                dataset_generator_wrapper(session_ids),
                output_signature=(
                    tf.TensorSpec(shape=(segment_size, n_features), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                ),
            )
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    return (
        create_dataset(train_sessions),
        create_dataset(val_sessions),
        create_dataset(test_sessions),
    )
