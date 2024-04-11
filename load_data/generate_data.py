import math
import random

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sqlalchemy import func

from db_model.sensor_data import SensorRecord, get_sensor_columns
from db_model.session import Session
from load_data import my_conn
from load_data.my_conn import get_db_session


def get_session_ids_with_labels():
    with my_conn.get_db_session() as db:
        sessions = db.query(Session.id, Session.activity_id).all()
        return sessions


def fetch_activity_label(db, session_id):
    return db.query(Session.activity_id).filter(Session.id == session_id).scalar() - 1


def fetch_all_session_lengths(session_ids):
    lengths = {}
    with get_db_session() as db:
        results = (
            db.query(SensorRecord.session_id, func.count().label("total"))
            .filter(SensorRecord.session_id.in_(session_ids))
            .group_by(SensorRecord.session_id)
            .all()
        )

        for session_id, total in results:
            lengths[session_id] = total
    return lengths


def calc_steps_per_epoch(session_ids, segment_size, batch_size, n_shifts):
    session_lengths = fetch_all_session_lengths(session_ids)
    shift_steps = calc_shift_steps(segment_size, n_shifts)

    total_segments = 0
    for session_id, length in session_lengths.items():
        for shift in shift_steps:
            fin_len = length - shift
            if fin_len >= segment_size:
                total_segments += (fin_len - segment_size) // segment_size

    steps_per_epoch = math.ceil(total_segments / batch_size)
    return steps_per_epoch


def count_features(incl_acc=True, incl_gyro=True, incl_mag=True, incl_ecg=True):
    columns = get_sensor_columns(
        incl_acc=incl_acc, incl_gyro=incl_gyro, incl_mag=incl_mag, incl_ecg=incl_ecg
    )
    return len(columns)


def fetch_session_data(
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


def calc_shift_steps(segment_size, n_shifts):
    return [i * segment_size // n_shifts for i in range(n_shifts)]


def create_dataset(
    ids: list[int],
    segment_size: int,
    batch_size: int,
    n_features: int,
    n_shifts: int = 1,
    incl_acc: bool = True,
    incl_gyro: bool = True,
    incl_mag: bool = True,
    incl_ecg: bool = True,
):
    shift_steps = calc_shift_steps(segment_size, n_shifts)

    def data_generator():
        random.shuffle(ids)
        with my_conn.get_db_session() as db:

            for shift in shift_steps:
                for session_id in ids:
                    session_data = fetch_session_data(
                        db, session_id, incl_acc, incl_gyro, incl_mag, incl_ecg
                    )
                    label = fetch_activity_label(db, session_id)
                    for offset in range(shift, len(session_data), segment_size):
                        end = offset + segment_size
                        if end <= len(session_data):
                            segment = session_data[offset:end]
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


def split_dataset(ids, labels, test_size, val_size, random_state=42):
    train_val_ids, test_ids, train_val_labels, _ = train_test_split(
        ids, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    train_ids, val_ids, _, _ = train_test_split(
        train_val_ids,
        train_val_labels,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=train_val_labels,
    )
    return train_ids, val_ids, test_ids


def prepare_train_val_test(
    segment_size: int,
    batch_size: int,
    val_size: float = 0.1,
    test_size: float = 0.1,
    n_shifts: int = 1,
    incl_acc: bool = True,
    incl_gyro: bool = True,
    incl_mag: bool = True,
    incl_ecg: bool = True,
):
    session_ids, activity_ids = zip(*get_session_ids_with_labels())
    train_ids, val_ids, test_ids = split_dataset(
        session_ids, activity_ids, test_size, val_size
    )

    train_steps = calc_steps_per_epoch(train_ids, segment_size, batch_size, n_shifts)
    val_steps = calc_steps_per_epoch(val_ids, segment_size, batch_size, n_shifts)
    test_steps = calc_steps_per_epoch(test_ids, segment_size, batch_size, n_shifts)

    n_features = count_features(incl_acc, incl_gyro, incl_mag, incl_ecg)

    train_dataset = create_dataset(
        train_ids, segment_size, batch_size, n_features, n_shifts
    ).repeat()
    val_dataset = create_dataset(
        val_ids, segment_size, batch_size, n_features, n_shifts
    ).repeat()
    test_dataset = create_dataset(
        test_ids, segment_size, batch_size, n_features, n_shifts
    ).repeat()

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_steps,
        val_steps,
        test_steps,
        n_features,
    )
