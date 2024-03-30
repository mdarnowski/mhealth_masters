import io
from zipfile import ZipFile

import requests
from tqdm import tqdm

from load_data import my_conn
from load_data.db_model.activity import Activity
from load_data.db_model.base import Base
from load_data.db_model.sensor_data import create_sensor_data
from load_data.db_model.session import Session


def reload_db(engine):
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def define_activities(db):
    activities = [
        (1, "Standing still"),
        (2, "Sitting and relaxing"),
        (3, "Lying down"),
        (4, "Walking"),
        (5, "Climbing stairs"),
        (6, "Waist bends forward"),
        (7, "Frontal elevation of arms"),
        (8, "Knees bending"),
        (9, "Cycling"),
        (10, "Jogging"),
        (11, "Running"),
        (12, "Jump front & back"),
    ]
    db.add_all([Activity(id=act_id, description=desc) for act_id, desc in activities])

    db.commit()


def download_file_in_memory(url):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(
        total=total_size_in_bytes, unit="iB", unit_scale=True, desc="Downloading"
    )
    data_stream = io.BytesIO()
    for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        data_stream.write(data)
    progress_bar.close()
    data_stream.seek(0)
    return data_stream


def import_data_from_log(db, subject_id, file):
    last_label = None
    session_obj = None
    sequence = 1

    for line in file:
        values = line.decode("utf-8").strip().split("\t")
        label = int(values[-1])

        if label == 0:
            last_label = label
            continue

        if label != last_label:
            session_obj = Session(subject_id=subject_id, activity_id=label)
            db.add(session_obj)
            db.commit()
            last_label = label
            sequence = 1
        else:
            sequence += 1

        sensor_data_values = [float(value) for value in values[:-1]]
        sensor_data = create_sensor_data(sequence, session_obj.id, *sensor_data_values)
        db.add(sensor_data)

    db.commit()


if __name__ == "__main__":
    db_engine, database = my_conn.load_engine()

    reload_db(db_engine)
    define_activities(database)

    zip_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip"
    zip_in_memory = download_file_in_memory(zip_url)

    with ZipFile(zip_in_memory) as zip_ref:
        for i in tqdm(range(1, 11), desc="Processing log files"):
            with zip_ref.open(f"MHEALTHDATASET/mHealth_subject{i}.log") as log_file:
                import_data_from_log(database, subject_id=i, file=log_file)

    database.close()
