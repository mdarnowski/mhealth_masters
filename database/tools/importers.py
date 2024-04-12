import io

import requests
from tqdm import tqdm

from data_layer.sensor_data import create_sensor_record
from database.model.session import Session


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
            session_obj = Session(subject_id=subject_id, activity_id=label - 1)
            db.add(session_obj)
            db.commit()
            last_label = label
            sequence = 1
        else:
            sequence += 1

        sensor_values = [float(value) for value in values[:-1]]
        sensor_record = create_sensor_record(sequence, session_obj.id, *sensor_values)
        db.add(sensor_record)

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
