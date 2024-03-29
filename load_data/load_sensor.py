import pandas as pd

from load_data.define_subjects import SessionLocal
from model.model import SensorData, Session


def import_sensor_data(file_path, subject_id):
    db = SessionLocal()
    # Assuming the filename or path contains a unique identifier for each subject
    df = pd.read_csv(file_path)

    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        label = row['Label']  # Assuming 'Label' is the last column as mentioned
        if label == 0:
            continue  # Skip rows where activity is not defined

        # Retrieve the session for the current subject and activity, or create it
        session_obj = db.query(Session).filter_by(subject_id=subject_id, activity_id=label).first()
        if not session_obj:
            session_obj = Session(subject_id=subject_id, activity_id=label)
            db.add(session_obj)
            db.commit()

        # Create and add the SensorData
        sensor_data = SensorData(
            session_id=session_obj.id,
            sequence=idx,
            # Assign all sensor data columns from the row
            acceleration_chest_x=row['acceleration from the chest sensor (X axis)'],
            # Repeat for all sensor data columns...
        )
        db.add(sensor_data)
    db.commit()
    db.close()


# Example usage - You'll need to loop through your actual files and subjects
import_sensor_data('path_to_file.csv', 1)
