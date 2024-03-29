from load_data.model.sensor_data import create_sensor_data
from load_data.model.session import Session
from load_data.my_session import SessionLocal
from load_data.setup_db import populate_initial_data


def import_data_from_log(subject_id, log_file_path):
    db = SessionLocal()

    last_label = None  # Variable to keep track of the last label encountered
    session_obj = None  # Variable for the current session object

    with open(log_file_path, "r") as file:
        for idx, line in enumerate(file):
            # Split the line assuming it's tab-separated
            values = line.strip().split("\t")

            # Assuming the last value is the label
            label = int(values[-1])
            if label == 0:
                continue

            # If the label changes (indicating a new session), or if it's the first line
            if label != last_label:
                # Create a new session, no need to check for existing ones
                session_obj = Session(subject_id=subject_id, activity_id=label)
                db.add(session_obj)
                db.commit()

                last_label = label  # Update the last label

            # Convert sensor data values from string to float, excluding the last label value
            sensor_data_values = list(map(float, values[:-1]))

            # Create SensorData instance and add to session
            sensor_data = create_sensor_data(
                idx + 1, session_obj.id, *sensor_data_values
            )
            db.add(sensor_data)

        db.commit()
    db.close()


populate_initial_data()
# Loop through log files and import data
for i in range(1, 11):
    import_data_from_log(
        subject_id=i, log_file_path=f"../data/log/mHealth_subject{i}.log"
    )
