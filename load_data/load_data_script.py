# reading log files source
# https://www.kaggle.com/code/nirmalsankalana/mhealth-dataset-data-set-reading-log-files

import pandas as pd

dir_name = '../data/log/mHealth_subject'


def log_to_csv(log_file_number):
    with open(dir_name + log_file_number + '.log') as file:
        lines = file.read().split('\n')
        lines = [lines[x].split('\t') for x in range(0, len(lines))]
    df = pd.DataFrame(lines, columns=column_names)
    df.to_csv('../data/csv/mHealth_subject' + log_file_number + '.csv', index=False)
    df.head()


column_names = ['acceleration from the chest sensor (X axis)',
                'acceleration from the chest sensor (Y axis)',
                'acceleration from the chest sensor (Z axis)',
                'electrocardiogram signal (lead 1)',
                'electrocardiogram signal (lead 2)',
                'acceleration from the left-ankle sensor (X axis)',
                'acceleration from the left-ankle sensor (Y axis)',
                'acceleration from the left-ankle sensor (Z axis)',
                'gyro from the left-ankle sensor (X axis)',
                'gyro from the left-ankle sensor (Y axis)',
                'gyro from the left-ankle sensor (Z axis)',
                'magnetometer from the left-ankle sensor (X axis)',
                'magnetometer from the left-ankle sensor (Y axis)',
                'magnetometer from the left-ankle sensor (Z axis)',
                'acceleration from the right-lower-arm sensor (X axis)',
                'acceleration from the right-lower-arm sensor (Y axis)',
                'acceleration from the right-lower-arm sensor (Z axis)',
                'gyro from the right-lower-arm sensor (X axis)',
                'gyro from the right-lower-arm sensor (Y axis)',
                'gyro from the right-lower-arm sensor (Z axis)',
                'magnetometer from the right-lower-arm sensor (X axis)',
                'magnetometer from the right-lower-arm sensor (Y axis)',
                'magnetometer from the right-lower-arm sensor (Z axis)',
                'Label']

for i in range(1, 11):
    log_to_csv(str(i))
