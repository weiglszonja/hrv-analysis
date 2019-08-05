import os
import pandas as pd
from datetime import datetime
from utils.preprocessing import SignalPreprocessing, PeakDetection
from utils.feature_extraction import TimeDomainMetrics, FrequencyDomainMetrics

headers = ['asrt_1_1', 'asrt_1_2', 'asrt_1_3', 'asrt_1_4',
           'asrt_1_5', 'asrt_2', 'asrt_3_1', 'asrt_3_2']


def initialize():
    if 'SOURCE' not in os.environ:
        os.environ['SOURCE'] = 'data/'
    if 'TARGET' not in os.environ:
        os.environ['TARGET'] = 'results/'


def main():
    initialize()

    source_path = os.getenv('SOURCE')
    target_path = os.getenv('TARGET')
    subjects = os.listdir(source_path)

    subjects_data = pd.DataFrame()
    for subject in subjects:
        for header in headers:
            subject_data = pd.DataFrame({'subject': [subject], 'header': [header], 'id': [f'{subject}_{header}']})

            preprocessing = SignalPreprocessing(
                data_path=os.path.join(source_path, subject, f'{subject}_{header}.vhdr'))

            peak_detection = PeakDetection(data=preprocessing.preprocessed_data,
                                           fs=preprocessing.fs,
                                           filtered_data=preprocessing.filtered_data)

            time_domain_metrics = TimeDomainMetrics(rr=peak_detection.rr,
                                                    fs=preprocessing.fs)

            frequency_domain_metrics = FrequencyDomainMetrics(rr=peak_detection.rr,
                                                              fs=preprocessing.fs)

            subject_data = subject_data.assign(**{**time_domain_metrics.metrics, **frequency_domain_metrics.metrics})
            subjects_data = subjects_data.append(subject_data, ignore_index=True)

    dt = datetime.utcnow().strftime("%Y%m%d")
    target = os.path.join(target_path, dt, 'hrv')
    os.makedirs(target, exist_ok=True)
    subjects_data.to_csv(os.path.join(target, f'{dt}_subjects_hrv.csv'))


if __name__ == '__main__':
    main()
