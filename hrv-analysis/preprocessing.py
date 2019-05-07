import os

import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt, freqz
from scipy.signal import find_peaks

import logging
logging.getLogger().setLevel(logging.INFO)

mne.set_log_level('ERROR')

CONFIG = os.path.join('config/analysis_config.json')


class SignalFiltering:
    def __init__(self, fs, low, high):
        self.fs = fs
        self.nyq = 0.5 * self.fs
        self.norm_low = low / self.nyq
        self.norm_high = high / self.nyq

    def butter_highpass_filter(self, signal, order=5):
        b, a = butter(order, self.norm_high, btype='high', analog=False)
        y = filtfilt(b, a, signal)
        return y

    def butter_lowpass_filter(self, signal, order=5):
        b, a = butter(order, self.norm_low, btype='low', analog=False)
        y = filtfilt(b, a, signal)
        return y

    def butter_bandpass_filter(self, signal, order=5):
        b, a = butter(order, [self.norm_low, self.norm_high], btype='band')
        y = filtfilt(b, a, signal)
        return y

    def plot_bandpass_freqz(self, orders):
        plt.figure(1)
        plt.clf()
        for order in orders:
            b, a = butter(order, [self.norm_low, self.norm_high], btype='band')
            w, h = freqz(b, a, worN=2000)
            plt.plot((self.fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.legend(loc='best')
        plt.show()


class SignalPreprocessing:
    def __init__(self, data_path):
        self.data_path = data_path
        self.fs = 0
        self.preprocessed_data = self._preprocess_data()

    def _read_data(self):
        logging.info(f'Reading data from "{self.data_path}"')
        raw = mne.io.read_raw_brainvision(self.data_path, preload=True)
        self.fs = int(raw.info['sfreq'])
        start_time = self.fs * 2
        end_time = (self.fs * 5 * 60) + self.fs * 2
        data, _ = raw[raw.ch_names.index('ECG'), start_time:end_time]
        return data

    @staticmethod
    def _read_config():
        return pd.read_json(CONFIG)['preprocessing'].to_dict()

    def _preprocess_data(self):
        data = self._read_data()
        config = self._read_config()

        filtering = SignalFiltering(fs=self.fs, low=config['low_pass'], high=config['high_pass'])
        if eval(config['plot']):
            plt.figure(1)
            filtering.plot_bandpass_freqz(orders=[3, 5, 6])

        filtered_data = filtering.butter_bandpass_filter(signal=data[0, :], order=6)

        if eval(config['plot']):
            plt.figure(2, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
            plt.plot(data[0, self.fs * 1: (self.fs * config['display_limit'])])
            plt.plot(filtered_data[self.fs * 1: (self.fs * config['display_limit'])])
            plt.legend(['raw', 'filtered'])

        differentiated_data = np.ediff1d(filtered_data)
        squared_data = differentiated_data.T ** 2
        integrated_data = np.convolve(squared_data, np.ones(15))

        if eval(config['plot']):
            plt.figure(3, figsize=(22, 14), dpi=80, facecolor='w', edgecolor='k')
            plt.subplot(4, 1, 1)
            plt.plot(filtered_data[self.fs: (self.fs * config['display_limit'])])
            plt.legend(['filtered'])
            plt.subplot(4, 1, 2)
            plt.plot(differentiated_data[self.fs: (self.fs * config['display_limit'])])
            plt.legend(['derivative'])
            plt.subplot(4, 1, 3)
            plt.plot(squared_data[self.fs: (self.fs * config['display_limit'])])
            plt.legend(['squared'])
            plt.subplot(4, 1, 4)
            plt.plot(integrated_data[self.fs: (self.fs * config['display_limit'])])
            plt.legend(['integrative'])
            plt.show()

        logging.info('Reprocessing finished')
        return integrated_data


class PeakDetection:
    '''
    Detects peaks on preprocessed data based on Pan Tompkins algorithm and returns peak intervals in milliseconds.

    Reference:
    J. Pan and W. J. Tompkins, "A Real-Time QRS Detection Algorithm,"  in IEEE Transactions on Biomedical Engineering,
     vol. BME-32, no. 3, pp. 230-236, March 1985. doi: 10.1109/TBME.1985.325532
    '''

    def __init__(self, data, fs):
        self.fs = fs
        self.rr = self._detect_peaks(data)

    @staticmethod
    def _read_config():
        return pd.read_json(CONFIG)['peak_detection'].to_dict()

    def _detect_peaks(self, data):
        config = self._read_config()
        qrs_period = config['qrs_period'] * self.fs
        refractory_period = config['refractory_period'] * self.fs

        data = data[self.fs: (self.fs * 5 * 60)]
        rr = []

        def _batch(iterable, batch_length=1):
            iterable = list(iterable)
            ln = len(iterable)
            for ndx in range(0, ln, batch_length):
                yield np.asarray([e for e in iterable[ndx:min(ndx + batch_length, ln)]])

        def _get_rr_intervals_in_ms(rr_peak_indices, rr_min, rr_max):
            rr_intervals = np.diff(rr_peak_indices)
            rr_intervals = [(rri * 1000) / self.fs for rri in rr_intervals]
            rr_intervals = np.asarray(rr_intervals)
            return rr_intervals[(rr_intervals > rr_min) * (rr_intervals < rr_max)]

        epoch_length = int(self.fs * 30)
        epoch_num = 0
        for epoch in _batch(data, epoch_length):
            epoch_num += 1
            epoch_peaks = []

            last_peak = 0
            threshold_value = 0.0
            signal_peak_value = 0.0
            noise_peak_value = 0.0

            normalized_epoch = (epoch - min(epoch)) / (max(epoch) - min(epoch))
            peak_candidates, _ = find_peaks(normalized_epoch, height=0.1, distance=self.fs * 0.4)
            if len(peak_candidates):
                for ind, peak in enumerate(peak_candidates):
                    if qrs_period < (peak - last_peak) < refractory_period or not len(epoch_peaks):
                        if epoch[peak] > threshold_value:
                            epoch_peaks.append(peak)
                            signal_peak_value = 0.125 * epoch[peak] + 0.875 * signal_peak_value
                        else:
                            noise_peak_value = 0.125 * epoch[peak] + 0.875 * noise_peak_value

                        threshold_value = noise_peak_value + 0.25 * (signal_peak_value - noise_peak_value)

                    else:
                        signal_peak_value = 0.5 * signal_peak_value
                        threshold_value = noise_peak_value + 0.25 * (signal_peak_value - noise_peak_value)
                        if epoch[peak] > threshold_value:
                            epoch_peaks.append(peak)

                    last_peak = epoch_peaks[-1]

                epoch_rr_intervals = _get_rr_intervals_in_ms(rr_peak_indices=epoch_peaks, rr_min=400.0, rr_max=1400.0)

                rr.extend(epoch_rr_intervals)

            else:
                logging.info(f'No peak candidates were found in epoch {epoch_num}!')

            if eval(config['plot']):
                plt.figure()
                plt.plot(normalized_epoch)
                plt.plot(epoch_peaks, normalized_epoch[epoch_peaks], '*')
                plt.show()

        logging.info(f'Average RR interval in data: {np.mean(rr)} ms')

        return rr
