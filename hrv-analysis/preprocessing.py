import os
import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt, freqz

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
