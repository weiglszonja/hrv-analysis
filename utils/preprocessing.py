import os

import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import pylab
from scipy.signal import butter, filtfilt, freqz
from scipy.signal import find_peaks

import logging
logging.getLogger().setLevel(logging.INFO)

mne.set_log_level('ERROR')

CONFIG = os.path.join('utils/config/analysis_config.json')

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)


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
        for order in orders:
            b, a = butter(order, [self.norm_low, self.norm_high], btype='band')
            w, h = freqz(b, a, worN=2000)
            plt.plot((self.fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        plt.xlabel(r'Normalized Frequency ($\times$$\pi$ rad/sample)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(False)
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

        self.filtered_data = filtering.butter_bandpass_filter(signal=data[0, :], order=6)

        if eval(config['plot']):
            sample_raw = data[0, 0: (self.fs * config['display_limit'] + 1)]
            sample_filtered = self.filtered_data[0: (self.fs * config['display_limit'] + 1)]
            times = np.linspace(0, len(sample_raw) / self.fs, len(sample_raw))

            plt.plot(times, sample_raw)
            plt.plot(times, sample_filtered)
            plt.ylabel('Voltage (mV)')
            plt.xlabel('Time (s)')
            plt.xticks(np.arange(0, config['display_limit'] + 1, 1))
            plt.legend(['Original signal', 'Filtered signal'])

        differentiated_data = np.ediff1d(self.filtered_data)
        squared_data = differentiated_data.T ** 2
        integrated_data = np.convolve(squared_data, np.ones(15))

        if eval(config['plot']):
            sample_derivative = differentiated_data[0: (self.fs * config['display_limit'] + 1)]
            sample_squared = squared_data[0: (self.fs * config['display_limit'] + 1)]
            sample_integrated = integrated_data[0: (self.fs * config['display_limit'] + 1)]
            plt.figure(3, figsize=(22, 14), dpi=80, facecolor='w', edgecolor='k')
            plt.subplot(5, 1, 1)
            plt.plot(times, sample_raw)
            plt.xticks(np.arange(0, config['display_limit'] + 1, 1))
            plt.legend(['Original'], loc='upper right')
            plt.subplot(5, 1, 2)
            plt.plot(times, sample_filtered)
            plt.xticks(np.arange(0, config['display_limit'] + 1, 1))
            plt.legend(['Filtering'], loc='upper right')
            plt.subplot(5, 1, 3)
            plt.plot(times, sample_derivative)
            plt.xticks(np.arange(0, config['display_limit'] + 1, 1))
            plt.legend(['Differentiation'], loc='upper right')
            plt.subplot(5, 1, 4)
            plt.plot(times, sample_squared)
            plt.xticks(np.arange(0, config['display_limit'] + 1, 1))
            plt.legend(['Squaring'], loc='upper right')
            plt.subplot(5, 1, 5)
            plt.plot(times, sample_integrated)
            plt.xticks(np.arange(0, config['display_limit'] + 1, 1))
            plt.legend(['Integration'], loc='upper right')
            plt.xlabel('Time (s)')
            plt.show()

        logging.info('Preprocessing finished')
        return integrated_data


class PeakDetection:
    '''
    Detects peaks on preprocessed data based on Pan Tompkins algorithm and returns peak intervals in milliseconds.

    Reference:
    J. Pan and W. J. Tompkins, "A Real-Time QRS Detection Algorithm,"  in IEEE Transactions on Biomedical Engineering,
     vol. BME-32, no. 3, pp. 230-236, March 1985. doi: 10.1109/TBME.1985.325532
    '''

    def __init__(self, data, fs, filtered_data):
        self.fs = fs
        self.epoch_length = 30
        self.filtered_data = filtered_data
        self.rr = self._detect_peaks(data)

    @staticmethod
    def _read_config():
        return pd.read_json(CONFIG)['peak_detection'].to_dict()

    def _detect_peaks(self, data):
        config = self._read_config()
        qrs_period = config['qrs_period'] * self.fs
        refractory_period = config['refractory_period'] * self.fs

        data = data[0: (self.fs * 5 * 60)]
        self.peaks = []
        rr = []

        def _batch(iterable, batch_length=1):
            iterable = list(iterable)
            ln = len(iterable)
            for ndx in range(0, ln, batch_length):
                yield np.asarray([e for e in iterable[ndx:min(ndx + batch_length, ln)]])

        def _filter_irregular_rr_peaks(rr_peaks, rr_min, rr_max):
            nn_intervals = []
            nn_peaks = []

            rr_batches = [rr_peaks[i:min(i + 8, len(rr_peaks))] for i in range(0, len(rr_peaks), 8)]
            if len(rr_batches[-1]) == 1:
                try:
                    rr_batches[-2].extend(rr_batches[-1])
                    rr_batches.remove(rr_batches[-1])
                except Exception:
                    raise

            for rr_buffer in rr_batches:
                rr_intervals = np.diff(rr_buffer)
                # convert to ms
                rr_intervals = [(rri * 1000) / self.fs for rri in rr_intervals]

                average_rr = np.mean(rr_intervals)

                if rr_min < average_rr < rr_max:
                    rr_min = 0.66 * np.mean(rr_intervals)
                    rr_max = 1.16 * np.mean(rr_intervals)
                    nn_intervals.extend(rr_intervals)
                    nn_peaks.extend(rr_buffer)

            return nn_peaks, nn_intervals

        epoch_length = int(self.fs * self.epoch_length)
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

                if len(epoch_peaks) > 1:
                    epoch_nn_peaks, epoch_nn_intervals = _filter_irregular_rr_peaks(rr_peaks=epoch_peaks,
                                                                                    rr_min=600.0,
                                                                                    rr_max=1200.0)
                    self.peaks.append(epoch_nn_peaks)
                    rr.extend(epoch_nn_intervals)

            else:
                logging.info(f'No peak candidates were found in epoch {epoch_num}!')

            if eval(config['plot']):
                plt.figure(figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
                plt.plot(normalized_epoch)
                plt.plot(epoch_nn_peaks, normalized_epoch[epoch_nn_peaks], '*', markersize=12)
                plt.xticks(np.arange(0, len(normalized_epoch), self.fs * 2))
                plt.title('Detected peaks on preprocessed signal', fontsize=16)
                plt.xlabel('Time (data samples)')
                plt.ylabel('Normalized value')
                plt.show()

                plt.figure(figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
                filt = - self.filtered_data[0: epoch_length]
                normalized_filt = (filt - min(filt)) / (max(filt) - min(filt))
                plt.plot(normalized_filt)
                plt.plot(epoch_nn_peaks, normalized_epoch[epoch_nn_peaks], '*', markersize=12)

                plt.xticks(np.arange(0, len(normalized_epoch), self.fs * 2))
                plt.xlabel('Time (data samples)')
                plt.ylabel('Normalized value')
                plt.title('Detected peaks on original signal', fontsize=16)
                plt.show()

        logging.info(f'Average RR interval in data: {np.mean(rr)} ms')

        return rr
