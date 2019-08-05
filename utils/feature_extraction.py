import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import interpolate, signal

CONFIG = os.path.join('utils/config/analysis_config.json')


class TimeDomainMetrics:
    '''
    Computes time-domain features from R-R interval differences.
    '''

    def __init__(self, rr, fs):
        '''
        rr: RR-interval (in milliseconds)
        fs: sampling frequency
        '''
        self.rr = rr
        self.fs = fs
        self.metrics = self._get_time_domain_metrics()

    def _get_time_domain_metrics(self):
        '''
        fs: sampling frequency
        :return: dictionary of the calculated time-domain metrics
        '''
        return {'rmssd': np.sqrt(np.mean(np.square(np.diff(self.rr)))),
                'sdrr': np.std(self.rr),
                'nn50': np.sum(np.abs(np.diff(self.rr)) > 0.05 * self.fs) * 1,
                'pnn50': 100 * (np.sum(np.abs(np.diff(self.rr)) > 0.05 * self.fs) * 1) / len(self.rr)}


class FrequencyDomainMetrics:
    '''
    Computes frequency domain features from the power spectral decomposition.
    '''

    def __init__(self, rr, fs):
        '''
        rr: RR-interval (in milliseconds)
        fs: sampling frequency
        '''
        self.rr = rr
        self.fs = fs
        self.metrics = self._get_frequency_domain_metrics()

    @staticmethod
    def _read_config():
        return pd.read_json(CONFIG)['feature_extraction'].to_dict()

    def _get_frequency_domain_metrics(self):
        '''
        :return: dictionary of the calculated frequency-domain metrics
        '''
        config = self._read_config()

        timestamps = np.cumsum(self.rr) / 1000
        timestamps = timestamps - timestamps[0]
        # interpolation function
        funct = interpolate.interp1d(x=timestamps, y=[rrs / 1000 for rrs in self.rr], kind='linear')
        interpolated_timestamps = np.linspace(timestamps[0], timestamps[-1], num=timestamps[-1] * self.fs)
        rr_interpolation = funct(interpolated_timestamps)
        # compensate for DC
        rr_interpolation = rr_interpolation - np.mean(rr_interpolation)
        # compute PSD
        freq, psd = signal.welch(x=rr_interpolation, fs=self.fs, window='hann', nfft=2 ** 15, nperseg=25600)

        if eval(config['plot']):
            plt.figure(1, figsize=(12, 8))
            plt.plot(freq, psd)
            plt.xlim([0.0, 1.0])
            plt.xlabel('frequency [Hz]')
            plt.ylabel('PSD [s2/ Hz]')
            plt.show()

        vlf_indexes = np.logical_and(freq >= config['vlf_band'][0], freq < config['vlf_band'][1])
        lf_indexes = np.logical_and(freq >= config['lf_band'][0], freq < config['lf_band'][1])
        hf_indexes = np.logical_and(freq >= config['hf_band'][0], freq < config['hf_band'][1])

        lf = np.trapz(y=psd[lf_indexes], x=freq[lf_indexes])
        hf = np.trapz(y=psd[hf_indexes], x=freq[hf_indexes])

        vlf = np.trapz(y=psd[vlf_indexes], x=freq[vlf_indexes])
        total_power = vlf + lf + hf
        lf_hf_ratio = lf / hf
        lfnu = (lf / (lf + hf)) * 100
        hfnu = (hf / (lf + hf)) * 100

        frequency_band_index = [vlf_indexes, lf_indexes, hf_indexes]
        label_list = ["VLF component", "LF component", "HF component"]
        if eval(config['plot']):
            plt.figure(2, figsize=(12, 8))
            plt.xlabel("Frequency (Hz)", fontsize=15)
            plt.ylabel("PSD (s2/ Hz)", fontsize=15)

            plt.title("FFT Spectrum : Welch's periodogram", fontsize=20)
            for band_index, label in zip(frequency_band_index, label_list):
                plt.fill_between(freq[band_index], 0, psd[band_index] / (1000 * len(psd[band_index])), label=label)
                plt.legend(prop={"size": 15}, loc="best")
                plt.xlim(0, config['hf_band'][1])

            plt.show()

        return {'lf': lf,
                'hf': hf,
                'vlf': vlf,
                'total_power': total_power,
                'lf_hf_ratio': lf_hf_ratio,
                'lfnu': lfnu,
                'hfnu': hfnu}
