import os

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from matplotlib import pyplot as plt

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
        return {'meanrr': np.mean(self.rr),
                'rmssd': np.sqrt(np.mean(np.square(np.diff(self.rr)))),
                'sdrr': np.std(self.rr, ddof=1),
                'nn50': np.sum(np.abs(np.diff(self.rr)) > 50),
                'pnn50': 100 * (np.sum(np.abs(np.diff(self.rr)) > 50)) / len(self.rr)}


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
        freq, psd = LombScargle(timestamps, self.rr, normalization='psd').autopower(
            minimum_frequency=config['vlf_band'][0],
            maximum_frequency=config['hf_band'][1])

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

        frequency_band_index = [lf_indexes, hf_indexes]
        label_list = ["LF component", "HF component"]
        if eval(config['plot']):
            plt.figure(2, figsize=(12, 8))
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD (s2/ Hz)")
            plt.title("FFT Spectrum : Lomb-Scargle's periodogram", fontsize=20)
            for band_index, label in zip(frequency_band_index, label_list):
                plt.fill_between(freq[band_index], 0, psd[band_index] / (1000 * len(psd[band_index])), label=label)
                plt.legend(prop={"size": 18}, loc="best")
                plt.xlim(config['lf_band'][0], config['hf_band'][1])

            plt.show()

        return {'lf': lf,
                'hf': hf,
                'vlf': vlf,
                'total_power': total_power,
                'lf_hf_ratio': lf_hf_ratio,
                'lfnu': lfnu,
                'hfnu': hfnu}
