import numpy as np
import matplotlib.pyplot as plt
import librosa 
from scipy.ndimage import gaussian_filter1d


def minmax(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def vad(spec, plot = False):
    mag = spec.squeeze().sum(axis = 0)
    mag_ = minmax(mag)

    threshold =  np.quantile(mag_, 0.5)

    va_ind1 = np.where(mag_ >= threshold, 1, 0)
    va_point = np.where(va_ind1==1)[0][ [0, -1] ]
    if va_point[0] <0:
        va_point[0] = 0
    
    if plot:
        va_ind2 = np.zeros_like(mag)
        va_ind2[va_point[0]:va_point[1]] = 1
        plt.plot( mag_ )
        plt.hlines( xmin = 0, xmax = len(mag_),  y = threshold)
        plt.plot( np.where(mag_ >= threshold, 1, 0))
        plt.plot( va_ind2)
    return va_point



def gvad(spec, plot = False, n_spare = 3):
    time_mag = minmax(librosa.amplitude_to_db(spec)).sum(axis = 0)
    sorted_time_mag = np.sort(time_mag)
    sorted_time_mag_smoothed = gaussian_filter1d(sorted_time_mag, sigma = 10)
    sorted_time_mag_smoothed_grad = np.gradient(sorted_time_mag_smoothed)
    threshold = sorted_time_mag_smoothed[np.argmax(sorted_time_mag_smoothed_grad)]
    va_point = (np.where( time_mag > threshold)[0][ [0, -1]] + np.array([-n_spare, n_spare])).tolist()
    if va_point[0] <0:
        va_point[0] = 0
    if plot:
        plt.plot(time_mag)
        vad_line = np.zeros_like(time_mag)
        vad_line[va_point[0]: va_point[1]] = np.max(time_mag)
        plt.plot(vad_line)
    return va_point
