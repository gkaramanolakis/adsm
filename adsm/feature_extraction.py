import librosa
import numpy as np
import help_functions

def get_mfccdd(fpath, n_mfcc=13, winsize=0.25, sampling_rate=16000):
    '''
    Compute MFCCs, first and second derivatives
    :param fpath: the file path
    :param n_mfcc: the number of MFCC coefficients. Default = 13 coefficients
    :param winsize: the time length of the window for MFCC extraction. Default 0.25s (250ms)
    :param sampling_rate: the sampling rate. The file is loaded and converted to the specified sampling rate. 
    :return: a 2D numpy matrix (frames * MFCCdd) 
    '''
    help_functions.check_existence(fpath)
    data, sr = librosa.load(fpath, sr=sampling_rate, mono=True)
    winlen = int(winsize * sr)
    winstep = int(winlen / 2.0)
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, n_fft=winlen, hop_length=winstep)
    deltas = librosa.feature.delta(mfccs)
    deltadeltas = librosa.feature.delta(deltas)
    mfccdd = np.concatenate((mfccs, deltas, deltadeltas), axis=1)
    return mfccdd

