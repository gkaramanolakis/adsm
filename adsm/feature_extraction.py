import librosa
import numpy as np
import help_functions

def extract_mfccdd(fpath, n_mfcc=13, winsize=0.25, sampling_rate=16000):
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
    winlen = int(2 * winsize * sr)
    winstep = int(winlen / 2.0)
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, n_fft=winlen, hop_length=winstep)
    deltas = librosa.feature.delta(mfccs)
    deltadeltas = librosa.feature.delta(deltas)
    mfccdd = np.concatenate((mfccs, deltas, deltadeltas), axis=1)
    return mfccdd

def extract_multiple_features(fpath, n_mfcc=13, sampling_rate=16000):
    chroma_feature = librosa.feature.chroma_stft(fpath, sampling_rate) # 12
    mfcc_feature = librosa.feature.mfcc(fpath, sampling_rate, n_mfcc=n_mfcc) # default = 20
    rmse_feature = librosa.feature.rmse(fpath) # 1
    spectral_centroid_feature = librosa.feature.spectral_centroid(fpath, sampling_rate) #1
    spectral_bandwidth_feature = librosa.feature.spectral_bandwidth(fpath, sampling_rate) #1
    #spectral_contrast_feature = librosa.feature.spectral_contrast(data,rate) #7
    spectral_rolloff_feature = librosa.feature.spectral_rolloff(fpath, sampling_rate) #1
    poly_features = librosa.feature.poly_features(fpath, sampling_rate) #2
    #tonnetz_feature = librosa.feature.tonnetz(data,rate) #6
    zero_crossing_rate_feature = librosa.feature.zero_crossing_rate(fpath, sampling_rate) #1

    l = len(chroma_feature[0])
    chroma_feature = np.reshape(chroma_feature,[l ,len(chroma_feature)])
    mfcc_feature = np.reshape(mfcc_feature,[l ,len(mfcc_feature)])
    rmse_feature = np.reshape(rmse_feature,[l ,len(rmse_feature)])
    spectral_centroid_feature = np.reshape(spectral_centroid_feature,[l ,len(spectral_centroid_feature)])
    spectral_bandwidth_feature = np.reshape(spectral_bandwidth_feature,[l ,len(spectral_bandwidth_feature)])
    #spectral_contrast_feature = np.reshape(spectral_contrast_feature,[l ,len(spectral_contrast_feature)])
    spectral_rolloff_feature = np.reshape(spectral_rolloff_feature,[l ,len(spectral_rolloff_feature)])
    poly_features = np.reshape(poly_features,[l ,len(poly_features)])
    #tonnetz_feature = np.reshape(tonnetz_feature,[l ,len(tonnetz_feature)])
    zero_crossing_rate_feature = np.reshape(zero_crossing_rate_feature,[l ,len(zero_crossing_rate_feature)])

    # Concatenate all features to a feature vector (length = 32)
    features =  np.concatenate((chroma_feature,mfcc_feature,rmse_feature,
                  spectral_centroid_feature,spectral_bandwidth_feature,
                  spectral_rolloff_feature, poly_features,
                  zero_crossing_rate_feature),axis=1)
    return features