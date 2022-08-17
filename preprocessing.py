from sklearn.preprocessing import normalize

from sklearn.preprocessing import MinMaxScaler
import numpy as np
from hyperparameters import hyperparameters


def preprocess_stream(stream, hps = hyperparameters):
    preprocess_data_columnwise = hps.preprocess_data_columnwise
    normalising_data = hps.normalising_data
    scale_data = hps.scale_data
    time_vars = int(hps.time_variable) + int(hps.daytime_variable)
    if normalising_data:
        if preprocess_data_columnwise:
            stream_norm = normalize(stream, axis = 0)
            return(stream_norm)
        else:
            stream_data = stream[:, :-time_vars]
            stream_time = stream[:, -time_vars:]
            stream_data_norm = stream_data/np.linalg.norm(stream_data)
            stream_time_norm = normalize(stream_time, axis = 0)
            stream_norm = np.concatenate((stream_data_norm, stream_time_norm), axis = 1)
            return stream_norm
    if scale_data:
        if preprocess_data_columnwise:
            min_max_scaler = MinMaxScaler()
            stream_norm = min_max_scaler.fit_transform(stream)
            return(stream_norm)
        else:
            stream_data = stream[:, :-time_vars]
            stream_time = stream[:, -time_vars:]
            stream_data = stream_data/stream_data.max()
            min_max_scaler = MinMaxScaler()
            stream_time = min_max_scaler.fit_transform(stream_time)
            stream_norm = np.concatenate((stream_data, stream_time), axis = 1)
            return stream_norm
    return stream

def preprocess_signature(sig, hps = hyperparameters):
    preprocess_signature_columnwise = hps.preprocess_signature_columnwise
    normalising_signature = hps.normalising_signature
    scale_signature = hps.scale_signature
    if normalising_signature:
        if preprocess_signature_columnwise:
            sig_norm = normalize(sig, axis = 0)
            return(sig_norm)
        else:
            sig_norm = sig/np.linalg.norm(sig)
            return sig_norm
    if scale_signature:
        if preprocess_signature_columnwise:
            min_max_scaler = MinMaxScaler()
            sig_norm = min_max_scaler.fit_transform(sig)
            return(sig_norm)
        else:
            sig_norm = sig/sig.max()
            return sig_norm
    return sig
