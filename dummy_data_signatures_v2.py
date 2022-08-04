from signal import signal
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import normalize

from sklearn.preprocessing import MinMaxScaler
import sys
import esig
import numpy as np

esig.set_backend("iisignature")

np.set_printoptions(threshold=sys.maxsize)  

day = 86400

#observed airports -> oa
oa = [np.int_(0), np.int_(1), np.int_(2), np.int_(3), np.int_(4)]
depth = 2
time_variable = True
daytime_variable = True
stream_length = 7 * day
stream_spacing = 2 * day

normalising_data = False 
scale_data = True
preprocess_data_columnwise = False
normalising_signature = False
scale_signature = False
preprocess_signature_columnwise = False

#data files reading
relevant_events = pd.read_csv('C:\\Users\\dmitr\\Desktop\\project work\\dummy data\\data.csv')
anomalous_events_labels = pd.read_csv('C:\\Users\\dmitr\\Desktop\\project work\\dummy data\\data_anomaly_labels.csv')
anomalous_events_timestamps = set(anomalous_events_labels['timestamp'].values.tolist())
relevant_events = relevant_events[relevant_events["icao"].isin(oa)]

def normalise_stream(stream):
    if normalising_data:
        if preprocess_data_columnwise:
            stream_norm = normalize(stream, axis = 0)
            return(stream_norm)
        else:
            time_vars = time_variable + daytime_variable
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
            time_vars = time_variable + daytime_variable
            stream_data = stream[:, :-time_vars]
            stream_time = stream[:, -time_vars:]
            stream_data = stream_data/stream_data.max()
            min_max_scaler = MinMaxScaler()
            stream_time = min_max_scaler.fit_transform(stream_time)
            stream_norm = np.concatenate((stream_data, stream_time), axis = 1)
            return stream_norm
    return stream

def normalise_signature(sig):
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




split_stream = []
ends = []
anomaly_count = []
window_left = 0
window_right = 0
new_stream_delay = relevant_events['timestamp'][0] - 1

for _, row in relevant_events.iterrows():
    time = row['timestamp']
    if time > new_stream_delay:
        split_stream.append([])
        anomaly_count.append(0)
        ends.append(time + stream_length)
        window_right += 1
        new_stream_delay += stream_spacing
    while window_left > window_right or time > ends[window_left]:
        window_left += 1

    index = 0
    for i in range(len(oa)):
        if row['icao'] == oa[i]:
            index = 2 * i
    if row['event'] == 'departed':
        index += 1

    for i in range(window_left, window_right):
        if(row['timestamp'] in anomalous_events_timestamps):
            anomaly_count[i] += 1
        if len(split_stream[i]) == 0:
            entry = [0.0 for i in range(len(oa) * 2 + time_variable + daytime_variable)]
        else:
            entry = split_stream[i][-1].copy()
        entry[index] += 1
        if time_variable:
            entry[-1] = row['timestamp']
        if daytime_variable:
            entry[-1 - time_variable] = row['timestamp'] % day
        split_stream[i].append(entry)

signatures = []


for stream in split_stream:
    np_stream = np.array(stream)
    stream = normalise_stream(np_stream)
    sig = esig.stream2sig(np_stream, depth)
    sig = normalise_signature(sig)
    signatures.append(sig)

print(len(signatures))

df = pd.DataFrame(signatures)
#desination file name
filename = 'C:\\Users\\dmitr\\Desktop\\project work\\dummy data\\signatures_' + str(depth) +'_'+ str(stream_length / day) +'_' + str(stream_spacing / day) +'_time_' * time_variable + '_daytime_' * daytime_variable  + '.csv'

anomaly_count_labels = pd.DataFrame(anomaly_count, columns = ["count"])
#destination file name
anomaly_count_filename = 'C:\\Users\\dmitr\\Desktop\\project work\\dummy data\\signatures_' + str(depth) +'_'+ str(stream_length / day) + '_' + str(stream_spacing / day) +'_time_' * time_variable + '_daytime_' * daytime_variable  + '_ANOMALYCOUNT_' + '.csv'


df.to_csv(filename)
anomaly_count_labels.to_csv(anomaly_count_filename)
print(filename)
