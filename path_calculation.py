from sqlite3 import Timestamp
import sys
import esig
import numpy as np
import pandas as pd
import itertools

from hyperparameters import hyperparameters

esig.set_backend("iisignature")

np.set_printoptions(threshold=sys.maxsize)  

day = 86400

class index_calculator:
    subsets = []

    def __init__(self, elements) -> None:
        self.subsets = list(map(set, itertools.combinations(set(elements), 2)))
    
    def calc_index(self, x, y):
        set = {x, y}
        return self.subsets.index(set)
    
    

def paths_from_vertices(events, anomalous_events_timestamps, hps = hyperparameters):
    observed = hps.observed
    stream_length = hps.stream_length
    stream_spacing = hps.stream_spacing
    time_variable = hps.time_variable
    daytime_variable = hps.daytime_variable
    no_extrapollation = hps.no_extrapollation
    lead_lag = hps.lead_lag

    events = events[events["icao"].isin(observed)]
    features = len(observed) * 2  + time_variable + daytime_variable
    split_stream = []
    ends = []
    anomaly_count = []
    window_left = 0
    window_right = 0
    new_stream_delay = events['timestamp'][0] - 1
    

    for _, row in events.iterrows():
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
        for i in range(len(observed)):
            if row['icao'] == observed[i]:
                index = 2 * i
        if row['event'] == 'takeoff':
            index += 1

        for i in range(window_left, window_right):
            if(row['id'] in anomalous_events_timestamps):
                anomaly_count[i] += 1
            if len(split_stream[i]) == 0:
                entry = [0.0 for i in range(features * (1 + lead_lag))]
            else:
                entry = split_stream[i][-1].copy()
            if (lead_lag):
                old_entry = entry[:features]
                if time_variable:
                    entry[features-1] = row['timestamp']
                if daytime_variable:
                    entry[features - 1 - time_variable] = row['timestamp'] % day
                if no_extrapollation:
                    split_stream[i].append(entry.copy())
                entry[index] += 1
                if no_extrapollation:
                    split_stream[i].append(entry.copy())
                entry[features:] = old_entry
                split_stream[i].append(entry.copy())
            else:
                if time_variable:
                    entry[-1] = row['timestamp']
                if daytime_variable:
                    entry[-1 - time_variable] = row['timestamp'] % day
                if no_extrapollation:
                    split_stream[i].append(entry.copy())
                entry[index] += 1
                split_stream[i].append(entry.copy())
    return (split_stream, anomaly_count)


def paths_from_edges(events, anomalous_events_id, hps = hyperparameters):
    observed = hps.observed
    stream_length = hps.stream_length
    stream_spacing = hps.stream_spacing
    time_variable = hps.time_variable
    daytime_variable = hps.daytime_variable
    no_extrapollation = hps.no_extrapollation
    lead_lag = hps.lead_lag
    indexes = index_calculator(hps.observed)

    events = events[events["icao"].isin(observed)]
    takeoff = events.loc[events['event'] == 'takeoff']
    landed =  events.loc[events['event'] == 'landed']
    flights = pd.merge(takeoff, landed, on='id')

    events = []
    for _, flight in flights.iterrows():
        index = indexes.calc_index(flight['icao_x'], flight['icao_y'])
        events.append([index, flight['timestamp_x'], "takeoff", flight['id']])
        events.append([index, flight['timestamp_y'], "landed", flight['id']])
    
        

    nobs = len(observed)
    features = int(nobs * (nobs - 1) / 2  + time_variable + daytime_variable)
    split_stream = []
    ends = []
    anomaly_count = []
    window_left = 0
    window_right = 0
    new_stream_delay = events[0][1] - 1
    

    for [index, timestamp, event, id] in events:
        if timestamp > new_stream_delay:
            split_stream.append([])
            anomaly_count.append(0)
            ends.append(timestamp + stream_length)
            window_right += 1
            new_stream_delay += stream_spacing
        while window_left > window_right or timestamp > ends[window_left]:
            window_left += 1

        for i in range(window_left, window_right):
            if(id in anomalous_events_id):
                anomaly_count[i] += 1
            if len(split_stream[i]) == 0:
                entry = [0.0 for i in range(features * (1 + lead_lag))]
            else:
                entry = split_stream[i][-1].copy()
            if (lead_lag):
                old_entry = entry[:features]
                if time_variable:
                    entry[features-1] = timestamp
                if daytime_variable:
                    entry[features - 1 - time_variable] = timestamp % day
                if no_extrapollation:
                    split_stream[i].append(entry.copy())
                if event == "takeoff":
                    entry[index] += 1
                else:
                    entry[index] -= 1
                if no_extrapollation:
                    split_stream[i].append(entry.copy())
                entry[features:] = old_entry
                split_stream[i].append(entry.copy())
            else:
                if time_variable:
                    entry[-1] = timestamp
                if daytime_variable:
                    entry[-1 - time_variable] = timestamp % day
                if no_extrapollation:
                    split_stream[i].append(entry.copy())
                if event == "takeoff":
                    entry[index] += 1
                else:
                    entry[index] -= 1
                split_stream[i].append(entry.copy())
    return (split_stream, anomaly_count)

