import pandas as pd
from datetime import datetime
import sys
import esig
import numpy as np

np.set_printoptions(threshold=sys.maxsize)  

#observed airports -> oa
oa = ["EGLL", "EGKK", "EGGW", "EGCC", "EGPH"]
depth = 2

def datetime_convert(row):
    date = row['date']
    time = row['time']
    date = date[2:]
    date_time = date + " " + time
    date_time_obj = datetime.strptime(date_time, '%y-%m-%d %H:%M:%S')
    return str(int(round(date_time_obj.timestamp())))

def single_icao(row):
    origin_icao = row["origin_icao"]
    destination_icao = row["destination_icao"]
    if row["event"] == "takeoff" :
        return origin_icao
    else:
        return destination_icao

df = pd.read_csv('C:/Users/dmitr/Desktop/project work/air_graph.csv')
events = df[(df["event"] == "takeoff") | (df["event"] == "landed")]
events['timestamp'] = events.apply(lambda row: datetime_convert(row), axis = 1)
events['icao'] = events.apply(lambda row: single_icao(row), axis = 1)
events = events[['icao', 'timestamp', "event"]]
relevant_events = events[events["icao"].isin(oa)]
relevant_events = relevant_events.sort_values("timestamp")
stream = np.array([[0.0 for i in range(len(oa) * 2 + 1)] for i in range(int(relevant_events.size / 3))])
count = 0
entry = np.array([0.0 for i in range(len(oa) * 2 + 1)])
for _, row in relevant_events.iterrows():
    index = 0
    for i in range(5):
        if row['icao'] == oa[i]:
            index = 2 * i
    if row['event'] == 'takeoff':
        index += 1
    entry[index] += 1
    entry[-1] = row['timestamp']
    stream[count] = entry
    count += 1

print(stream.shape)
sig = esig.stream2sig(stream, depth)
print(sig)