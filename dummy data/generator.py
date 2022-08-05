import pandas as pd 
import numpy as np 
import esig 
import random

data = []
clean_data = []

times = [
[0, 3*3600, 0, 0, 4*3600],
[3*3600, 0, 6*3600, 4*3600, 0],
[0, 6*3600 ,0, 0, 0],
[0, 4*3600, 0, 0, 7*3600],
[4*3600, 0, 0, 7*3600, 0]
]

edges = [(0,1), (0,4), (1, 0), (1, 2), (1, 3), (2, 1), (3, 1), (3, 4), (4, 0), (4, 3)]
anomaly_weights = (1.5, 0.5, 1.5, 1, 0.5, 1, 0.5, 1.5, 0.5, 1.5)
anomaly_chaining_delay_left = 10800
anomaly_chaining_delay_right = 12000
flight_len_spread = 0.2

timestamp = 0
timestamp_2 = 0
anomaly_labels = []

for i in range(500000):
    timestamp_2 += (169 + random.randrange(-60, 60))
    (a, b) = random.choice(edges)
    delay = times[a][b] * (random.random() * flight_len_spread + (1 - flight_len_spread/2))
    clean_data.append([a, timestamp_2, "departed"])
    clean_data.append([b, timestamp_2+delay, "landed"])

for i in range(200000):
    timestamp += (169 + random.randrange(-60, 60))
    (a, b) = random.choice(edges)
    delay = times[a][b] * (random.random() * flight_len_spread + (1 - flight_len_spread/2))
    data.append([a, timestamp, "departed"])
    data.append([b, timestamp+delay, "landed"])

#chaining anomaly
for i in range(3000):
    timestamp += (169 + random.randrange(-60, 60))
    (a, b) = random.choice([x for x in edges if x != (1, 2)])
    delay = times[a][b] * (random.random() * flight_len_spread + (1 - flight_len_spread/2))
    data.append([a, timestamp, "departed"])
    data.append([b, timestamp+delay, "landed"])
    if(a == 0 and b == 1):
        delay2 = random.randrange(anomaly_chaining_delay_left, anomaly_chaining_delay_right)
        delay3 = 6 * 3600 * (random.random() * flight_len_spread + (1 - flight_len_spread/2))
        data.append([1, timestamp + delay + delay2, "departed"])
        anomaly_labels.append([timestamp + delay + delay2, 1])
        data.append([2, timestamp + delay + delay2 + delay3, "landed"])
        anomaly_labels.append([timestamp + delay + delay2 + delay3, 1])

for i in range(100000):
    timestamp += (169 + random.randrange(-60, 60))
    (a, b) = random.choice(edges)
    delay = times[a][b] * (random.random() * flight_len_spread + (1 - flight_len_spread/2))
    data.append([a, timestamp, "departed"])
    data.append([b, timestamp+delay, "landed"])

#rate anomaly
for i in range(3000):
    timestamp += (169 + random.randrange(-60, 60))
    (a, b) = random.choices(edges, weights = anomaly_weights)[0]
    delay = times[a][b] * (random.random() * flight_len_spread + (1 - flight_len_spread/2))
    data.append([a, timestamp, "departed"])
    anomaly_labels.append([timestamp, 2])
    data.append([b, timestamp+delay, "landed"])
    anomaly_labels.append([timestamp + delay, 2])


for i in range(100000):
    timestamp += (169 + random.randrange(-60, 60))
    (a, b) = random.choice(edges)
    delay = times[a][b] * (random.random() * flight_len_spread + (1 - flight_len_spread/2))
    data.append([a, timestamp, "departed"])
    data.append([b, timestamp+delay, "landed"])


df = pd.DataFrame(data, columns = ["icao", "timestamp", "event"])
df_anomaly = pd.DataFrame(anomaly_labels, columns = ['timestamp', 'type'])
df_clean = pd.DataFrame(clean_data, columns = ["icao", "timestamp", "event"])
df = df.sort_values("timestamp")
df_clean = df_clean.sort_values("timestamp")

#destination files for data and anomalies
df.to_csv('C:\\Users\\dmitr\\Desktop\\project work\\dummy data\\data.csv')
df_clean.to_csv('C:\\Users\\dmitr\\Desktop\\project work\\dummy data\\clean_data.csv')
df_anomaly.to_csv('C:\\Users\\dmitr\\Desktop\\project work\\dummy data\\data_anomaly_labels.csv')