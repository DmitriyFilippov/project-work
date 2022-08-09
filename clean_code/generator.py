import pandas as pd 
import numpy as np 
import esig 
import random

hour = 3600
day = 24 * hour

#distances between airports in time
times = [
[0, 3*hour, 0, 0, 4*hour],
[3*hour, 0, 6*hour, 4*hour, 0],
[0, 6*hour ,0, 0, 0],
[0, 4*hour, 0, 0, 7*hour],
[4*hour, 0, 0, 7*hour, 0]
]

file_location = 'C:\\Users\\dmitr\\Desktop\\project work\\dummy data'
#edges present in the graph
edges = [(0,1), (0,4), (1, 0), (1, 2), (1, 3), (2, 1), (3, 1), (3, 4), (4, 0), (4, 3)]

#weights for regulating edge relative frequency in 2nd type of anomaly
anomaly_weights = (1.5, 0.5, 1.5, 1, 0.5, 1, 0.5, 1.5, 0.5, 1.5)
#delay between chain flights in 1st kind of anomaly
anomaly_chaining_delay_left = 10800
anomaly_chaining_delay_right = 12000

#random variation in flight length
flight_len_spread = 0.2

data = []
clean_data = []
#timestamp is the running record of current timestamp
timestamp = 0
timestamp_clean = 0
anomaly_labels = []

for i in range(500000):
    timestamp_clean += (169 + random.randrange(-60, 60))   #random delay between flights
    (a, b) = random.choice(edges)
    delay = times[a][b] * (random.random() * flight_len_spread + (1 - flight_len_spread/2)) #length of the flight
    clean_data.append([a, timestamp_clean, "departed"])
    clean_data.append([b, timestamp_clean+delay, "landed"])

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
    if(a == 0 and b == 1):                     #chaining only applies to this pair
        delay2 = random.randrange(anomaly_chaining_delay_left, anomaly_chaining_delay_right)               #delay between chained flights
        delay3 = 6 * 3600 * (random.random() * flight_len_spread + (1 - flight_len_spread/2))              #length of a second flight
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
df.to_csv(file_location + '\\data.csv')
df_clean.to_csv(file_location + '\\clean_data.csv')
df_anomaly.to_csv(file_location + '\\data_anomaly_labels.csv')