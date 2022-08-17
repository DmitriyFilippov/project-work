import pandas as pd 
import random

hour = 3600
day = 24 * hour

#distances between airports in time
times = [
[0, 6*hour, 0, 0],
[6*hour, 0, 4*hour, 4*hour],
[0, 4*hour ,0, 3*hour],
[0, 4*hour, 0, 3*hour]
]

file_location = 'C:\\Users\\dmitr\\Desktop\\project work\\dummy data'
#edges present in the graph
edges = [(0,1), (1,2), (1,3), (2,3), (1,0), (2,1), (3,1), (3,2)]
c1_chaining_prob = [
[0, 0.05, 0.05, 0, 0, 0, 0, 0, 0.9],
[0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0.1, 0, 0, 0, 0.9],
[0, 0, 0, 0, 0.1, 0, 0, 0, 0.9],
[0, 0, 0, 0, 0, 0, 0, 0, 1],
]
c1_edge_weights = (2, 1, 1, 2, 2, 1, 1, 2)

c2_chaining_prob = [
[0, 0.25, 0.25, 0, 0, 0, 0, 0, 0.5],
[0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 0, 0.5, 0, 0, 0, 0.5],
[0, 0, 0, 0, 0.5, 0, 0, 0, 0.5],
[0, 0, 0, 0, 0, 0, 0, 0, 1],
]
c2_edge_weights = (2, 0.5, 0.5, 2, 1, 1, 1, 2)
#delay between chain flights in 1st kind of anomaly
chaining_delay_left = int(hour * 0.8)
chaining_delay_right = int(hour * 1.2)

#random variation in flight length
flight_len_spread = 0.2

flight_spread = 169
flight_spread_variation = 60


data_c1 = []
timestamp = 0
id = 0
for i in range(500000):
    timestamp += (flight_spread + random.randrange(-flight_spread_variation, flight_spread_variation))   #random delay between flights
    [edge] = random.choices(range(len(edges)), weights = c1_edge_weights)
    (a, b) = edges[edge]
    delay = times[a][b] * (random.random() * flight_len_spread + (1 - flight_len_spread/2)) #length of the flight
    data_c1.append([a, timestamp, "takeoff", id])
    data_c1.append([b, timestamp+delay, "landed", id])
    id += 1

    chaining = random.choices(range(len(edges) + 1), weights = c1_chaining_prob[edge])[0]
    if chaining < len(edges):
        (a, b) = edges[chaining]
        delay = times[a][b] * (random.random() * flight_len_spread + (1 - flight_len_spread/2)) #length of the flight
        data_c1.append([a, timestamp, "takeoff", id])
        data_c1.append([b, timestamp+delay, "landed", id])
        id += 1

data_c2 = []
timestamp = 0
id = 0
for i in range(500000):
    timestamp += (flight_spread + random.randrange(-flight_spread_variation, flight_spread_variation))   #random delay between flights
    [edge] = random.choices(range(len(edges)), weights = c2_edge_weights)
    (a, b) = edges[edge]
    delay = times[a][b] * (random.random() * flight_len_spread + (1 - flight_len_spread/2)) #length of the flight
    data_c2.append([a, timestamp, "takeoff", id])
    data_c2.append([b, timestamp+delay, "landed", id])
    id += 1

    chaining = random.choices(range(len(edges) + 1), weights = c2_chaining_prob[edge])[0]
    if chaining < len(edges):
        (a, b) = edges[chaining]
        delay = times[a][b] * (random.random() * flight_len_spread + (1 - flight_len_spread/2)) #length of the flight
        data_c2.append([a, timestamp, "takeoff", id])
        data_c2.append([b, timestamp+delay, "landed", id])
        id += 1


df_c1 = pd.DataFrame(data_c1, columns = ["icao", "timestamp", "event", "id"])
df_c1 = df_c1.sort_values("timestamp")

df_c2 = pd.DataFrame(data_c2, columns = ["icao", "timestamp", "event", "id"])
df_c2 = df_c1.sort_values("timestamp")

#destination files for data and anomalies
df_c1.to_csv(file_location + '\\data_c1.csv')
df_c2.to_csv(file_location + '\\data_c2.csv')