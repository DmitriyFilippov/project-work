import pandas as pd
from datetime import datetime

def datetime_convert_origin(row):
    date = row['date_x']
    time = row['time_x']
    date = date[2:]
    date_time = date + " " + time
    date_time_obj = datetime.strptime(date_time, '%y-%m-%d %H:%M:%S')
    return str(int(round(date_time_obj.timestamp())))

def datetime_convert_destination(row):
    date = row['date_y']
    time = row['time_y']
    date = date[2:]
    date_time = date + " " + time
    date_time_obj = datetime.strptime(date_time, '%y-%m-%d %H:%M:%S')
    return str(int(round(date_time_obj.timestamp())))

df = pd.read_csv('C:/Users/dmitr/Desktop/project work/air_graph.csv')
holding = df.loc[df['event'] == 'Diverting']
takeoff = df.loc[df['event'] == 'takeoff']
landed =  df.loc[df['event'] == 'landed']
flights = pd.merge(takeoff, landed, on='flight_id')
flights['origin_time'] = flights.apply(lambda row: datetime_convert_origin(row), axis = 1)
flights['destination_time'] = flights.apply(lambda row: datetime_convert_destination(row), axis = 1)
edges = flights[['origin_icao_x', 'destination_icao_x', 'origin_time', 'destination_time']]

edges.rename(columns = {'origin_icao_x' : 'origin_icao', 'destination_icao_x' : 'destination_icao'})
print(holding.to_string()) 