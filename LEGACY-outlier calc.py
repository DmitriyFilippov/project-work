from signal import signal
import pandas as pd
from scipy.stats import chi2
import sys
import numpy as np


signatures = pd.read_csv('C:\\Users\\dmitr\\Desktop\\project work\\dummy data\\signatures.csv')


means = []

for i in range(57):
    means.append(signatures[str(i)].mean())

diviations = []

for index, row in signatures.iterrows():
    diviation = 0
    for i in range(57):
        val = row[str(i)]
        mean =  means[i]
        diviation += abs((val - mean)/mean)
    diviations.append((diviation, index))

diviations.sort()
print(diviations)