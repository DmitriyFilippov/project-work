from signal import signal
import pandas as pd
from scipy.stats import chi2
import sys
import numpy as np

#signatures file reading
df = pd.read_csv('C:\\Users\\dmitr\\Desktop\\project work\\dummy data\\signatures_3_7.0_1.0_time__daytime_.csv')
df.head()

df = df.dropna()
signatures = df.to_numpy()
dim = len(signatures[0])
# Covariance matrix
covariance  = np.cov(signatures , rowvar=False) + np.identity(dim) * 1e-6
# Covariance matrix power of -1
covariance_pm1 = np.linalg.matrix_power(covariance, -1)

# Center point
centerpoint = np.mean(signatures , axis=0)

distances = []
for i, val in enumerate(signatures):
      p1 = val
      p2 = centerpoint
      distance = (p1-p2).T.dot(covariance_pm1).dot(p1-p2)
      distances.append(distance)
distances = np.array(distances)

# Cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers 
cutoff = chi2.ppf(0.999999, df.shape[1])

# Index of outliers
outlierIndexes = np.where(distances > cutoff )

print('--- Index of Outliers ----')
print(outlierIndexes)

#print('--- Observations found as outlier -----')
#print(df[ distances > cutoff , :])