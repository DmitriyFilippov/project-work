from signal import signal
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import sys
import numpy as np

#signatures file reading
df = pd.read_csv('C:\\Users\\dmitr\\Desktop\\project work\\dummy data\\signatures_2_7.0_2.0_time__daytime_.csv')
df.head()

df = df.dropna()
signatures = df.to_numpy()
outlier_count_pd = pd.read_csv('C:\\Users\\dmitr\\Desktop\\project work\\dummy data\\signatures_2_7.0_2.0_time__daytime__ANOMALYCOUNT_.csv')
outlier_count = outlier_count_pd['count'].to_numpy()
labels = [outlier_count[i] > 0 for i in range(len(outlier_count))]
dim = len(signatures[0])

def evaluate(estimate, label):
      return roc_auc_score(label, estimate)


#MAH distance estimation
def mah_distance():

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

      certainty = [chi2.cdf(distances[i], df.shape[1]) for i in range(len(distances))]
      print("evaluate mah")
      print(evaluate(certainty, labels))

# MAH distance with robust covariance estimation

# fit a MCD robust estimator to data
robust_cov = MinCovDet().fit(signatures).covariance_ + np.identity(dim) * 1e-6
robust_cov_pm1 = np.linalg.matrix_power(robust_cov, -1)
# fit a MLE estimator to data
emp_cov = EmpiricalCovariance().fit(signatures).covariance_ + np.identity(dim) * 1e-6
emp_cov_pm1 = np.linalg.matrix_power(emp_cov, -1)

robust_distances = []
for i, val in enumerate(signatures):
      p1 = val
      p2 = centerpoint
      distance = (p1-p2).T.dot(robust_cov_pm1).dot(p1-p2)
      robust_distances.append(distance)
robust_distances = np.array(robust_distances)
robust_certainty = [chi2.cdf(robust_distances[i], df.shape[1]) for i in range(len(robust_distances))]

emp_distances = []
for i, val in enumerate(signatures):
      p1 = val
      p2 = centerpoint
      distance = (p1-p2).T.dot(emp_cov_pm1).dot(p1-p2)
      distances.append(distance)
emp_distances = np.array(emp_distances)
emp_certainty = [chi2.cdf(emp_distances[i], df.shape[1]) for i in range(len(emp_distances))]

print("evaluate robust mah")
print(evaluate(robust_certainty , labels))

print("evaluate emp mah")
print(evaluate(emp_certainty , labels))





mah_distance()