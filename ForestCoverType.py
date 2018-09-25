import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset - The datasets can be found here https://www.kaggle.com/c/forest-cover-type-kernels-only/data
dataset = pd.read_csv('../input/train.csv')
dataset_sub = pd.read_csv('../input/test.csv')
X = dataset.iloc[:, 1:55].values
y = dataset.iloc[:, 55].values
X_sub = dataset_sub.iloc[:, 1:55].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_sub = sc.transform(X_sub)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
score =  accuracy_score(y_test, y_pred)
print(score)

y_pred_sub = classifier.predict(X_sub)
sub = pd.read_csv('../input/sample_submission.csv')
sub['Cover_Type'] = y_pred_sub

sub.to_csv('sample_submission.csv',index = False)
