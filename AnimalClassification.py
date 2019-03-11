import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_pickle('data.pk')

X = np.concatenate(data['image']).reshape(3000,3072)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2019)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=.66, random_state=2019)

knn = KNeighborsClassifier(n_neighbors=3, p=1)
knn.fit(X_train, y_train)