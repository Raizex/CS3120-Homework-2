import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

neighbors = list(range(1,30))
cv_train_scores = []
cv_valid_scores = []
data = pd.read_pickle('data.pk')

X = np.concatenate(data['image']).reshape(3000,3072)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2019)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=.66, random_state=2019)

for k in neighbors:
    
    knn = KNeighborsClassifier(n_neighbors=k, p=1)
    knn.fit(X_train, y_train)

    y_train_pred = knn.predict(X_train)
    cv_train_scores.append(accuracy_score(y_train,
                                          y_train_pred))

    y_valid_pred = knn.predict(X_valid)
    cv_valid_scores.append(accuracy_score(y_valid,
                                          y_valid_pred))

print(cv_train_scores)
print(cv_valid_scores)