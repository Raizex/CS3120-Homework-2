import pandas as pd
import numpy as np
import codecs, json

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

neighbors = list(range(1,2))
cv_train_scores = []
cv_valid_scores = []
cv_train_confusion_matrices = []
cv_valid_confusion_matrices = []
data = pd.read_pickle('data.pk')

X = np.concatenate(data['image']).reshape(3000,3072)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2019)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=.66, random_state=2019)

results = pd.DataFrame()

P = []
K = []

for p in [1,2]:
    for k in neighbors:
        
        P.append(p)
        K.append(k)
    
        knn = KNeighborsClassifier(n_neighbors=k, p=p)
        knn.fit(X_train, y_train)

        y_train_pred = knn.predict(X_train)
        cv_train_scores.append(accuracy_score(y_train,
                                              y_train_pred))
        cv_train_confusion_matrices.append(confusion_matrix(y_train,
                                           y_train_pred))

        y_valid_pred = knn.predict(X_valid)
        cv_valid_scores.append(accuracy_score(y_valid,
                                              y_valid_pred))
        cv_valid_confusion_matrices.append(confusion_matrix(y_valid,
                                           y_valid_pred))

results['P'] = P
results['K'] = K
results['Training Score'] = cv_train_scores
results['Validation Score'] = cv_valid_scores
results['Training Matrix'] = cv_train_confusion_matrices
results['Validation Matrix'] = cv_valid_confusion_matrices

results.to_pickle('validation_results.pk')
print(results)

json.dump(X_test.tolist(), codecs.open('X_test.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=False, indent=2)
json.dump(y_test.tolist(), codecs.open('y_test.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=False, indent=2)