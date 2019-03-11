# Import code adapted from https://stackoverflow.com/a/32850511

import numpy as np
import pandas as pd
import codecs, json

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

results = pd.DataFrame()

X_train_text = codecs.open('X_train.json', 'r', encoding='utf-8').read()
X_train_json = json.loads(X_train_text)
X_train = np.array(X_train_json)

y_train_text = codecs.open('y_train.json', 'r', encoding='utf-8').read()
y_train_json = json.loads(y_train_text)
y_train = np.array(y_train_json)

X_test_text = codecs.open('X_test.json', 'r', encoding='utf-8').read()
X_test_json = json.loads(X_test_text)
X_test = np.array(X_test_json)

y_test_text = codecs.open('y_test.json', 'r', encoding='utf-8').read()
y_test_json = json.loads(y_test_text)
y_test = np.array(y_test_json)

k = 44
p = 1

print('Training k=' + str(k) + ', p=' + str(p) + '...\n')

knn = KNeighborsClassifier(n_neighbors=k, p=p)
knn.fit(X_train, y_train)

print('Testing k=' + str(k) + ', p=' + str(p) + '...\n')

y_test_pred = knn.predict(X_test)
test_score = accuracy_score(y_test, y_test_pred)
test_confusion_matrix = confusion_matrix(y_test, y_test_pred) 


results['k'] = [k]
results['p'] = [p]
results['Test Score'] = [test_score]
results['Test Confusion Matrix'] = [test_confusion_matrix]

print(results)

validation_results_1 = pd.read_pickle('validation_results_1-30.pk')
validation_results_2 = pd.read_pickle('validation_results_30-50.pk')
comb_val_results = pd.concat([validation_results_1, validation_results_2]).sort_values(by=['Validation Score'], ascending=False)
print(comb_val_results)