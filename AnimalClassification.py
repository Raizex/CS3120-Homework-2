import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_pickle('data.pk')

X = np.array(data['image'])
y = np.array(data['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2019)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=.66, random_state=2019)
