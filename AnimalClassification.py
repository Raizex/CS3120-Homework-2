import pandas as pd

data = pd.read_pickle('data.pk')

image = data.at[0,'image']

print(image.shape)