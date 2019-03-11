import pandas as pd

results = pd.read_pickle('validation_results.pk')
sorted_results = results.sort_values(by=['Validation Score'], ascending=False)
top_results = sorted_results[['K','P','Validation Score', 'Validation Matrix']].head()
print(top_results)
top_results.to_pickle('validation_results_1-30.pk')
test_results = pd.read_pickle('validation_results_1-30.pk')
print(test_results)