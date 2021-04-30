import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

input_data = pd.read_csv("/content/sample_data/mobile_price_classification.csv")

independent = input_data.iloc[:,0:20]
dependent = input_data.iloc[:,-1]

model = ExtraTreesClassifier()
model.fit(independent,dependent)

print(model.feature_importances_)

select_feature = pd.Series(model.feature_importances_,index=independent.columns)

select_feature.nlargest(10).plot(kind='barh')
plt.show()

import seaborn as sns

correlation_matrix=input_data.corr()
corr_features=correlation_matrix.index

plt.figure(figsize=(20,20))
mat_table=sns.heatmap(input_data[corr_features].corr(),annot=True,cmap='RdYlGn')
