import pandas as pd
import numpy as np
# import lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# read dataset
df = pd.read_csv('Iris.csv')
# drop id column
df.drop('Id', axis=1, inplace=True)
# drop rows with missing values
df.dropna(inplace=True)

# split dataset into train and test
X = df.drop('Species', axis=1)
y = df['Species']

# apply LDA to training and test data
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X, y)
X = lda.transform(X)

print(X)