import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# read dataset
df = pd.read_csv('CC GENERAL.csv')
# drop CUST_ID column
df.drop('CUST_ID', axis=1, inplace=True)
# drop rows with missing values
df.dropna(inplace=True)

# split dataset into train and test
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

# scale fit training data
scaler = StandardScaler()
scaler.fit(X_train)

# apply transform to training and test data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Apply k-means algorithm on the original data
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_train)
sil_original = silhouette_score(X_train, y_pred)
print('Silhouette score for k-means on original data: ', sil_original)

# apply PCA to training and test data
pca = PCA(n_components=2)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_train)
sil_pca = silhouette_score(X_train, y_pred)
print('Silhouette score for k-means on PCA result: ', sil_pca)


print('Silhouette score for k-means on original data is ', sil_original, ' and silhouette score for k-means on PCA result is ', sil_pca)
if(sil_pca > sil_original):
    print('Silhouette score has improved')
else:
    print('Silhouette score has not improved')
    
# report performance on test data
y_pred = kmeans.predict(X_test)
sil_test = silhouette_score(X_test, y_pred)
print('Silhouette score for k-means on test data: ', sil_test)



