import numpy as np
import pandas as pd
import sklearn
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from time import time

loc_input = 'clustering input.csv'
loc_output = 'clustering output.csv'

df = pd.read_csv(loc_input, sep=',', encoding='ISO-8859-1')
df = df.fillna(0) #nas probably came from divide by 0 errors during preprocessing

#set aside unused columns when getting features
unused_cols = ['player name', 'player id', 'position id']
features = [col for col in df.columns if col not in unused_cols]

n_components = 2

dimensionality_reducers = [
('PCA', PCA(n_components)),
('tSNE', TSNE(n_components))]

n_clusters = [1, 6, 8, 6]
clustering_df = pd.DataFrame()
extra = 0 #amount to shift over clusters for each position so different positions have different cluster spaces
for i in range(1, 5):
	position_df = df[df['position id'] == i]
	x = position_df[features]

	for name, reducer in dimensionality_reducers:
		t0 = time()
		transform = reducer.fit_transform(x)
		position_df[name + '1'] = transform[:,0]
		position_df[name + '2'] = transform[:,1]
		t1 = time()
		print("Dim. reduction %s: done for position %f in %f seconds" % (name, i, t1-t0))

	#do k-means on just pca
	kmeans = KMeans(n_clusters=n_clusters[i-1])
	kmeans_features = position_df[['PCA1', 'PCA2']]
	kmeans.fit(kmeans_features)
	position_df['cluster'] = kmeans.predict(kmeans_features) + extra
	extra += n_clusters[i-1]

	#append dataframe
	clustering_df = pd.concat([clustering_df, position_df], axis=0)

#write to csv
clustering_df.to_csv(loc_output, sep=',', index=False)
