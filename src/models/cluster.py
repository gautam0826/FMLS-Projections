import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from time import time
pd.options.mode.chained_assignment = None  #gets rid of SettingWithCopyWarnings

loc_input = '2018_clustering_input.csv'
loc_output = '2018_clustering_output.csv'

df = pd.read_csv(os.path.join('..', '..', 'data', 'processed', loc_input), sep=',', encoding='ISO-8859-1')
df = df.fillna(0) #nas probably came from divide by 0 errors during preprocessing
df_unused = df[df['games'] < 5]
df = df[df['games'] >= 5]

#set aside unused columns when getting features
unused_cols = ['player name', 'player id', 'position id', 'season', 'games']
features = [col for col in df.columns if col not in unused_cols and 'std' not in col]

n_components = 2

dimensionality_reducers = [
#('PCA', PCA(n_components)),
('tSNE', TSNE(n_components)),
('ICA', FastICA(n_components))]

n_clusters = [1, 6, 8, 6]
#clustering = [
#(GaussianMixture(n_components=6, covariance_type='full'))
#]
extra = 1 #amount to shift over clusters for each position so different positions have different cluster spaces
for i in range(1, 5):
	x = df.loc[df['position id'] == i][features]  

	for name, reducer in dimensionality_reducers:
		t0 = time()
		transform = reducer.fit_transform(x)
		df.loc[df['position id'] == i, name + '1'] = transform[:,0]
		df.loc[df['position id'] == i, name + '2'] = transform[:,1]
		t1 = time()
		print("Dim. reduction %s: done for position %f in %f seconds" % (name, i, t1-t0))

	#do clustering on just ica for only field players(exclude goalkeepers)
	if i != 1:
		kmeans = GaussianMixture(n_components=n_clusters[i-1], covariance_type='full')#KMeans(n_clusters=n_clusters[i-1])
		kmeans_features = df.loc[df['position id'] == i][['ICA1', 'ICA2']]#StandardScaler().fit_transform()#
		kmeans.fit(kmeans_features)
		df.loc[df['position id'] == i, 'cluster'] = kmeans.predict(kmeans_features) + extra

	#increment extra so no overlap between different position clusters
	extra += n_clusters[i-1] + 1

	#low gametime players get their own cluster
	df_unused.loc[df_unused['position id'] == i, 'cluster'] = extra - 1

#add in barely used players
df = pd.concat([df, df_unused], axis=0, sort=True)

#assign gk clusters
df.loc[df['position id'] == 1, 'cluster'] = 0

#write to csv
df.to_csv(os.path.join('..', '..', 'data', 'processed', loc_output), sep=',', index=False)