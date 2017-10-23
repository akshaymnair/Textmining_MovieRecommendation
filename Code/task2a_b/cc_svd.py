#CSE515 Project Phase2
#Author: Akshay

import pandas as pd
from sklearn.decomposition import TruncatedSVD
import sys
import os
import util
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


output_file = 'task2b_C_C_out.txt'
task = 'Coactor -> Coactor Similarity'
no_of_components = 3
actor_info = util.read_actor_info()

#Coactor Coactor similarity matrix
cc_matrix = pd.read_pickle('../task3/coactor_matrix.pkl')
actorid_list = list(cc_matrix.columns.values)
actor_list = actor_info[actor_info['id'].isin(actorid_list)]['name'].tolist()
#pd.set_option('display.max_rows', 500)
#print cc_matrix

svd = TruncatedSVD(n_components=no_of_components, n_iter=100, random_state=None)
svd.fit(cc_matrix)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd,normalizer)

dataC = lsa.fit_transform(cc_matrix)
#print dataC

concepts = []
for i in range(no_of_components):
	concept = []
	for j, component in enumerate(svd.components_[i]):
		concept.append((actor_list[j], component))
	concept.sort(key=lambda tup:tup[1], reverse=True)
	concepts.append(concept)
util.print_output(task, concepts)
util.write_output_file(task, concepts, output_file)

cluster_rule = KMeans(n_clusters=3)
cluster_rule.fit(dataC)

labels = cluster_rule.predict(dataC)
centroids = cluster_rule.cluster_centers_

#print labels
print('\n')
print("Centroids of 3 new clusters: \n")
print(centroids)
print('\n')
print("Clustered actors into the 3 groups (0 : 1 :2) \n")
for i, j in zip(actor_list, labels):
	print ('% 50s\t is in Cluster: %10s\t' %(i, j))
util.append_output_file(actor_list, centroids, labels,output_file)

#END

