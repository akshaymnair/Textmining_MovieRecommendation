import pandas as pd
import actor_vector as av
import numpy as np
import math

actor_tfidf_dict = dict()

def get_tfidf(i):
	if(i not in actor_tfidf_dict):
		actor_tfidf_dict[i]=av.tfidf(i)
	return actor_tfidf_dict[i]

def cosine_similarity(list1,list2):
	dic1 = {k[0]:k[1] for k in list1}
	dic2 = {k[0]:k[1] for k in list2}
	numerator = 0.0
	dena = 0.0
	for key1,val1 in dic1.iteritems():
		numerator += val1*dic2.get(key1,0.0)
		dena += val1*val1
	denb = 0.0
	for val2 in dic2.values():
		denb += val2*val2
	if (dena!=0.0) & (denb!= 0.0):    
		return numerator/math.sqrt(dena*denb)
	return 0.0


movie_actor = pd.read_csv('../../Phase2_data/movie-actor.csv')
actor_ids = movie_actor.actorid.unique()
df = pd.DataFrame(np.zeros((len(actor_ids),len(actor_ids))), columns=actor_ids, index=actor_ids)

for i in actor_ids:
	for j in actor_ids:
		if i==j:
			df[i][j] = 0.0
		elif df[j][i] != 0.0:
			df[i][j] = df[j][i]	
		else:
			df[i][j] = cosine_similarity(get_tfidf(i),get_tfidf(j))

df.to_pickle('actor_actor_sim_matrix.pkl')
