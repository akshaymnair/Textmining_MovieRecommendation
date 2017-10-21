import pandas as pd
import numpy as np
import math

def coactor_value(i,j):
	movie_actor1 = movie_actor[(movie_actor['actorid']==i)].dropna()
	movie_actor2 = movie_actor.where(movie_actor['movieid'].isin(movie_actor1.loc[:,'movieid']) ).dropna()
	movie_actor2 = movie_actor2.where(movie_actor2['actorid'] == j).dropna()
	return len(movie_actor2)

movie_actor = pd.read_csv('../../Phase2_data/movie-actor.csv')
actor_ids = movie_actor.actorid.unique()
df = pd.DataFrame(np.zeros((len(actor_ids),len(actor_ids))), columns=actor_ids, index=actor_ids)

for i in actor_ids:
	for j in actor_ids:
		if i==j:
			df[i][j] = 0
		elif df[j][i] != 0.0:
			df[i][j] = df[j][i]	
		else:
			df[i][j] = coactor_value(i,j)

df.to_pickle('coactor_matrix.pkl')
