import pandas as pd
import numpy as np
import math

def coactor_value(i,j):
	return 1

movie_actor = pd.read_csv('../../Phase2_data/movie-actor.csv')
actor_ids = movie_actor.actorid.unique()
df = pd.DataFrame(np.zeros((len(actor_ids),len(actor_ids))), columns=actor_ids, index=actor_ids)

for i in (607316,607316):
	for j in actor_ids:
		if i==j:
			df[i][j] = 0
		elif df[j][i] != 0.0:
			df[i][j] = df[j][i]	
		else:
			df[i][j] = coactor_value(i,j)


print df[607316]
