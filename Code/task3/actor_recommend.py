import pandas as pd
import actor_vector as av
import numpy as np
import math
import sys

df = pd.read_pickle('actor_actor_sim_matrix.pkl')
#Normalize the matrix to sum upto 1. This becomes the transition probability then.
df_norm = df.div(df.sum(axis=1), axis=0)
df = df_norm.fillna(0.0)
df_norm = df

seed_actors = sys.argv[1].split(',')
seed_actors = list(map(int, seed_actors))

#Sample seed_actors = (1014988,1342347,1698048)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
movie_actor = pd.read_csv('../../Phase2_data/movie-actor.csv')
actor_ids = movie_actor.actorid.unique()
#Intialize PageRank values to 1.0
pr = pd.DataFrame(1.0, columns=actor_ids, index=('PageRank',))

for i in range(0,300):
	pr_new = pr.copy()
	#Update page rank
	for i in actor_ids:
		if i in seed_actors:
			pr_new[i] = (0.15/len(seed_actors)) + .85*df_norm[i].dot(pr.loc['PageRank'])
		else:
			pr_new[i] = .85*df_norm[i].dot(pr.loc['PageRank'])
	pr = pr_new.copy()

#Remove seeded actors from list
for i in seed_actors:
	try:
		pr_final = pr.drop(i ,axis=1)
		pr = pr_final
	except:
		pass

#Display top 10 related actors
print (pr.loc['PageRank'].nlargest(10))
