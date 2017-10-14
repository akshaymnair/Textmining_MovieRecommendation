import pickle
from sklearn.metrics import pairwise_distances as Distance
import pandas as pd

def load_obj(file):
	with open(file,'r') as input:
		obj = pickle.load(input)
	return obj

df = load_obj('movie_final.pkl')

movie_ids = [3323, 3467, 4354, 5000, 3366]

# Calculate cosine distance between movies watched by the user and all the other movies and store it in a matrix D
X = df.loc[movie_ids]
Y = df
D = Distance(X,Y, metric='cosine')


D_sum = D.sum(axis = 0)


index_order = D_sum.argsort() 

count = 0 
recommend_movie_ids = []
for index1 in index_order:
	if(df.index[index1] not in movie_ids):
		recommend_movie_ids.append(df.index[index1])
		count = count + 1
	if(count==5):
		break

print '--------------Movies watched by users----------------'

df_movies = pd.read_csv('../../Phase2_data/mlmovies.csv')

for mid in  movie_ids:
	print df_movies.loc[df_movies['movieid']==mid,'moviename'].iloc[0]

print '---------------Recommended movies--------------------'

for mid in  recommend_movie_ids:
	print df_movies.loc[df_movies['movieid']==mid,'moviename'].iloc[0]