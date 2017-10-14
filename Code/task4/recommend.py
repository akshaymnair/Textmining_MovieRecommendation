import pandas as pd
import pickle

def save_obj(obj, filename):
	with open(filename, 'wb') as output:
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)



df_movies = pd.read_csv('../../Phase2_data/mlmovies.csv')
df_tags = pd.read_csv('../../Phase2_data/genome-tags.csv')
df_users = pd.read_csv('../../Phase2_data/mlusers.csv')
df_actors = pd.read_csv('../../Phase2_data/imdb-actor-info.csv')

movies = df_movies.movieid.unique().tolist()
tags = df_tags.tagId.unique().tolist()
users = df_users.userid.unique().tolist()
actors = df_actors.id.unique().tolist()

movie_genre = pd.DataFrame(df_movies.genres.str.split('|').tolist(), index = df_movies.movieid).stack()
movie_genre = movie_genre.reset_index()[[0,'movieid']]
movie_genre.columns = ['genres','movieid']
genres = movie_genre.genres.unique().tolist()

user_dict = dict()
df_mlratings = pd.read_csv('../../Phase2_data/mlratings.csv')
df_mltags = pd.read_csv('../../Phase2_data/mltags.csv')
for row in df_mlratings.iterrows():
	
	if row[1]['movieid'] in user_dict:
		user_dict[row[1]['movieid']].append(row[1]['userid'])
	else:
		user_dict[row[1]['movieid']] = [row[1]['userid']]
print user_dict

def check_tag(tags, mid):
	df_mltags = pd.read_csv('../../Phase2_data/mltags.csv')
	values = []
	for tag in tags:
		if((df_mltags[['movieid','tagid']].values == [mid, tag]).all(axis=1).any()):
			values.append(1)
		else:
			values.append(0)
	return values

def check_user(users, mid):
	df_mlratings = pd.read_csv('../../Phase2_data/mlratings.csv')
	df_mltags = pd.read_csv('../../Phase2_data/mltags.csv')
	values = []
	for user in users:
		if(user in user_dict[mid]):
			values.append(1)
		else:
			values.append(0)
	return values

def check_genre(genres, mid):
	values = []
	for genre in genres:
		if((movie_genre[['movieid','genres']].values == [mid, genre]).all(axis=1).any()):
			values.append(1)
		else:
			values.append(0)
	return values

def check_actor(actors, mid):
	df_mactors = pd.read_csv('../../Phase2_data/movie-actor.csv')
	values = []
	for actor in actors:
		if((df_mactors[['movieid','actorid']].values == [mid, actor]).all(axis=1).any()):
			values.append(1)
		else:
			values.append(0)
	return values



print len(movies), len(tags), len(users), len(genres), len(actors)
print type(tags)
cols = tags+users+genres+actors
print len(cols)
df_movie_final = pd.DataFrame(columns=cols)

for movie_id in movies:
	tag_values = check_tag(tags, movie_id)
	genre_values = check_genre(genres, movie_id)
	user_values = check_user(users, movie_id)
	actor_values = check_actor(actors, movie_id)
	df_movie_final.loc[id] = tag_values + user_values + genre_values + actor_values
print df_movie_final
save_obj(df_movie_final, 'movie_final.pkl')