import pandas as pd


df_movies = pd.read_csv('../../Phase2_data/mlmovies.csv')
df_tags = pd.read_csv('../../Phase2_data/genome-tags.csv')
df_users = pd.read_csv('../../Phase2_data/mlusers.csv')
df_actors = pd.read_csv('../../Phase2_data/movie-actor.csv')

movies = df_movies.movieid.unique().tolist()
tags = df_tags.tagId.unique().tolist()
users = df_users.userid.unique().tolist()
actors = df_actors.actorid.unique().tolist()

movie_genre = pd.DataFrame(df_movies.genres.str.split('|').tolist(), index = df_movies.movieid).stack()
movie_genre = movie_genre.reset_index()[[0,'movieid']]
movie_genre.columns = ['genres','movieid']
genres = movie_genre.genres.unique().tolist()

def check_tags(tags, mid):
	df_mltags = pd.read_csv('../../Phase2_data/mltags.csv')
	tag_values = []
	for tag in tags:
		if((df_mltags[['movieid','tagid']].values == [mid, tag]).all(axis=1).any()):
			tag_values.append(1)
		else:
			tag_values.append(0)
	return tag_values

def check_users(users, mid):
	df_mltags = pd.read_csv('../../Phase2_data/mltags.csv')
	tag_values = []
	for tag in tags:
		if((df_mltags[['movieid','tagid']].values == [mid, tag]).all(axis=1).any()):
			tag_values.append(1)
		else:
			tag_values.append(0)
	return tag_values

def check_tags(tags, mid):
	df_mltags = pd.read_csv('../../Phase2_data/mltags.csv')
	tag_values = []
	for tag in tags:
		if((df_mltags[['movieid','tagid']].values == [mid, tag]).all(axis=1).any()):
			tag_values.append(1)
		else:
			tag_values.append(0)
	return tag_values

def check_tags(tags, mid):
	df_mltags = pd.read_csv('../../Phase2_data/mltags.csv')
	tag_values = []
	for tag in tags:
		if((df_mltags[['movieid','tagid']].values == [mid, tag]).all(axis=1).any()):
			tag_values.append(1)
		else:
			tag_values.append(0)
	return tag_values



print len(movies), len(tags), len(users), len(genres), len(actors)
print type(tags)
cols = tags+users+genres+actors
print len(cols)
df_movie_final = pd.DataFrame(columns=cols)

for movie_id in movies:
	tag_values = check_tags(tags, movie_id)
	#genre_values = check_genre(genres, movie_id)
	#user_values = check_user(users, movie_id)
	#actor_values = check_actor(actors, movie_id)
	#df_movie_final.loc[id] = []
