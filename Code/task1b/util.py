from datetime import datetime
import pandas as pd
from numpy import log
import os

db_folder_path = os.path.join(os.path.dirname(__file__), "..", "..", "Phase2_data")
output_folder = os.path.join(os.path.dirname(__file__), "..", "..", "Output")

mlmovies_file = 'mlmovies.csv'
movie_actor_file = 'movie-actor.csv'
imdb_actor_info_file = 'imdb-actor-info.csv'


############### HELPER FUNCTION TO READ FILES #################################

# import mlmovies
def read_mlmovies():
	return pd.read_csv(os.path.abspath(os.path.join(db_folder_path, mlmovies_file)))

# import imdb-actor-info
def read_imdb_actor_info():
	return pd.read_csv(os.path.abspath(os.path.join(db_folder_path, imdb_actor_info_file)))

# import movie-actor
def read_movie_actor():
	movie_actor = pd.read_csv(os.path.abspath(os.path.join(db_folder_path, movie_actor_file)))
	for i,row in movie_actor.iterrows():
		movie_actor.iloc[i,2] = 1/float(row['actor_movie_rank'])
	return movie_actor

################## HELPER FUNCTION TO PROCESS AND RETRIEVE ######################

def get_movie_genre(mlmovies):
	movie_genre = pd.DataFrame(mlmovies.genres.str.split('|').tolist(), index = mlmovies.movieid).stack()
	movie_genre = movie_genre.reset_index()[[0,'movieid']]
	movie_genre.columns = ['genre','movieid']
	return movie_genre

def filter_by_genre(movie_actor, mlmovies, genre):
	movie_genre = get_movie_genre(mlmovies)
	movieids_of_genre = movie_genre.where(movie_genre['genre']==genre).dropna().loc[:,'movieid'].unique()
	actors_of_genre = movie_actor.where(movie_actor['movieid'].isin(movieids_of_genre)).dropna()
	genre_movie_count = int(movieids_of_genre.shape[0])

	return (genre_movie_count, actors_of_genre)

def get_tf_idf_matrix(genre):
	movie_actor = read_movie_actor()
	mlmovies = read_mlmovies()
	movie_genre = get_movie_genre(mlmovies)
	genre_movie_count, actors_of_genre = filter_by_genre(movie_actor, mlmovies, genre)

	# Needed for TF denominators. Calcuate total number of actors per movie
	actors_per_movie = actors_of_genre.groupby('movieid', as_index=False)['actorid'].agg({'m_count' : pd.Series.count})

	# Needed for IDF denonminators. Calculate unique movieids for each actor
	movies_per_actor = actors_of_genre.groupby('actorid', as_index=False)['movieid'].agg({'a_count' : pd.Series.nunique})

	# Grouped so as to create unique actorid, movieid pairs and calculate term frequency from actor_movie_rank
	actor_movieid_grouped = actors_of_genre.groupby(['actorid', 'movieid'], as_index=False)['actor_movie_rank'].agg({'tf': 'sum'})

	# Merge actor_counts. Add new column including calculated actors_per_movie.
	M1 = pd.merge(actor_movieid_grouped, actors_per_movie, on=['movieid','movieid'], how='inner')

	# Merge movie_counts. Add new column including calculated movies_per_actor.
	M2 = pd.merge(M1, movies_per_actor, on=['actorid', 'actorid'], how = 'inner')

	# Perform TF-IDF from the data.
	M2['tfidf'] = M2['tf']*log(genre_movie_count/M2['m_count'])/M2['a_count']
	#print M2

	# Pivot the matrix to get in required form 
	R = M2.pivot(index='movieid', columns='actorid', values='tfidf').fillna(0)
	#print R
	return R

def print_output(genre, concepts):
	print('For genre: ' + genre + ', output is :')
	for idx, concept in enumerate(concepts):
		print ('\nThe ' + str(idx+1) +'th concept is: ')
		print('%40s\t%15s\t' %('Actor', 'Weight'))
		for row in concept:
			print ('%40s\t%15s\t' %(row[0], row[1]))

def write_output_file(genre, concepts, filename):
	f = open(os.path.abspath(os.path.join(output_folder, filename)),'w')
	f.write('For genre: ' + genre + ', output is :' + '\n')
	for idx, concept in enumerate(concepts):
		f.write('\nThe ' + str(idx+1) +'th concept is: ')
		f.write('\n%40s\t%15s\t' %('Actor', 'Weight'))
		for row in concept:
			f.write('\n%40s\t%15s\t' %(row[0], row[1]))
	f.close()

